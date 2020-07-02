'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import collections

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import functional as F
import models
from utils import progress_bar
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# pylint: disable=invalid-name,redefined-outer-name,global-statement

model_names = sorted(name for name in models.__dict__ if not name.startswith(
    "__") and callable(models.__dict__[name]))
best_acc = 0 # best test accuracy
k = 0
parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
parser.add_argument('--data',  default='data/cinic10')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: vgg16)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--F', type=str, default="0")
                    
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='disables CUDA training')                    
parser.add_argument('--save_path',type=str,default="default_path.pth")
parser.add_argument('--pretrained_path',type=str,default="")
parser.add_argument('--resume',type=str,default="")
parser.add_argument('--tb_filename', type=str)
parser.add_argument('--det_from_epoch', type=int,default=1000)
parser.add_argument('--hs', type=float,default=1)
parser.add_argument('--det', type=bool, default=False)
parser.add_argument('--neg', type=bool, default=False)
parser.add_argument('--no_norm', type=bool, default=True)
parser.add_argument('--extra_transform', type=bool, default=False)
parser.add_argument('--acc_loss', type=bool, default=False)
parser.add_argument('--epochs2change', type=str, default='',help="please enter epochs a following:120,140,160. Please enter no more then 4 epochs")
parser.add_argument('--factors2change', type=str, default='',help="please enter factors the following way:1,0,0.2. Please enter no more then 4 factor, each 2 facors is a for one phase")
parser.add_argument('--seed', type=int,default=60)

def max_softmax_pred(outputs,targets,wrong=True):
    y_soft = F.softmax(outputs,dim=-1)
    target_onehot = F.one_hot(targets,10)
    my_pred_soft = target_onehot*y_soft    
    index = y_soft.max(-1, keepdim=True)[1]
    my_pred = torch.zeros_like(outputs, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)       
    x = target_onehot*my_pred#my max times the real target if its not equal the row will be all 0s
    wrong_or_right = torch.sum(x,dim=-1)    
    if wrong:
        wrong = torch.ones(wrong_or_right.shape).cuda() - wrong_or_right#correct is 0 , wrong is 1
        sum_of_wrongs = wrong.sum()
        wrong = wrong.view(my_pred_soft.shape[0],1)
        new_soft =my_pred_soft*wrong
        new_soft = torch.sum(new_soft[new_soft.sum(dim=1) != 0],dim=-1) 
    else:
        sum_of_wrongs = wrong_or_right.sum()
        wrong_or_right = wrong_or_right.view(my_pred_soft.shape[0],1)
        new_soft =my_pred_soft*wrong_or_right
        new_soft = torch.sum(new_soft[new_soft.sum(dim=1) != 0],dim=-1) 

    return new_soft
def expectation(outputs,targets):
    new_soft = max_softmax_pred(outputs,targets)
    len_of_exampale = new_soft.shape[0]
    hist = torch.histc(new_soft,10,min=0,max=1)
    hist_normalize = hist/len_of_exampale
    vec = torch.tensor([0.1,0.2,0.3,0.4,0.5,0,0,0,0,0]).cuda()
    E = vec*hist_normalize
    return E
        
def hist_of_wrong(outputs,targets):
    new_soft = max_softmax_pred(outputs,targets)

    len_of_exampale = new_soft.shape[0]
    hist = torch.histc(new_soft,10,min=0,max=1)
    hist_normalize = hist#/len_of_exampale
    return hist_normalize
def hist_of_correct(outputs,targets):
    new_soft = max_softmax_pred(outputs,targets,wrong = False)
    len_of_exampale = new_soft.shape[0]
    hist = torch.histc(new_soft,10,min=0,max=1)
    hist_normalize = hist#/len_of_exampale
    return hist_normalize   



def check_valid(epochs2change_list,factors2change_list):
    if len(epochs2change_list) == 0 or len(factors2change_list) == 0:        
        return
    elif len(epochs2change_list) > 4 or len(factors2change_list) > 4: 
        raise Exception("epochs2change or factors2change must not get more than 4 values!")
    elif len(factors2change_list) % 2 != 0: 
        raise Exception("you must follow factors2change % 2 == 0!")
    elif len(factors2change_list) == 4 and len(epochs2change_list) < 2: 
        raise Exception("your factors2change holds 4 values, but you got less than 2 phases, means epoch2change hold less than 2 values")
    elif len(factors2change_list) == 2 and len(epochs2change_list) == 0: 
        raise Exception("your factors2change holds 2 values, but you got no spaciel phase!")
def getfactors(epoch):
    #hs = hf in paper or in other code parts
    hs = args.hs
    N = args.epochs
    K = len(args.F)

    fpart = int(N*hs/K)
    if epoch < fpart:
        k = int(epoch*K/(N*hs))
    else:
        k = 1+int((epoch-fpart)*(K-1)/(N-fpart))
    alpha = 1-int(args.F[k]/0.5)    
    beta  = 1/(1-2*args.F[k]*alpha)
    """
    if not args.no_norm:
        max = abs(beta*args.F[k]**2+args.F[k]*(alpha - beta) - alpha)
    else:
        max = 1
    """
    gamma = 1#/max
    return alpha,beta,gamma




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def acc_func(pred,target,tau=1,neg=False):
    target = F.one_hot(target,10)
    #pred_soft  = F.gumbel_softmax(pred,hard=False)
    if neg:
        pred   = F.gumbel_softmax(pred,tau,hard=True)*2 - 1
    else:
        pred   = F.gumbel_softmax(pred,tau,hard=True,dim=-1)
    acc = torch.mean(torch.sum(pred*target,dim=-1))
    one_minus_acc = 1 - acc
    return one_minus_acc,pred
    
def acc_func_deter(pred,target,tau=1,neg=False):
    dim=-1
    target = F.one_hot(target,10)
    y_soft = F.softmax(pred/tau,dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(pred, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    if neg:
        y_hard = y_hard*2 - 1
    else:
        y_hard = y_hard
    ret = y_hard - y_soft.detach() + y_soft    
    acc = torch.mean(torch.sum(ret*target,dim=-1))    
    one_minus_acc = 1 - acc
    return one_minus_acc,pred 
    
def write_env(args):
    path = os.path.join(args.save_path,'env.txt')
    os.popen('cp main.py '+ args.save_path)
    os.popen('python3 collect_env.py >'+path)
    args_path = os.path.join(args.save_path,'cmdline.txt')

    args_path = os.path.join(args.save_path,'cmdline.txt')
    with open(args_path,'w+') as f:
        for k,v in args.__dict__.items():
           f.write(str(k))
           f.write(":")
           f.write(str(v))
           f.write('\n')

torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if "/" in args.save_path:
    last = args.save_path.split('/')[-1]
    args.tb_filename = os.path.join(args.save_path,last+"_tb")
else:
    args.tb_filename = os.path.join(args.save_path,args.save_path+"_tb")
    args.tb_filename = os.path.join(args.save_path,args.save_path+"_tb")
tb = SummaryWriter(args.tb_filename)

if not os.path.exists(args.save_path) and not args.evaluate:
    os.mkdir(args.save_path)
#####

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
#####
write_env(args)
############
epochs2change_list = args.epochs2change.split(',') 
factors2change_list = args.factors2change.split(',')
if epochs2change_list[0] == '' and factors2change_list[0] == '':
    epochs2change_list = []
    factors2change_list = []

check_valid(epochs2change_list,factors2change_list)
############
print('==> Creating model {}...'.format(args.arch))
model = models.__dict__[args.arch]().cuda()

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 45]) 

###########################
traindir = os.path.join(args.data, 'train_and_val')
testdir = os.path.join(args.data, 'test')
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

if args.extra_transform:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])
else:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])
print('==> Preparing data..')
trainset = datasets.ImageFolder(root=traindir, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.workers)

testset = datasets.ImageFolder(root=testdir, transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1000,
                                         shuffle=True,
                                         num_workers=args.workers)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if args.cuda:
    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()


print("Parameters:",count_parameters(model))
def train(epoch,last_max_softmax_avg):
    ''' Trains the model on the train dataset for one entire iteration '''
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if args.acc_loss:
            if args.det:
                a,gs_pred = acc_func_deter(outputs, targets,neg=args.neg)
            else:   
                a,gs_pred = acc_func(outputs, targets,neg=args.neg)

           
            main_factor,factor,normalize =  getfactors(epoch)
            if batch_idx == 0:
                print("Pay attention, we are at epoch:",epoch,",the factors are now:",main_factor,round(factor,4), "Normalization factor is:",normalize)
            loss = normalize*(main_factor*loss + factor*a)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        current_loss = train_loss/(batch_idx+1)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    epoch_loss = current_loss
    return acc,epoch_loss


##validat using test set also
def validate(epoch):
    ''' Validates the model's accuracy on validation dataset and saves if better
        accuracy than previously seen. '''
    

    global best_acc,gt_avg_correct_pred,gt_avg_wrong_pred
    model.eval()
    alpha = 0.1
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            ##################
            #outE = expectation(outputs,targets)
            #lost_of_expectations.append(outE)
            histogram = hist_of_wrong(outputs,targets)
            lost_of_tensors_for_histo.append(histogram)
            histogram_correct =hist_of_correct(outputs,targets)
            lost_of_tensors_for_histo_correct.append(histogram_correct)
            ##################
            max_softmaxs = max_softmax_pred(outputs,targets,wrong = False).mean()
            max_softmaxs_wrongs = max_softmax_pred(outputs,targets,wrong = True).mean()

            if gt_avg_correct_pred == 0:
                gt_avg_correct_pred = max_softmaxs
                gt_avg_wrong_pred = max_softmaxs_wrongs
            else:
                gt_avg_correct_pred = max_softmaxs*alpha + (1-alpha)*gt_avg_correct_pred
                gt_avg_wrong_pred = max_softmaxs_wrongs*alpha + (1-alpha)*gt_avg_wrong_pred
            ##################
            
            loss = criterion(outputs, targets)
            
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_loss = valid_loss/(batch_idx+1)
            #print('max_softmax_mean:%.2f'%(gt_avg_correct_pred))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)| MaxSoftmax: %.2f'
                         % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total,gt_avg_correct_pred))
    #print(torch.mean(torch.stack(lost_of_expectations),dim=0)[:5])

    # Save checkpoint.
    acc = 100.*correct/total
    epoch_loss = current_loss
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer' : optimizer.state_dict(),
            'best_acc' : best_acc,            
        }
        path = os.path.join(args.save_path,"state.pth")
        torch.save(state, path)
        best_acc = acc
    return acc,epoch_loss,best_acc
def plot(ep_list,wrong=True):

    length = len(ep_list)
    out_tensor = torch.stack(ep_list).cpu().numpy()
    likelihood = np.linspace(0, 1, 10)
    epoch = np.linspace(0,length , length)
    likelihood, epoch = np.meshgrid(likelihood, epoch)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if wrong:
        ax.plot_surface(likelihood,epoch , out_tensor, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')

        ax.set_title('surface');
        fig.savefig(os.path.join(args.save_path,"wrong_histo.jpg"))
    else:
        ax.plot_surface(epoch , likelihood,out_tensor, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
        ax.set_title('surface');
        fig.savefig(os.path.join(args.save_path,"correct_histo.jpg"))

def test():
    ''' Final test of the best performing model on the testing dataset. '''
    checkpoint = torch.load(args.pretrained_path)
    dict = checkpoint['model']
    new_dict = {}

    if "vgg" in args.arch:
        #print("in if")
        #stop
        new_dict = dict
        """
        for k,v in dict.items():
            print(k)

            if "module" in k:
                newk = k[15:]
            print(newk)
            new_k = "features" + newk  
            new_dict[new_k] = v
        """
        pass
    else:
        for k,v in dict.items():
            if "module" in k:
                newk = k[7:]
            new_dict[k] = v
    model.load_state_dict(new_dict)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('Test best performing model from epoch {} with accuracy {:.3f}%'.format(
        checkpoint['epoch'], checkpoint['acc']))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            histogram = hist_of_wrong(outputs,targets)
            lost_of_tensors_for_histo.append(histogram)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


start_time = datetime.now()
print('Runnning training and test for {} epochs'.format(args.epochs))



if args.evaluate:
    lost_of_tensors_for_histo = []
    test()
    #print(sum(lost_of_tensors_for_histo)/90)

else:
    temp = args.F.split(',')
    args.F = [float(f) for f in temp]
    lost_of_tensors_for_histo = []
    lost_of_expectations     = []
    lost_of_tensors_for_histo_correct = []
    last_max_softmax_avg = 0
    top1_train = collections.deque([1,2,3,4,5,6,7,8,9,10], 10)
    top1_valid = collections.deque([1,2,3,4,5,6,7,8,9,10], 10)
    val_der = 10
    for epoch in range(args.start_epoch, args.epochs):
        gt_avg_correct_pred = 0
        gt_avg_wrong_pred = 0
        
        train_acc,train_loss = train(epoch,last_max_softmax_avg)
        scheduler.step()
        val_acc,val_loss,best_acc = validate(epoch)

        tb.add_scalar('Train_Top1',train_acc,epoch)     
        tb.add_scalar('Train_loss',train_loss,epoch)  

        tb.add_scalar('Val_loss',val_loss,epoch)
        tb.add_scalar('Val_Top1',val_acc,epoch)
        tb.add_scalar('Ground truth avarage-Correct predictions',gt_avg_correct_pred,epoch)
        tb.add_scalar('Ground truth avarage-Wrong predictions',gt_avg_wrong_pred,epoch)
        last_max_softmax_avg = gt_avg_correct_pred.item()
        time_elapsed = datetime.now() - start_time
            
    print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
    torch.save(torch.stack(lost_of_tensors_for_histo),os.path.join(args.save_path,"wrong_avarage_predictions.pth"))
   
    np.savetxt(os.path.join(args.save_path,"hist_val.txt"),torch.stack(lost_of_tensors_for_histo).detach().cpu().numpy(),fmt='%.d')
    np.savetxt(os.path.join(args.save_path,"hist_val_correct.txt"),torch.stack(lost_of_tensors_for_histo_correct).detach().cpu().numpy(),fmt='%.d')

    plot(lost_of_tensors_for_histo)
   
    plot(lost_of_tensors_for_histo_correct,wrong=False)
    # Run final test on never before seen data
    #test()

