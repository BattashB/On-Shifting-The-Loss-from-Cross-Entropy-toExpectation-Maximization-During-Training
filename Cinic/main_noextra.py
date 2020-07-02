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
parser.add_argument('--hf', type=float,default=1)
parser.add_argument('--extra_transform', type=bool, default=False)
parser.add_argument('--accloss', type=bool, default=False)
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



def getfactors(epoch):
    hf = args.hf
    N = args.epochs
    K = len(args.F)
    fpart = int(N*hf/K)
    if epoch < fpart:
        k = int(epoch*K/(N*hf))
    else:
        k = 1+int((epoch-fpart)*(K-1)/(N-fpart))
    alpha = 1-int(args.F[k]/0.5)    
    beta  = 1/(1-2*args.F[k]*alpha)
    return alpha,beta




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def acc_func_deter(pred,target):
    dim=-1
    target = F.one_hot(target,10)
    y_soft = F.softmax(pred,dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(pred, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft    
    acc = torch.mean(torch.sum(ret*target,dim=-1))    
    one_minus_acc = 1 - acc
    return one_minus_acc
    
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
        model = model.cuda()


print("Parameters:",count_parameters(model))
def train(epoch):
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
        if args.accloss:
            a = acc_func_deter(outputs, targets)
            alpha,beta =  getfactors(epoch)
            if batch_idx == 0: 
                print("##################################")
                print("Alpha:",alpha,"Beta:",beta)
                print("##################################")
            loss = alpha*loss + beta*a
        
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
    

    global best_acc
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

            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_loss = valid_loss/(batch_idx+1)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
    for epoch in range(args.start_epoch, args.epochs):
        gt_avg_correct_pred = 0
        gt_avg_wrong_pred = 0
        
        train_acc,train_loss = train(epoch)
        scheduler.step()
        val_acc,val_loss,best_acc = validate(epoch)

        tb.add_scalar('Train_Top1',train_acc,epoch)     
        tb.add_scalar('Train_loss',train_loss,epoch)  

        tb.add_scalar('Val_loss',val_loss,epoch)
        tb.add_scalar('Val_Top1',val_acc,epoch)
        time_elapsed = datetime.now() - start_time
            
    print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
   
