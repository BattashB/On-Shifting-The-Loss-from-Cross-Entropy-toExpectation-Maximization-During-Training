import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
import resnet

from models import resnet as res_net
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pre_act_resnet
import resnext29 


arch_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


model_names=['resnet','shufflenet','mobilenet','lenet5'  ]
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch',  default='resnet32')
parser.add_argument('--model', default='resnet',
                    choices=model_names)                   
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--data_path',default="data", type=str)

parser.add_argument('--pretrained_path', type=str)
parser.add_argument('--tb_filename', type=str)

parser.add_argument('--accloss', type=bool, default=False)
parser.add_argument('--hf', type=float, default=1)
parser.add_argument('--extra_transform', type=bool, default=False)
parser.add_argument('--F', type=str, default="0")
parser.add_argument('--seed', type=int, default=60)


parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)
best_prec1 = 0
count_sto = 0
lost_of_tensors_for_histo = []

def max_softmax_pred(outputs,targets,wrong=True):
    y_soft = F.softmax(outputs,dim=-1)
    target_onehot = F.one_hot(targets,100)
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
    hist_normalize = hist/len_of_exampale
    return hist_normalize
def write_env(args):
    path = os.path.join(args.save_dir,'env.txt')
    os.popen('python3 collect_env.py >'+path)

    args_path = os.path.join(args.save_dir,'cmdline.txt')
    with open(args_path,'w+') as f:
        for k,v in args.__dict__.items():
           f.write(str(k))
           f.write(":")
           f.write(str(v))
           f.write('\n')


        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main():
    global args, best_prec1
    args = parser.parse_args()

    #####
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    #####

    #############

    temp = args.F.split(',')
    args.F = [float(f) for f in temp]
    best_prec1 = 0
    if "/" in args.save_dir:
        last = args.save_dir.split('/')[-1]
        args.tb_filename = os.path.join(args.save_dir,last+"_tb")
    else:
        args.tb_filename = os.path.join(args.save_dir,args.save_dir+"_tb")
    
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    ###extract a file that holds the env information###
    write_env(args)
    ###################################################
    if args.arch == "preactresnet":
       model = pre_act_resnet.PreActResNet18_CIFAR100()
    elif args.arch == "resnext29":  
       model = resnext29.resnext29_2_64()
    elif args.arch == "res_net18":
       model = res_net.res_net18() 
    elif args.arch == "res_net34":
       model = res_net.res_net34()  
    elif args.arch == "shufflenetv2":
       model =   shufflenetv2.shuffle_netv2()#models.__dict__[args.arch]()


    else:
       model = resnet.__dict__[args.arch]()

    print("Number of parameters:",count_parameters(model))
    model.cuda()
    if args.pretrained:
        new_dict = {}
        dict = torch.load(args.pretrained_path)['state_dict']
        for k,v in dict.items():
            if "module" in k:
                k = k[7:]
            new_dict[k] = v
        model.load_state_dict(new_dict)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


          
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=args.data_path, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100(root=args.data_path, train=False, transform=transforms.Compose([
             transforms.ToTensor(),
             normalize,
          ])),
          batch_size=1000, shuffle=False,
          num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[120,160,200], last_epoch=args.start_epoch - 1)                                                           


    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    tb = SummaryWriter(args.tb_filename)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_top1,t_loss = train(train_loader, model, criterion, optimizer,epoch)
        lr_scheduler.step()
        tb.add_scalar('Train_Top1',train_top1,epoch)     
        tb.add_scalar('Train_loss',t_loss,epoch)  
        # evaluate on validation set
        prec1,v_loss = validate(val_loader, model, criterion)
        tb.add_scalar('Val_loss',v_loss,epoch)
        tb.add_scalar('Val_Top1',prec1,epoch)  
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec1 = round(best_prec1,2)
        print("best_prec1:",best_prec1)
        if is_best:
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, best_prec1,epoch)
    plot(lost_of_tensors_for_histo)
    torch.save(torch.stack(lost_of_tensors_for_histo),os.path.join(args.save_dir,"wrong_avarage_predictions.pth"))


def acc_func_deter(pred,target):
    dim=-1
    target = F.one_hot(target,100)
    y_soft = F.softmax(pred,dim)    
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(pred, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft        
    acc = torch.mean(torch.sum(ret*target,dim=-1))    
    one_minus_acc = 1 - acc
    return one_minus_acc


def getfactors(epoch):
    #hf = hf in paper or in other code parts
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

    
    
def train(train_loader, model, criterion, optimizer, epoch):
    global count_sto
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        
        if args.accloss:
            a = acc_func_deter(output, target_var)

            alpha,beta =  getfactors(epoch)
            if i == 0:
                print("##########################################")
                print(" alpha:",alpha,"beta:",beta)
                print("##########################################")
            loss = alpha*loss + beta*a

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    
    return top1.avg,losses.avg
    
def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    global lost_of_tensors_for_histo
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            histogram = hist_of_wrong(output,target_var)

            lost_of_tensors_for_histo.append(histogram)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg,losses.avg

def save_checkpoint(state, best_prec1,epoch):
    filename=os.path.join(args.save_dir, 'best_'+str(best_prec1)+'_'+str(epoch)+'.pth')
    print("Saving:",filename)
    torch.save(state, filename)
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
        fig.savefig(os.path.join(args.save_dir,"wrong_histo.jpg"))
    else:
        ax.plot_surface(epoch , likelihood,out_tensor, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
        ax.set_title('surface');
        fig.savefig(os.path.join(args.save_dir,"correct_histo.jpg"))
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
