#from scipy.stats import wilcoxon, ranksums
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from argparse import ArgumentParser
from UCI.UCIGeneral import UCIDatasetGeneral
from torch.nn import functional as F
import random
import numpy as np
import time

def getfactors(epoch,args):
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

class Model(nn.Module):
    def __init__(self,input_dim,num_classes,hidden_dim):
      super(Model,self).__init__()
      #self.fc = nn.Linear(input_dim,num_classes)

      self.fc = nn.Linear(input_dim,hidden_dim)
      self.fc1 = nn.Linear(hidden_dim,num_classes)  
    def forward(self,x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc1(x)
        return x 


def acc_func_deter(pred,target,n_classes):
        dim=-1
        target = F.one_hot(target,n_classes)
        y_soft = F.softmax(pred,dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(pred, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft    
        acc = torch.mean(torch.sum(ret*target,dim=-1))    
        one_minus_acc = 1 - acc
        return one_minus_acc 

def main(): 

    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Adult')
    parser.add_argument('--accloss',type=bool, default=False)

    parser.add_argument('--det',type=bool, default=False)

    parser.add_argument('--epochs',type=int, default=200)
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--F', type=str, default="0")
    parser.add_argument('--hf', type=float,default=1)

    parser.add_argument('--add2input4hidden',type=int, default=1)
    parser.add_argument('--train_batch',type=int, default=8)
    parser.add_argument('--val_batch',type=int, default=128)
    parser.add_argument('--seed',type=int, default=60)
    args = parser.parse_args()

    #####
    #print("Determenistic mode")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    #####

    
    ######
    temp = args.F.split(',')
    args.F = [float(f) for f in temp]
    #####
    args.dataset = args.dataset.lower()
    
    trainset = UCIDatasetGeneral(dataset=args.dataset.lower(), root='data', train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch,
                                              shuffle=True, num_workers=2)
    # ratiovector = getclassratiovector(trainset)

    validationset = UCIDatasetGeneral(dataset=args.dataset.lower(), root='data', train=False,
                                      validation=True)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.val_batch,
                                                   shuffle=False, num_workers=2)

    testset = UCIDatasetGeneral(dataset=args.dataset.lower(), root='data', train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.val_batch,shuffle=False, num_workers=2)
    classes = range(trainset.num_classes)

    NUM_CLASSES = int(trainset.num_classes)
    INPUT_DIM = trainset.input_dim()
    hidden_dim = INPUT_DIM#2*NUM_CLASSES#INPUT_DIM+args.add2input4hidden#2*NUM_CLASSES#
    model = Model(INPUT_DIM,NUM_CLASSES,hidden_dim)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),args.lr, momentum=0.9)
    best_epoch = 0
    best_epoch_val = 0
    best_test = 0
    best_val = 0
    for epoch in range(args.epochs):
        model.train()
        sum_acc = 0
        sum_acc_val = 0
        sum_losses = 0
        #print("=================")
        #print("Epoch:",epoch)
        #print("=================")
        for batch_idx, (data, target) in enumerate(trainloader):           
           outputs = model(data)
           ce_out = ce_loss(outputs,target)
           #begin = time.time()
           if args.accloss:
              acc_loss = acc_func_deter(outputs,target,n_classes = NUM_CLASSES)
              main_factor,factor =  getfactors(epoch,args)
              loss = main_factor*ce_out + factor*acc_loss
           else:
               loss = ce_out 
           #end = time.time()
           #print("Time:",end-begin)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           args_out = torch.argmax(outputs,dim=-1)
           results = torch.sum((args_out == target).int()).item()
           sum_acc = sum_acc + results/data.shape[0]
           sum_losses = sum_losses + (loss/data.shape[0]).item()

        sum_losses = sum_losses/(batch_idx+1)
        sum_acc    = sum_acc/(batch_idx+1)
        #print("Train- Accuracy:",round(sum_acc*100,2),"%")
        model.eval()
        sum_acc_val = 0

        for batch_idx, (data, target) in enumerate(validationloader):
           with torch.no_grad():
              outputs = model(data)
              loss = ce_loss(outputs,target)
              optimizer.zero_grad()
              args_out = torch.argmax(outputs,dim=-1)
              results = torch.sum((args_out == target).int()).item()
              sum_acc_val = sum_acc_val + results/data.shape[0]#).item()
        sum_losses = sum_losses/(batch_idx+1)
        sum_acc_val = sum_acc_val/(batch_idx+1)
        #print("Val- Accuracy:",round(sum_acc_val*100,2),"%")

           
        #############################################
        sum_acc = 0

        for batch_idx, (data, target) in enumerate(testloader):#validationloader):
           with torch.no_grad():
              outputs = model(data)
              loss = ce_loss(outputs,target)
              optimizer.zero_grad()
              args_out = torch.argmax(outputs,dim=-1)
              results = torch.sum((args_out == target).int()).item()
              sum_acc = sum_acc + results/data.shape[0]#).item()
        sum_acc = sum_acc/(batch_idx+1)
        #print("Test- Accuracy:",round(sum_acc*100,2),"%")        
        if sum_acc_val >= best_val:
           best_val = sum_acc_val
           best_test = sum_acc
    return round(best_test,4),round(best_val,4)  


if __name__ == "__main__":
   acc = main()
   print(acc)