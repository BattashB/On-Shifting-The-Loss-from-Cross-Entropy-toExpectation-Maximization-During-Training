import torch
import torch.nn as nn

from argparse import ArgumentParser
from UCI.UCIGeneral import UCIDatasetGeneral
from torch.nn import functional as F
import random
#from kornia.losses import focal_loss
import numpy as np

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

def acc_func(pred,target,n_classes):
    target = F.one_hot(target,n_classes)
    pred_soft   = F.gumbel_softmax(pred,hard=False)
    pred   = F.gumbel_softmax(pred,hard=True)
    acc = torch.mean(torch.sum(pred*target,dim=-1))
    one_minus_acc = 1 - acc
    return one_minus_acc

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
    parser.add_argument('--onlyacc',type=bool, default=False)

    parser.add_argument('--focal',type=bool, default=False)
    parser.add_argument('--l2',type=bool, default=False)
    parser.add_argument('--det',type=bool, default=False)

    parser.add_argument('--epochs',type=int, default=10)
    parser.add_argument('--lr',type=float, default=0.001)
    
    parser.add_argument('--add2input4hidden',type=int, default=20)
    parser.add_argument('--train_batch',type=int, default=32)
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
    model = Model(INPUT_DIM,NUM_CLASSES,hidden_dim)#.cuda()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),args.lr, momentum=0.9)
    best_test = 0
    best_val = 0
    for epoch in range(args.epochs):
        print("epoch:",epoch)
        model.train()
        sum_acc = 0
        for batch_idx, (data, target) in enumerate(trainloader):           
           outputs = model(data)
           if args.focal:
             loss = torch.mean(focal_loss(outputs, target,alpha=0.25))
           elif args.onlyacc:
              loss=acc_func_deter(outputs,target,n_classes = NUM_CLASSES)
           else:
              loss = ce_loss(outputs,target)

   

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           args_out = torch.argmax(outputs,dim=-1)
           results = torch.sum((args_out == target).int()).item()
           sum_acc = sum_acc + results/data.shape[0]#).item()
        sum_acc    = sum_acc/(batch_idx+1)
        #print("Train:  Loss:",round(sum_losses,2),"| Accuracy:",round(sum_acc*100,2),"%")

        model.eval()
        sum_acc_val = 0
        for batch_idx, (data, target) in enumerate(validationloader):
           with torch.no_grad():

              outputs = model(data)
              loss = ce_loss(outputs,target)
              optimizer.zero_grad()
              args_out = torch.argmax(outputs,dim=-1)
              results = torch.sum((args_out == target).int()).item()
              sum_acc_val = sum_acc_val + results/data.shape[0]

        sum_acc_val = sum_acc_val/(batch_idx+1)
        #print("Val- Accuracy:",round(sum_acc_val*100,2),"%")        

        ################################
        sum_acc = 0
        for batch_idx, (data, target) in enumerate(testloader):#validationloader):
           with torch.no_grad():
              #data, target = data.#(), target.cuda()

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