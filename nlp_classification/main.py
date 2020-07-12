import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
parser.add_argument('--accloss', default=False, action='store_true')
parser.add_argument('--seed',type=int,  default=60)
parser.add_argument('--epochs',type=int,  default=90)

parser.add_argument('--save_path',type=str,  default="save_path/")
parser.add_argument('--F',type=str,  default="0")
parser.add_argument('--hf',type=float,  default=1)
parser.add_argument('--batch_size',type=int,  default=1024)
parser.add_argument('--hidden_size',type=int,  default=50)
parser.add_argument('--embedding_dim',type=int,  default=50)
parser.add_argument('--num_layers',type=int,  default=1)


best_acc=0
best_epoch=0
best_histo = [0,0,0,0,0,0,0,0,0,0]

def max_softmax_pred(outputs,targets,wrong=True):
    y_soft = F.softmax(outputs,dim=-1)
    target_onehot = F.one_hot(targets,5)
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
    gamma = 1
    return alpha,beta,gamma


def acc_func_deter(pred,target):
    dim=-1
    target = F.one_hot(target,5)
    y_soft = F.softmax(pred,dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(pred, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft    
    acc = torch.mean(torch.sum(ret*target,dim=-1))    
    one_minus_acc = 1 - acc
    return one_minus_acc,pred 

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]
def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

def train_model(model, epochs, lr=0.001):
    global best_acc, best_histo, best_epoch
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    histolist=[]
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for batch_idx,(x, y, l) in enumerate(train_dl):
            x = x.long().cuda()
            y = y.long().cuda()
            l = l.cuda()
            y_pred = model(x,l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            if args.accloss:
               a,gs_pred = acc_func_deter(y_pred, y)        
               main_factor,factor,normalize =  getfactors(i)
               #if i % 45 == 0:
                   #print("Pay attention, we are at epoch:",i,",the factors are now:",main_factor,round(factor,4), "Normalization factor is:",normalize)
               loss = normalize*(main_factor*loss + factor*a)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, histogram  = validation_metrics(model, val_dl)
        histolist.append(histogram)
        if val_acc > best_acc:
           best_acc = val_acc
           best_histo = histogram 
           best_epoch = i
        #if i % 5 == 1:
        #    print("train loss %.3f, val loss %.3f, val accuracy %.3f" % (
        #    sum_loss / total, val_loss, val_acc))
    return histolist

def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long().cuda()
        y = y.long().cuda()
        l = l.cuda()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        histogram = hist_of_wrong(y_hat,y)
        

    return sum_loss / total, correct / total,histogram 


class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_layers=1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers= num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
    def forward(self, x, l):
        x = self.embeddings(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

if __name__ == "__main__":
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
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
    temp = args.F.split(',')
    args.F = [float(f) for f in temp]

    reviews = pd.read_csv("Reviews.csv")
    #print(reviews.shape)
    #Replacing Nan values
    reviews['Title'] = reviews['Title'].fillna('')
    reviews['Review Text'] = reviews['Review Text'].fillna('')

    reviews['review'] = reviews['Title'] + ' ' + reviews['Review Text']
    reviews = reviews[['review', 'Rating']]
    reviews.columns = ['review', 'rating']
    reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
    # changing ratings to 0-numbering
    zero_numbering = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])
    np.mean(reviews['review_length'])
    tok = spacy.load('en_core_web_sm')



    counts = Counter()
    for index, row in reviews.iterrows():
        counts.update(tokenize(row['review']))
    #print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    #print("num_words after:",len(counts.keys()))
    #creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
    reviews.head()
    Counter(reviews['rating'])
    X = list(reviews['encoded'])
    y = list(reviews['rating'])
    #haluka = int(len(X)*0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=args.seed)#X[:haluka],X[haluka:],y[:haluka],y[haluka:]#
    train_ds = ReviewsDataset(X_train, y_train)
    valid_ds = ReviewsDataset(X_valid, y_valid)


     
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=args.batch_size)
    model_fixed = LSTM_fixed_len(vocab_size, args.embedding_dim, args.hidden_size,args.num_layers)

    histolist = train_model(model_fixed.cuda(), epochs=args.epochs, lr=0.01)
    #print("Best accuracy:",best_acc,"Histogram:",best_histo," at epoch:", best_epoch)
    print(best_acc.item())
    with open(os.path.join(args.save_path,"log.log"),'w+') as f:
        f.write("Best Accuracy:")
        f.write(str(best_acc))
        f.write(", At epoch: ")
        f.write(str(best_epoch))
        f.write("Best Epoch histogram: ")
        best_histo = best_histo.tolist()
        for ele in best_histo:
           f.write(str(int(ele))+',')
        f.write('\n')

    with open(os.path.join(args.save_path,"histo_val_along_epochs.log"),'w+') as f:
       for histo in histolist:
           f.write('[')
           for ele in histo:
              f.write(str(int(ele))+',')
           f.write(']')
           f.write('\n')
              
