import sys
import torch
import subprocess 
from itertools import combinations
import itertools
import os
import statistics
from tqdm import tqdm
import pandas as pd

def return_mean_std(mylist):
    try:
       std  = statistics.stdev(mylist[2:])
       std  = round(std,2)
    except:
       std=0
    mean = statistics.mean(mylist[2:])
    mean = round(mean,2)
    string = str(mean)+"\u00B1"+str(std)
    return string    

if __name__ == "__main__":  
    path = sys.argv[1]
    arch="base "
    batch_size = 100
    all_results = []
    baselines   = [arch+' ce','-']
    acclossdets  = [arch+' ce+el','-']
    acclossdets05  = [arch+' ce+el','{0,0.5}']
    acclossdets0till5  = [arch+' ce+el','{0,0.1.0.2,0.3,0.4,0.5}']
    columns_list=['Model','F']#'Top-1']
    seeds = []
    for dir in os.listdir(path):    
       subpath= os.path.join(path,dir)
    
       for dir_subpath in os.listdir(subpath):
          #print("subpath:",dir_subpath)
          acc = round(float(dir_subpath.split('_')[-1]),2)
          seed = dir_subpath.split('_')[-2]
          if seed not in seeds:
             seeds.append(seed)

          #print("dir:",dir)
          if dir == "baseline":
               baselines.append(acc)
          elif  dir == "accloss":
               acclossdets.append(acc)
          elif  dir == "accloss_F0_0.5_hf1.9":
               acclossdets05.append(acc)  
          elif  dir	 == "accloss_F0till0.5_hf1":
               acclossdets0till5.append(acc)

    baselines.append(return_mean_std(baselines))
    acclossdets.append(return_mean_std(acclossdets))
    acclossdets05.append(return_mean_std(acclossdets05))
    acclossdets0till5.append(return_mean_std(acclossdets0till5))
       
    all_results.append(baselines)
    all_results.append(acclossdets)
    all_results.append(acclossdets05)
    all_results.append(acclossdets0till5)
    #print(baselines)
    for seed in seeds:
       columns_list.append("seed="+str(seed))
    columns_list.append("Top1")
    print(columns_list)
    df = pd.DataFrame(all_results, columns = columns_list) 

    str = os.path.join(path,"output.tex")
    with open(str, 'w') as f:
      f.write(df.to_latex(index=False))
    
