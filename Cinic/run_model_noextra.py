import sys
import subprocess 
from itertools import combinations
import itertools
import os
from tqdm import tqdm
import pandas as pd
if __name__ == "__main__":  
    F = "0,0.5"       
    F1 = "0,0.1,0.2,0.3,0.4,0.5"       
    arch = sys.argv[1]
    data = sys.argv[2]
    seeds = [60,253,1234]
    batch_size = 512
    if arch == "shufflenet" or arch == "resnet20" or arch == "resnet44" or arch == "res_net18" or arch == "resnet32" or arch == "vgg16":
       batch_size = 1024
       print("Batch Size:",batch_size)
    for seed in seeds:
       path = "experiments_noextra/"+str(arch)+"/"+str(arch)+"_seed" +str(seed)+"/baseline"
       acc_path = "experiments_noextra/"+str(arch)+"/"+str(arch)+"_seed" +str(seed)+"/accloss_det"

       if not os.path.exists(path) and not  os.path.exists(acc_path):
          baseline          = subprocess.check_output("python3 main_noextra.py --save_path  experiments_noextra/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/baseline  --arch "+str(arch)+" --epochs 60 --b " + str(batch_size) + " --seed "+str(seed) + " --data " + data, shell=True)
          accloss_det       = subprocess.check_output("python3 main_noextra.py --save_path experiments_noextra/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det  --arch "+str(arch)+" --accloss True --epochs 60 --b " + str(batch_size) + " --seed "+str(seed)+ " --data " + data, shell=True)
       accloss_det05     = subprocess.check_output("python3 main_noextra.py --save_path experiments_noextra/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det_F0_0.5_hf1.9 --arch "+str(arch)+" --accloss True --F "+ F+ " --hf 1.9 --epochs 60 --b " + str(batch_size) + " --seed "+str(seed)+ " --data " + data, shell=True)
       accloss_det0till5 = subprocess.check_output( "python3 main_noextra.py --save_path experiments_noextra/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det_F0till0.5_hf1  --arch "+str(arch)+" --accloss True  --F "+ F1+ " --hf 1 --epochs 60 --b " + str(batch_size) + " --seed "+str(seed)+ " --data " + data, shell=True)
