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
    gpu  = sys.argv[3]# gpu index

    s = [60,253,1234]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    batch_size = 256
    cuda = "CUDA_VISISBLE_DEVICES="+str(gpu)
    for seed in seeds:
      baseline          = subprocess.check_output(cuda+" python3 main.py --save_path  experiments_b256_90ep/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/baseline  --arch "+str(arch)+" --epochs 90 --b " + str(batch_size) + " --seed "+str(seed) + " --data " + data, shell=True)
      accloss_det       = subprocess.check_output(cuda+" python3 main.py --save_path experiments_b256_90ep/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det  --arch "+str(arch)+" --accloss True --epochs 90 --b " + str(batch_size) + " --seed "+str(seed)+ " --data " + data, shell=True)
      accloss_det05     = subprocess.check_output(cuda+" python3 main.py --save_path experiments_b256_90ep/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det_F0_0.5_hf1.9 --arch "+str(arch)+" --accloss True --F "+ F+ " --hf 1.9 --epochs 90 --b " + str(batch_size) + " --seed "+str(seed)+ " --data " + data, shell=True)
      accloss_det0till5 = subprocess.check_output(cuda+" python3 main.py --save_path experiments_b256_90ep/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det_F0till0.5_hf1  --arch "+str(arch)+" --accloss True  --F "+ F1+ " --hf 1 --epochs 90 --b " + str(batch_size) + " --seed "+str(seed)+ " --data " + data, shell=True)
