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
    print("###########################################")
    print("arch:",arch)
    print("###########################################")
    seeds = [1234,60,253,692]
    if "resnet" in arch:           
      for seed in seeds:
         #baseline      = subprocess.check_output("python3 trainer.py --save-dir experiments/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/baseline --model resnet --arch "+str(arch)+" --epochs 90  --seed "+str(seed), shell=True)
         accloss_det   = subprocess.check_output("python3 trainer.py --save-dir experiments/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det --model resnet --arch "+str(arch)+" --accloss True --epochs 90 --seed "+str(seed), shell=True)
         accloss_det05 = subprocess.check_output("python3 trainer.py --save-dir experiments/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det_F_0_0.5_hf1.9 --model resnet --arch "+str(arch)+" --accloss True  --F "+ F+ " --hf 1.9 --epochs 90  --seed "+str(seed), shell=True)
         accloss_det0till05 = subprocess.check_output("python3 trainer.py --save-dir experiments/"+str(arch)+"/"+str(arch)+"_seed"+str(seed)+"/accloss_det_F_0till0.5_hf1 --model resnet --arch "+str(arch)+" --accloss True  --F "+ F1+ " --hf 1 --epochs 90  --seed "+str(seed), shell=True)

