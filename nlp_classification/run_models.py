import sys
import os
import subprocess
import argparse
parser = argparse.ArgumentParser(description='Training')
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

if __name__ == "__main__":
    args = parser.parse_args()

    F = "0,0.5"
    F1 = "0,0.1,0.2,0.3,0.4,0.5"
    seeds = [60,253,692,1234,2349,11,17777,99]
    #seeds = [2349,11,17777,99]
    path = "experiments/"+args.save_path

    if not os.path.exists(path):
       print(path)
       os.mkdir(path)
    for seed in seeds:
       for iter in range(1):
         save_path = path+"/baseline"
         if not os.path.exists(save_path):
           os.mkdir(save_path)
         save_path = save_path+"/seed_"+str(seed)
         best_acc = subprocess.check_output("python3 main.py  --batch_size 1024  --epochs "+ str(args.epochs)  +" --save_path "+ save_path +" --seed "+str(seed)+" --hidden_size "+str(args.hidden_size) +" --embedding_dim "+str(args.embedding_dim)+" --num_layers "+str(args.num_layers), shell=True)      
         best_acc  = best_acc.decode("utf-8")
         best_acc = round(float(best_acc)*100,4)
         best_acc = subprocess.check_output("mv "+save_path+ " "  +save_path+"_"+str(best_acc), shell=True)      
       for iter in range(1):
         save_path = path+"/accloss"
         if not os.path.exists(save_path):
           os.mkdir(save_path)
         save_path = save_path+"/seed_"+str(seed)
         best_acc = subprocess.check_output("python3 main.py  --batch_size 1024 --epochs "+ str(args.epochs)  +" --save_path "+ save_path +" --accloss --seed "+str(seed)+" --hidden_size "+str(args.hidden_size) +" --embedding_dim "+str(args.embedding_dim)+" --num_layers "+str(args.num_layers), shell=True)      
         best_acc  = best_acc.decode("utf-8")
         best_acc = round(float(best_acc)*100,4)
         best_acc = subprocess.check_output("mv "+save_path+ " "  +save_path+"_"+str(best_acc), shell=True)      
       for iter in range(1):
         save_path = path+"/accloss_F0_0.5_hf1.9"
         if not os.path.exists(save_path):
           os.mkdir(save_path)
         save_path = save_path+"/seed_"+str(seed)

         best_acc = subprocess.check_output("python3 main.py  --batch_size 1024 --epochs "+ str(args.epochs)  +"  --save_path "+ save_path +" --accloss --F "+F +" --hf 1.9 --seed "+str(seed)+" --hidden_size "+str(args.hidden_size) +" --embedding_dim "+str(args.embedding_dim)+" --num_layers "+str(args.num_layers), shell=True)      
         best_acc  = best_acc.decode("utf-8")
         best_acc = round(float(best_acc)*100,4)
         best_acc = subprocess.check_output("mv "+save_path+ " "  +save_path+"_"+str(best_acc), shell=True)    
       for iter in range(1):
         save_path = path+"/accloss_F0till0.5_hf1"
         if not os.path.exists(save_path):
           os.mkdir(save_path)
         save_path = save_path+"/seed_"+str(seed)
         best_acc = subprocess.check_output("python3 main.py  --batch_size 1024  --epochs "+ str(args.epochs)  +"  --save_path "+ save_path +" --accloss --F "+F1 +" --hf 1 --seed "+str(seed)+" --hidden_size "+str(args.hidden_size) +" --embedding_dim "+str(args.embedding_dim)+" --num_layers "+str(args.num_layers), shell=True)      
         best_acc  = best_acc.decode("utf-8")
         best_acc = round(float(best_acc)*100,4)
         best_acc = subprocess.check_output("mv "+save_path+ " "  +save_path+"_"+str(best_acc), shell=True)    
