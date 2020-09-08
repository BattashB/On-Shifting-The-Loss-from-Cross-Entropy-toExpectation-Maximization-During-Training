import sys
import subprocess 
from itertools import combinations
import itertools
import os
from tqdm import tqdm
import pandas as pd
if __name__ == "__main__":  
   datasets = os.listdir("data")
   epochs = [50]
   batch_sizes = [8]
   lrs = [0.01,0.005,0.001]
   add2input4hidden = [1]

   all_results = []
   baseline_results = []
   onlyacc_results = []
   focal_results = []
   basethenacc_results=[]
   accdet0_05_results= []
   accdet0till05_results= []
   accdet_results = []
   s=[batch_sizes, lrs , epochs,add2input4hidden ]
   all_perm = list(itertools.product(*s))
   list_of_bad_datasets = []
   how_much_better = 0
   how_much_eq = 0
   how_much_better_det = 0
   how_much_eq_det = 0
   #dict_results = {}
   for dataset in tqdm(datasets):
      best_base = 0
      best_accdet05 = 0
      best_accdetill05 = 0
      best_accdet  = 0
      best_onlyacc  = 0
      best_focal  = 0
      ############
      best_base_test = 0
      best_accdet05_test = 0
      best_accdetill05_test = 0
      best_accdet_test  = 0
      best_onlyacc_test  = 0
      best_focal_test  = 0

      for i,(batch_size,lr,epoch,add2input4hidden) in  enumerate(all_perm):    
         #dict_results[dataset]= {}
         arg_str = " --train_batch " + str(batch_size) + " --lr " +str(lr) + " --epochs " + str(epoch) + " --add2input4hidden " + str(add2input4hidden)
         baseline = subprocess.check_output("python3 main_best_focal_l2_acc_val.py  --seed 60  --dataset "+dataset+arg_str, shell=True)
         base2save_tuple  = baseline.decode("utf-8") 
         base_test= round(float(base2save_tuple.split(',')[0].split('(')[1])*100,4)
         base_val = round(float(base2save_tuple.split(',')[1].split(')')[0])*100,4)
         if base_val> best_base:
            best_base = base_val
            best_base_test = base_test            
         #print("base:",best_base,best_base_test)
         ########

         focal  = subprocess.check_output("python3 main_best_focal_l2_acc_val.py --seed 60 --focal True --dataset "+ dataset + arg_str, shell=True)
         focal2save_tuple  = focal.decode("utf-8") 
         focal_test= round(float(focal2save_tuple.split(',')[0].split('(')[1])*100,4)
         focal_val = round(float(focal2save_tuple.split(',')[1].split(')')[0])*100,4)
         if focal_val> best_focal:
            best_focal = focal_val
            best_focal_test = focal_test            
         #print("focal:",best_focal,best_focal_test)
         #########

         only_acc  = subprocess.check_output("python3 main_best_focal_l2_acc_val.py  --seed 60  --accloss True --det True --onlyacc True --dataset "+ dataset + arg_str, shell=True)
         onlyacc2save_tuple  = only_acc.decode("utf-8") 
         onlyacc_test= round(float(onlyacc2save_tuple.split(',')[0].split('(')[1])*100,4)
         onlyacc_val = round(float(onlyacc2save_tuple.split(',')[1].split(')')[0])*100,4)
         if onlyacc_val> best_onlyacc:
            best_onlyacc      = onlyacc_val
            best_onlyacc_test = onlyacc_test 
         #print("only acc:",best_onlyacc,best_onlyacc_test)
         ######### 
         
         accloss_det  = subprocess.check_output("python3 main_best_0.5_0_val.py  --seed 60  --accloss True --det True --dataset "+ dataset + arg_str, shell=True)
         acclossdet2save_tuple  = accloss_det.decode("utf-8")
         acclossdet_test   = round(float(acclossdet2save_tuple.split(',')[0].split('(')[1])*100,4)
         acclossdet_val    = round(float(acclossdet2save_tuple.split(',')[1].split(')')[0])*100,4)
         if acclossdet_val> best_accdet:
            best_accdet      = acclossdet_val
            best_accdet_test = acclossdet_test 
         #print(" acc:",best_accdet ,best_accdet_test)
         ######### 

         F_str = str("0,0.5")
         hf = 1.9         
         accloss_det0_05  = subprocess.check_output("python3 main_best_0.5_0_val.py --seed 60  --accloss True --det True --F "+ F_str +" --hf "+ str(hf) +" --dataset "+ dataset + arg_str, shell=True)
         accdet2save05_tuple  = accloss_det0_05.decode("utf-8")
         accdet05_test   = round(float(accdet2save05_tuple.split(',')[0].split('(')[1])*100,4)
         accdet05_val    = round(float(accdet2save05_tuple.split(',')[1].split(')')[0])*100,4)
         if accdet05_val> best_accdet05:
            best_accdet05 = accdet05_val
            best_accdet05_test = accdet05_test 
         #print(" acc 05:",best_accdet05 ,best_accdet05_test)
         ######### 
         
         F_str = str("0,0.1,0.2,0.3,0.4,0.5")
         hf = 1
         accloss_det0till05  = subprocess.check_output("python3 main_best_0.5_0_val.py --seed 60  --accloss True --det True --F "+ F_str +" --hf "+ str(hf) +" --dataset "+ dataset + arg_str, shell=True)
         acclossdet2save0till5_tuple  = accloss_det0till05.decode("utf-8")
         accdetill05_test   = round(float(acclossdet2save0till5_tuple.split(',')[0].split('(')[1])*100,4)
         accdetill05_val    = round(float(acclossdet2save0till5_tuple.split(',')[1].split(')')[0])*100,4)
         if accdetill05_val> best_accdetill05:
            best_accdetill05= accdetill05_val
            best_accdetill05_test = accdetill05_test 
         #print(" acc till 05:",best_accdetill05,best_accdetill05_test)
         ######### 

 
      dataset_results = [dataset,best_base_test,best_accdet_test,best_accdet05_test,best_accdetill05_test,best_onlyacc_test,best_focal_test]


      baseline_results.append(best_base_test)  
      accdet0_05_results.append(best_accdet05_test)
      accdet0till05_results.append(best_accdetill05_test)
      accdet_results.append(best_accdet_test)
      onlyacc_results.append(best_onlyacc_test)
      focal_results.append(best_focal_test)
      print(dataset ,":","Dataset results:",dataset_results)
      all_results.append(dataset_results)
   #print("list of bads:",list_of_bad_datasets)
   df = pd.DataFrame(all_results, columns = ['Dataset', 'CE','CE&ACC','CE&ACC_0_0.5','CE&ACC_0till0.5','ACC','FOCAL']) 

   #path = "only_det_batch_"+str(batch_size)+"_lr_"+ str(lr) +"_epoch_"+ str(epoch)+"_add2input4hidden_"+str(add2input4hidden)
   #path = "50_8_0.01_0.005_0.001_hiddenisinput_all7_check0.5_0"
   path = "50_8_0.01_0.005_0.001_2fc_seed60_val"

   if not os.path.exists(path):
      os.mkdir(path)
   df.to_csv(r'table.csv', index = False)
   with open(os.path.join(path,'mytable.tex'), 'w') as f:
      f.write(df.to_latex())


