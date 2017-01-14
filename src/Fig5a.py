# -*- coding: utf-8 -*-
import numpy as np
import pickle
import importlib
import gfa
importlib.reload(gfa)
import matplotlib.pyplot as plt
import sys


datasetindex = int(sys.argv[1])
rstart = int(sys.argv[2])
rend = int(sys.argv[3])
rstep = int(sys.argv[4])
num_datasets = int(sys.argv[5])
 

R_array = range(rstart,rend+1,rstep) #model rank
RMSE = np.zeros([len(R_array),1])  # sequence of RMSE
NRMSE = np.zeros([len(R_array),1])  # sequence of RMSE (normalized)
NSRMSE = np.zeros([len(R_array),1])  # sequence of RMSE (normalized with squared denominator)


with open('data-fig5.pkl', 'rb') as f:
    data = pickle.load(f)
rank2_datasets = data['rank-2']
rank6_datasets = data['rank-6']
rank10_datasets = data['rank-10']

data = [rank2_datasets, rank6_datasets, rank10_datasets]
datasets = data[datasetindex]

K = 30       # factors
numM = 50    # number of groups
Dm = 10      # dimension of each group
D = Dm * np.ones(numM, dtype=int) #groups
scalerD=Dm * numM
N_test = 10
N_train = 40
N_total = N_train + N_test


for r_index in range(len(R_array)):
    for i in range(num_datasets):
        print("Running inference for dataset {}/{} and model rank {}/{}".format(
			i+1, num_datasets, r_index, len(R_array)))
        # GFA     
        X = datasets[i] 
        X_train = X[:,0:N_train]
        X_test = X[:,N_train:N_total]          
        
        g = gfa.GFA_rep(X_train, D, n=5, debug_iter=False, rank=R_array[r_index], factors=K, optimize_method="l-bfgs-b", debug=True, max_iter=10000)
        
        leave = 0
                    
        Yminus=np.zeros([N_test,0]) # K x 0 (matrix will be expanded afterward)
        
        T=np.zeros([scalerD - Dm,scalerD - Dm]) # D-Dm x D-Dm
      
        Wminus=np.zeros([K,0]) # K x 0 (matrix will be expanded afterward)
        Sigma=np.identity(K) # K x K
        Wm=g.E_W(leave) # K x Dm               
                    
        X_unseen = X_test[leave * Dm:(leave+1) * Dm, :]
        
        count=0
        for i in range(0,numM,1):
                if i != leave:
                    count=count+1
                    Yminus=np.c_[Yminus, X_test.T[:,i*Dm:(i+1)*Dm]]  # N x D-Dm
                    T[(count-1)*Dm:count*Dm,(count-1)*Dm:count*Dm]=g.E_tau(i)*np.identity(Dm) # diag({<tau_j>I_Dj}_j\neqm)                        
                    Sigma=Sigma+g.E_tau(i)*g.E_WW(i) # I_k+\sum_{j\neqm}<tau_j><W^(j){W^(j)}^{T}>
                    Wminus=np.c_[Wminus,g.E_W(i)] # Finally K x D-Dm 
                    
        Xmpre = Yminus @ T @ Wminus.T @ np.linalg.pinv(Sigma) @ Wm  # (N x D-Dm)x(D-Dm x D-Dm)x(D-Dm x K)x(K x K)x(K x Dm)=(N x Dm)
        
        maxi = X_unseen.max()
        mini = X_unseen.min()
        preerr = np.sqrt(np.sum((X_unseen - Xmpre)**2))
        preerr_part_scaled = np.sqrt(np.sum((X_unseen - Xmpre)**2) / (maxi - mini))
        preerr_scaled = np.sqrt(np.sum( ((X_unseen - Xmpre) / (maxi - mini))**2))
        RMSE[r_index] += preerr
        NRMSE[r_index] += preerr_part_scaled        
        NSRMSE[r_index] += preerr_scaled
    RMSE[r_index] /= num_datasets
    NRMSE[r_index] /= num_datasets
    NSRMSE[r_index] /= num_datasets
        

np.save('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-yaxis'.format(
    num_datasets, datasetindex, rstart, rend, rstep), RMSE)
np.save('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-yaxis-N'.format(
    num_datasets, datasetindex, rstart, rend, rstep), NRMSE)
np.save('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-yaxis-NS'.format(
    num_datasets, datasetindex, rstart, rend, rstep), NSRMSE)
np.save('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-xaxis'.format(
    num_datasets, datasetindex, rstart, rend, rstep), np.array(R_array))

