import importlib
import gfa
importlib.reload(gfa)
import matplotlib.pyplot as plt
import numpy as np
from generate_data import *

groupmax=15 # maximum group number
groupmin=10  # minimum group number
groupcounter=1
RMSE = np.zeros([(groupmax-groupmin)//groupcounter+1,1])  # sequence of RMSE from M=4 to M=100
trialmax=10 # trial number for average



for changegroup in range(0,(groupmax-groupmin)//groupcounter+1,1):

        # Create Corentin's artifical data set corresponding to Experiment C (original paper)
        Leave = 0 # group which we will leave for prediction
        R = 4 #rank
        K = 8 #factors
        numM = changegroup*groupcounter+groupmin # number of groups
        Dm = 7 # dimension of each group
        D = Dm*np.ones(numM) #groups
        scalerD=Dm*numM
        N = 30 #samples


        for trial in range(0,trialmax,1):
                X, W, Z, alpha, Tau = generation(N, K, D, R) # independent trial = data is generated at each trial
                
                # Run GFA
                g = gfa.GFA(rank=R, factors=K,optimize_method="l-bfgs-b", debug=True, max_iter=5)
                g.fit(X,D)
               
                # Calculate leave-one-prediction
                Yminus=np.zeros([N,0]) # K x 0 (matrix will be expanded afterward)
                
                T=np.zeros([scalerD-Dm,scalerD-Dm]) # D-Dm x D-Dm
              
                Wminus=np.zeros([K,0]) # K x 0 (matrix will be expanded afterward)
                Sigma=np.identity(K) # K x K
                Wm=g.E_W(Leave) # K x Dm
               
                count=0
                for i in range(0,numM,1):
                        if i!=Leave:
                                count=count+1
                                Yminus=np.c_[Yminus,X.T[:,i*Dm:(i+1)*Dm]]  # N x D-Dm
                                T[(count-1)*Dm:count*Dm,(count-1)*Dm:count*Dm]=g.E_tau(i)*np.identity(Dm) # diag({<tau_j>I_Dj}_j\neqm)
                       
                                Sigma=Sigma+g.E_tau(i)*g.E_WW(i) # I_k+\sum_{j\neqm}<tau_j><W^(j){W^(j)}^{T}>
                                Wminus=np.c_[Wminus,g.E_W(i)] # Finally K x D-Dm 
        
                Xmpre=Yminus @ T @ Wminus.T @ np.linalg.inv(Sigma) @ Wm  # (N x D-Dm)x(D-Dm x D-Dm)x(D-Dm x K)x(K x K)x(K x Dm)=(K x Dm)
                print(Xmpre)

                preerr=np.sum((X.T[:,Leave*Dm:(Leave+1)*Dm]-Xmpre)**2)/N  # squared error of the prediction = scaler
                print(preerr)
                RMSE[changegroup]=RMSE[changegroup]+preerr  # summing (devided by trialmax afterward)

RMSE=np.sqrt(RMSE/float(trialmax))  # averaging

# Visualize the result
plt.figure(1)
plt.plot(range(groupmin,groupmax+1,groupcounter),RMSE,color='r',linewidth=3)

plt.title("prediction RMSE against group number")
plt.ylabel("Prediction RMSE")
plt.xlabel("Groups")

plt.show()
