import importlib
import gfa
importlib.reload(gfa)
import matplotlib.pyplot as plt
import numpy as np
from generate_data import *

groupmax=40 # maximum group number
groupmin=4  # minimum group number
groupcounter=2
RMSE = np.zeros([(groupmax-groupmin)//groupcounter+1,1])  # sequence of RMSE from M=4 to M=100
RMSE2 = RMSE
trialmax=50 # trial number for average



for changegroup in range(0,(groupmax-groupmin)//groupcounter+1,1):
        # Create Corentin's artifical data set corresponding to Experiment C (original paper)
        Leave = 0 # group which we will leave for prediction
        
        K = 18 #factors
        numM = changegroup*groupcounter+groupmin # number of groups
        Dm = 7 # dimension of each group
        D = Dm*np.ones(numM,dtype=int) #groups
        scalerD=Dm*numM
        N = 30 #samples
        R = 4#min(K,numM)#4 #rank


        for trial in range(0,trialmax,1):
                X, W, Z, alpha, Tau = generation(N, K, D, R,constrain_W=10) # independent trial = data is generated at each trial
                
                # Run GFA
                for rankdif in range(0,2,1):
                        if rankdif==0:
                                R=4
                        else:
                                R=numM
                
                        g= gfa.GFA_rep(X,D, n=1, debug_iter=False, rank=R, factors=K,optimize_method="l-bfgs-b", debug=False, max_iter=1000)
               
               # while W.max()>10000:
                #        g = gfa.GFA_rep(X,D, n=1, debug_iter=False, rank=R, factors=K,optimize_method="l-bfgs-b", debug=False, max_iter=100)
               
                # Calculate leave-one-prediction
                        print(changegroup)
                        print(trial)
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

                        Xmpre=Yminus @ T @ Wminus.T @ np.linalg.pinv(Sigma) @ Wm  # (N x D-Dm)x(D-Dm x D-Dm)x(D-Dm x K)x(K x K)x(K x Dm)=(K x Dm)

                        preerr=np.sqrt(np.sum(((X.T[:,Leave*Dm:(Leave+1)*Dm]-Xmpre)/(np.max(X.T[:,Leave*Dm:(Leave+1)*Dm])-np.min(X.T[:,Leave*Dm:(Leave+1)*Dm])))**2))  # squared error of the prediction = scaler
                        print(preerr)
                        if rankdif==0:
                                RMSE[changegroup]=RMSE[changegroup]+preerr  # summing (devided by trialmax afterward)
                                nameoffile='saved_data_when_number_of_group_is'
                                numbergroup=str(numM)
                                np.save(nameoffile+numbergroup, RMSE)
                        else:
                                RMSE2[changegroup]=RMSE2[changegroup]+preerr  # summing (devided by trialmax afterward)
                                nameoffile='saved_data2_when_number_of_group_is'
                                numbergroup=str(numM)
                                np.save(nameoffile+numbergroup, RMSE2)
                        
                
RMSE=RMSE/float(trialmax)  # averaging
RMSE2=RMSE2/float(trialmax)  # averaging
# Visualize the result
plt.figure(1)
plt.plot(range(groupmin,groupmax+1,groupcounter),RMSE,color='r',linewidth=3)
plt.plot(range(groupmin,groupmax+1,groupcounter),RMSE2,color='b',linewidth=3)

plt.title("prediction RMSE against group number")
plt.ylabel("Prediction RMSE")
plt.xlabel("Groups")
plt.savefig('Figure4a.eps')
plt.show()
