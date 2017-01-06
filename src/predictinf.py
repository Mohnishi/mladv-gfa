import importlib
import gfa
importlib.reload(gfa)
import matplotlib.pyplot as plt
import numpy as np
from generate_data import *

RMSE = np.zeros([100-4+1,1])  # sequence of RMSE from M=4 to M=100
trialmax=50 # trial number for average
groupmax=100 # maximum group number
groupmin=4  # minimum group number

for changegroup in range(groupmin-4,groupmax-3,1):

        # Create Corentin's artifical data set corresponding to Experiment C (original paper)
        Leave = 1 # group which we will leave for prediction
        R = 4 #rank
        K = 18 #factors
        numM = changegroup+4 # number of groups
        Dm = 7 # dimension of each group
        D = Dm*np.identity(numM) #groups     
        N = 30 #samples


        for trial in range(0,trialmax,1):
                X, W, Z = generation(N, K, D, R) # independent trial = data is generated at each trial
                
                # Run GFA
                g = gfa.GFA(optimize_method="bfgs", debug=True)
                g.fit(X,D)

                # Get estimated W (West)
                West = g.get_W()

                # Calculate leave-one-prediction
                Yminus=X.T[:,Dm:]  # K x D-Dm
                T=np.zeros([D-Dm,D-Dm]) # D-Dm x D-Dm
                Wminus=np.zeros([K,0]) # K x 0 (matrix will be expanded afterward)
                Sigma=np.identity(K) # K x K
                Wm=g.E_W(1) # K x Dm

                for i in range(2,numM+1,1):
                        T[(i-2)*Dm:(i-1)*Dm-1,(i-2)*Dm:(i-1)*Dm-1]=g.E_tau(i)*np.identity(Dm) # diag({<tau_j>I_Dj}_j\neqm)
                        Sigma=Sigma+g.E_tau(i)*g.E_WW(i) # I_k+\sum_{j\neqm}<tau_j><W^(j){W^(j)}^{T}>
                        Wminus=np.c_[Wminus,g.E_W(i)] # Finally K x D-Dm 

                Xmpre=Yminus @ T @ Wminus.T @ Sigma @ g.E_W(1)  # (K x D-Dm)x(D-Dm x D-Dm)x(D-Dm x K)x(K x K)x(K x Dm)=(K x Dm)

                preerr=np.sum((X.T[:,0:Dm-1]-Xmpre)**2)  # squared error of the prediction = scaler
                
                RMSE[changegroup]=RMSE[changegroup]+preerr  # summing (devided by trialmax afterward)

RMSE=RMSE/trialmax  # averaging


# Visualize the result
plt.figure(1)
plot(range(4,101,1),RMSE,color='r',linewidth=3)

plt.title("prediction RMSE against group number")
plt.ylabel("Prediction RMSE")
plt.xlabel("Groups")

plt.show()
