# generate figure 3 similar to one found in the paper
import numpy as np
import random
from genr import *
from genf import *
import importlib
import gfa
importlib.reload(gfa)

import matplotlib.pyplot as plt

tau = 10
R = 3
K = 7
D = np.array([10,10,10]) #groups     np.array([2,4,4])
N = 100

# create filter
F = gen_filter(len(D))

# make some columns W completely zero
W = np.zeros((K, sum(D)))

for k in range(K):
    # keep some groups all zero according to the filter
    base = 0
    for m in range(len(D)):
        #if random.getrandbits(1):
        if F[m,k] == 1:
            W[k,base:base+D[m]] = np.random.normal(loc=0, scale=1, size=D[m])
        base += D[m]
		
		
# random Z
Z = np.random.normal(loc=0, scale=1, size=(K, N))

# X from W and Z
X = W.T @ Z + np.random.normal(loc=0, scale=np.sqrt(1/tau), size=(sum(D), N))



# Run GFA
g = gfa.GFA(optimize_method="bfgs", debug=True)
g.fit(X,D)

# Get estimated W (West)
West = g.get_W()

# Extract groups of estimated projection mappings
West1 = West[:,:10]
West2 = West[:,10:20]
West3 = West[:,:30]		
		
		
		
# # Extract groups of the true projection mappings
W1 = W[:,:10]

plt.figure(1)
for k in range(7):
	for d in range(10):
		plt.scatter(k+1,d+1,s=W1[k,d],color='black', marker = 's')

plt.title("True W - Group 1")
plt.ylabel("K axis")
plt.xlabel("Dm axis")


plt.figure(2)
for k in range(7):
	for d in range(10):
		plt.scatter(k+1,d+1,s=West1[k,d],color='black', marker = 's')

plt.title("Estimated W - Group 1")
plt.ylabel("K axis")
plt.xlabel("Dm axis")

plt.show()