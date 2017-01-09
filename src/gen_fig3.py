# generate figure 3 similar to one found in the paper
import numpy as np
import random
from genr import *
from genf import *

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
            W[k,base:base+D[m]] = np.random.normal(loc=0, scale=5, size=D[m])
        base += D[m]

# random Z
Z = np.random.normal(loc=0, scale=1, size=(K, N))

# X from W and Z
X = W.T @ Z + np.random.normal(loc=0, scale=np.sqrt(1/tau), size=(sum(D), N))

np.save("res/w_real.npy", W)
np.save("res/x.npy", X)
np.save("res/d.npy", D)

print_to_R(X, D, R, K)
