# usage: python gen.r > <out_name>.r
# then do source("<out_name>.r") in R environment to get the data

import numpy as np
from generate_data import *

# Create Corentin's artifical data set corresponding to Fig. 3 (original paper)
R = 3 #rank
K = 7 #factors
D = np.array([10,10,10]) #groups     np.array([2,4,4])
N = 100 #samples
X, W, Z = generation(N, K, D, R)

# centering of data must be done beforehand
X_mean = X.mean(axis=1, keepdims=True)
X = X - X_mean

# transpose to author's format and split into 3 groups
Y = np.hsplit(X.T, 3)

# generate R code to stdout
for i in range(len(Y)):
    print("y{} <- matrix(c({}), nrow={}, ncol={},byrow=TRUE)".format(
        i+1, ",".join(map(str, Y[i].flatten().tolist())), N, D[i]))

print("Y <- list(y1, y2, y3)")
print("K <- {}".format(K))
print("R <- {}".format(R))

# calling in R:
# source("CCAGFA.R")
# opts <- getDefaultOpts()
# opts$R <- R
# opts$dropK <- FALSE

# dropK FALSE means the code won't drop small factors,
# hence keep going even if it doesn't find correlations
