# usage: python gen.r > <out_name>.r
# then do source("<out_name>.r") in R environment to get the data

import numpy as np
from generate_data import *

def print_to_R(X, D, R, K):
    # translate to zero mean
    X_mean = X.mean(axis=1, keepdims=True)
    X = X - X_mean

    # groups
    M = len(D)
    # samples
    N = X.shape[1]
    # transpose to author's format and split into 3 groups
    Y = np.hsplit(X.T, M)

    # generate R code to stdout
    ys = []
    for i in range(M):
        print("y{} <- matrix(c({}), nrow={}, ncol={},byrow=TRUE)".format(
            i+1, ",".join(map(str, Y[i].flatten().tolist())), N, D[i]))
        ys.append("y"+str(i+1))

    print("Y <- list({})".format(",".join(ys)))
    print("K <- {}".format(K))
    print("R <- {}".format(R))

    # calling in R:
    # source("CCAGFA.R")
    # opts <- getDefaultOpts()
    # opts$R <- R
    # opts$dropK <- FALSE

    # dropK FALSE means the code won't drop small factors,
    # hence keep going even if it doesn't find correlations
