# this scripts runs the reference GFA algorithm
# and outputs the results in python code
# it assumes the CCAGFA.R implementation file and
# the data.r data file in the same directory

# usage: Rscript run_convert.r

# import GFA code and data Y, parameters from test data
source("CCAGFA.R")
source("data.r")
# data Y, factors K and rank R now available

# set parameters to GFA
opts <- getDefaultOpts()
opts$R <- R
# silent run and don't optimize away factors
opts$dropK <- FALSE
opts$verbose <- 0

# run GFA
res <- GFA(Y, K, opts)

# convert W back to python (row major flattening)
# concatenate W
W <- do.call(rbind, res$W)
# W is transposed compared to paper
cat("import numpy as np\n\n")
cat("W_flat = np.array([", paste(c(W), collapse=",", sep=""), "])\n", sep="")
cat("W = W_flat.reshape((", ncol(W), ",", nrow(W), "))\n", sep = "")
cat("np.save('res/w_ref.npy', W)\n")

cat("bounds = np.array([", paste(res$cost,  collapse=",", sep=""), "])\n", sep="")
cat("np.save('res/bounds_ref.npy', bounds)\n")

# overwrite?
source("data.r")

opts <- getDefaultOpts()
opts$R <- "full"
# silent run and don't optimize away factors
opts$dropK <- FALSE
opts$verbose <- 0

# run GFA
res_full <- GFA(Y, K, opts)
# W is transposed compared to paper
W_full <- do.call(rbind, res_full$W)
cat("W_flat_full = np.array([", paste(c(W_full), collapse=",", sep=""), "])\n", sep="")
cat("W_full = W_flat.reshape((", ncol(W_full), ",", nrow(W_full), "))\n", sep = "")
cat("np.save('res/w_ref_full.npy', W_full)\n")

cat("bounds_full = np.array([", paste(res_full$cost,  collapse=",", sep=""), "])\n", sep="")
cat("np.save('res/bounds_ref_full.npy', bounds_full)\n")
