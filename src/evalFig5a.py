# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

datasetindex = int(sys.argv[1])
rstart = int(sys.argv[2])
rend = int(sys.argv[3])
rstep = int(sys.argv[4])
num_datasets = int(sys.argv[5])


RMSE = np.load('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-yaxis.npy'.format(
    num_datasets, datasetindex, rstart, rend, rstep))
NRMSE = np.load('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-yaxis-N.npy'.format(
    num_datasets, datasetindex, rstart, rend, rstep))
NSRMSE = np.load('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-yaxis-NS.npy'.format(
    num_datasets, datasetindex, rstart, rend, rstep))
R_array = np.load('Fig5a-numdatasets{}-datasetindex{}-modelranks{}-{}-{}-xaxis.npy'.format(
    num_datasets, datasetindex, rstart, rend, rstep))


plt.figure(1)
plt.plot(R_array, RMSE)
plt.show()

plt.figure(2)
plt.plot(R_array, NRMSE)
plt.show()

plt.figure(2)
plt.plot(R_array, NSRMSE)
plt.show()
