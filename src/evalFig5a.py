# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

rstart = int(sys.argv[1])
rend = int(sys.argv[2])
rstep = int(sys.argv[3])
num_datasets = int(sys.argv[4])


RMSE_6 = np.load('Fig5a-numdatasets{}-datasetindex1-modelranks{}-{}-{}-yaxis-NS.npy'.format(
    num_datasets, rstart, rend, rstep))
RMSE_10 = np.load('Fig5a-numdatasets{}-datasetindex2-modelranks{}-{}-{}-yaxis-NS.npy'.format(
    num_datasets, rstart, rend, rstep))
R_array_6 = np.load('Fig5a-numdatasets{}-datasetindex1-modelranks{}-{}-{}-xaxis.npy'.format(
    num_datasets, rstart, rend, rstep))
R_array_10 = np.load('Fig5a-numdatasets{}-datasetindex2-modelranks{}-{}-{}-xaxis.npy'.format(
    num_datasets, rstart, rend, rstep))

correct_rank_6 = np.argwhere(R_array_6 == 6)[0]
correct_rank_10 = np.argwhere(R_array_10 == 10)[0]

plt.figure(1)
plt1, = plt.plot(R_array_6, RMSE_6, '-o', color='r')
plt2, = plt.plot(R_array_10, RMSE_10, '-o', color='g')
plt.scatter(R_array_6[correct_rank_6], RMSE_6[correct_rank_6], s=1, c='b')
plt.scatter(R_array_6[correct_rank_6], RMSE_6[correct_rank_6], s=500, c='w')
plt.scatter(R_array_10[correct_rank_10], RMSE_10[correct_rank_10], s=1, c='b')
plt.scatter(R_array_10[correct_rank_10], RMSE_10[correct_rank_10], s=500, c='w')
plt.legend((plt1, plt2), ('Data rank 6', 'Data rank 10'),loc = 'upper right', fontsize='20')
plt.ylabel('Prediction RMSE', fontsize='20')
plt.xlabel('Model rank', fontsize='20')
plt.show()
