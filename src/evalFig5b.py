import importlib
import gfa
importlib.reload(gfa)
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
import visualize
import pickle
import sys 
import pylab


data1 = np.load('Fig5b-numdatasets20-datasetindex1-modelranks2-16-2.npy')
data2 = np.load('Fig5b-numdatasets20-datasetindex2-modelranks2-16-2.npy')

average1 = np.average(data1, axis = 1)
average2 = np.average(data2, axis = 1)

modelrank = np.array([2, 4, 6, 8, 10, 12, 14, 16])

correct_rank_6 = np.argwhere(modelrank == 6)[0]
correct_rank_10 = np.argwhere(modelrank == 10)[0]

plt.figure(1)

plt1, = plt.plot(modelrank, average1,'-o',color='red')
plt2, = plt.plot(modelrank, average2,'-o',color='green')

plt.scatter(6, average1[correct_rank_6], s=1, c='b')
plt.scatter(6, average1[correct_rank_6], s=500, c='w')
plt.scatter(10, average2[correct_rank_10], s=1, c='b')
plt.scatter(10, average2[correct_rank_10], s=500, c='w')

pylab.xlim([0,18])

plt.xlabel('Model rank', fontsize='20')
plt.ylabel('Lower bound', fontsize='20')

plt.legend((plt1, plt2), ('Data rank 6', 'Data rank 10'),loc = 'lower right', fontsize='20')

plt.show()
