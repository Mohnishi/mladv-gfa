import importlib
import gfa
importlib.reload(gfa)
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
import visualize
import pickle
import sys 



data1 = np.load('Fig5b-numdatasets20-datasetindex1-modelranks2-16-2.npy')


data2 = np.load('Fig5b-numdatasets20-datasetindex2-modelranks2-16-2.npy')


help1 = np.average(data1, axis = 1)

help2 = np.average(data2, axis = 1)



indices = np.array([2, 4, 6, 8, 10, 12, 14, 16])


plt.figure(1)

plt1 = plt.scatter(indices, help1,color='red', marker='s')


plt2 = plt.scatter(indices, help2,color='green', marker='s')


plt.xlabel('Model rank')

plt.ylabel('Lower bound')

plt.legend((plt1, plt2), ('Data rank 6', 'Data rank 10'),loc = 'lower right')

plt.show()
