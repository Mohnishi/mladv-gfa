import importlib
import gfa
importlib.reload(gfa)
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
import visualize
import pickle
import sys 

datasetindex = int(sys.argv[1])
rstart = int(sys.argv[2])
rend = int(sys.argv[3])


M = 50
Dm = 10 
D = Dm*np.ones(M,dtype = int)


with open('data-fig5.pkl', 'rb') as f:
    data = pickle.load(f)
rank2_datasets = data['rank-2']
rank6_datasets = data['rank-6']
rank10_datasets = data['rank-10']


R = range(rstart,rend+1) #model rank

data = [rank2_datasets, rank6_datasets, rank10_datasets]

bounds = []







dataset = data[datasetindex]

for r in R: # for all model ranks
	bound = 0
	for i in range(50): #always 50 artificial data sets
		g = gfa.GFA(debug=True, max_iter=10000, factors = 30, rank = r)
		g.fit(dataset[i],D)
		bounds.append(g.bound()) 
		
res = np.array(bounds).reshape(-1,50)

np.save('Fig5b-dataset{}-rank{}-{}'.format(datasetindex, rstart, rend), res)

	
	
		
		
	
