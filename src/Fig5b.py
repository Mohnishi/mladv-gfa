import importlib
import gfa
importlib.reload(gfa)
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
import visualize
import pickle



M = 50
Dm = 10 
D = Dm*np.ones(M)


with open('data-fig5.pkl', 'rb') as f:
    data = pickle.load(f)
rank2_datasets = data['rank-2']
rank6_datasets = data['rank-6']
rank10_datasets = data['rank-10']


R = range(1,21) #model rank

data = [rank2_datasets, rank6_datasets, rank10_datasets]

maxbound = np.zeros((3,21))

plt.figure(1)

for d in range(len(data)):
	
	dataset = data[d]
	
	for r in R: # for all model ranks
		bound = 0
		for i in range(50): #always 50 artificial data sets
			g = gfa.GFA(optimize_method="l-bfgs-b", debug=True, max_iter=10000, factors = 30, rank = r)
			g.fit(dataset[i],D)
			bound = bound + g.bound() 
			
		maxbound[d,r] = bound/50.0
	
		if d == 0: 
			string = 'black'
			plt2 = plt.scatter(r, maxbound[d,r],color='black', marker='o')
		elif d == 1: 
			string = 'red'
			plt6 = plt.scatter(r, maxbound[d,r],color='red', marker='o')
		else: 
			string = 'green'
			plt6 = plt.scatter(r, maxbound[d,r],color='green', marker='o')
	
	
plt.legend((plt2,plt6,plt10),('Data rank 2','Data rank 6', 'Data rank 10'))	
plt.xlabel('Model rank')
plt.ylabel('Lower bound')		

plt.show()
	
		
		
	
