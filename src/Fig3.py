import importlib
import gfa
importlib.reload(gfa)
import matplotlib.pyplot as plt

import numpy as np
from generate_data import *





# # Example how to use :
# R = 3 #rank
# K = 7 #factors
# Dm = [10,10,10] #groups
# N = 100 #samples
# X, W, Z = generation(N, K, Dm, R)


# print(W.shape)

W = np.ones((7, 30))

X1 = W[:,range(0,10)]
X2 = W[:,range(10,20)]
X3 = W[:,range(20,30)]

K = range(7) 
Dm = range(10)


plt.figure(1)

for k in K:
	for d in Dm:
		plt.scatter(k+1,d+1,s=X1[k,d],color='black', marker = 's')
	
plt.title("True")		

plt.ylabel("Group 1")

plt.figure(2)


for k in K:
	for d in Dm:
		plt.scatter(k+1,d+1,s=X2[k,d],color='black', marker = 's')
	
plt.title("True")		

plt.ylabel("Group 2")

plt.figure(3)

for k in K:
	for d in Dm:
		plt.scatter(k+1,d+1,s=X3[k,d],color='black', marker = 's')
	
plt.title("True")		

plt.ylabel("Group 3")


plt.show()



