import importlib
import gfa
importlib.reload(gfa)
import matplotlib.pyplot as plt
import numpy as np
from generate_data import *


# Create Corentin's artifical data set corresponding to Fig. 3 (original paper)
R = 3 #rank
K = 7 #factors
D = np.array([10,10,10]) #groups     np.array([2,4,4])
N = 100 #samples
X, W, Z = generation(N, K, D, R)



# Extract groups of the true projection mappings 
W1 = W[:,range(0,10)]
W2 = W[:,range(10,20)]
W3 = W[:,range(20,30)]



# Run GFA 
g = gfa.GFA(optimize_method="bfgs", debug=True)
g.fit(X,D)

# Get estimated W (West)
West = g.m_W
# Extract groups of estimated projection mappings
West1 = g.m_W[0]
West2 = g.m_W[1]
West3 = g.m_W[2]


# Visualize the group 1 of the true projection mappings 
plt.figure(1)
for k in range(7):
	for d in range(10):
		plt.scatter(k+1,d+1,s=W1[k,d],color='black', marker = 's')
	
plt.title("True W")		
plt.ylabel("Group 1")

# Visualize the group 1 of the estimated projection mappings
plt.figure(2)
for k in range(7):
	for d in range(10):
		plt.scatter(k+1,d+1,s=West1[k,d],color='black', marker = 's')
	
plt.title("Estimated W")		
plt.ylabel("Group 1")

plt.show()