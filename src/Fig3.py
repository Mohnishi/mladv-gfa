import importlib
import gfa
importlib.reload(gfa)
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *
import visualize

# Create Corentin's artifical data set corresponding to Fig. 3 (original paper)
R = 3 #rank
K = 7 #factors
D = np.array([10,10,10]) #groups     np.array([2,4,4])
N = 100 #samples
X, W, Z, alpha, Tau = generation(N, K, D, R, constrain_W=10)

print("Noise variance of generated data:", 1 / Tau)
# Extract groups of the true projection mappings
W1 = W[:,:10]
W2 = W[:,10:20]
W3 = W[:,20:30]

# Run GFA
g = gfa.GFA(optimize_method="l-bfgs-b", debug=True, max_iter=10000)
g.fit(X,D)

# Get estimated W (West)
West = g.get_W()

# Extract groups of estimated projection mappings
West1 = West[:,:10]
West2 = West[:,10:20]
West3 = West[:,:30]

# Visualize the group 1 of the true projection mappings
plt.subplot(1, 2, 1)
visualize.plot_W(W.T)
#for k in range(7):
#	for d in range(10):
#		plt.scatter(k+1,d+1,s=W1[k,d],color='black', marker = 's')

plt.title("True W")
plt.ylabel("Group 1")

# Visualize the group 1 of the estimated projection mappings
plt.subplot(1, 2, 2)
visualize.plot_W(visualize.sort_W(W, West).T)
#for k in range(7):
#	for d in range(10):
#		plt.scatter(k+1,d+1,s=West[k,d],color='black', marker = 's')

plt.title("Estimated W")
plt.ylabel("Group 1")

plt.show()
