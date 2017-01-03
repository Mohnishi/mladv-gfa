#import importlib
#import gfa
#importlib.reload(gfa)
import matplotlib.pyplot as plt

import numpy as np
# dims = 10
# samples = 100
# X = np.random.randn(dims, samples)
# D = np.array([3,4,3])


# g = gfa.GFA(optimize_method="BFGS", debug=True)
# g.fit(X,D)


# #doubling the width of markers
# K = range(1,8) 
# Dm = [0]*len(K)
# #W = [2*4**n for n in range(len(K))]
# W = [1 for n in range(len(K))]

# plt.scatter(K,Dm,s=W,color='black', marker = 's')
# plt.title("True")
# plt.xlim(0, 8)

# plt.show()


W = np.ones((7, 30))
#print(W)
X1 = W[:,range(0,10)]
X2 = W[:,range(10,20)]
X3 = W[:,range(20,30)]

K = range(7) 
Dm = range(10)

X1[5,5] = 20
X1[0,0] = 100

for k in K:
	for d in Dm:
		plt.scatter(k+1,d+1,s=X1[k,d],color='black', marker = 's')
	
plt.title("True")		
print(X1)

plt.show()



