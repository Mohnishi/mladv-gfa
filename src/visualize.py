import numpy as np
import matplotlib.pyplot as plt

def plot_W(W):
    row_indices, column_indices = np.indices(W.shape)
    plt.scatter(column_indices.flatten(), row_indices.flatten(),
                s=abs(W).flatten(), color='black', marker='s')
                
                
def sort_W(W_real, W_est):
    K = W_real.shape[0]
    sim = np.zeros([K, K])
    for i in range(K):
        for j in range(K):
            sim[i,j] = np.dot(np.abs(W_est[i,:]), np.abs(W_real[j,:])) / (np.linalg.norm(W_est[i,:]) * np.linalg.norm(W_real[j,:]))
    list_max = np.zeros(K, dtype=np.int)
    for m in range(K):
        index_maxi = np.argwhere(sim == np.max(sim))[0]
        list_max[index_maxi[1]] = index_maxi[0]
        for i in range(K):
            sim[index_maxi[0], i] = 0
            sim[i,index_maxi[1]] = 0
    return W_est[list_max,:]