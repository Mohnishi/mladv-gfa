import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import cosine
import math

def plot_W(W):
    row_indices, column_indices = np.indices(W.shape)
    plt.scatter(column_indices.flatten(), row_indices.flatten(),
                s=abs(W).flatten(), color='black', marker='s')


def sort_W_old(W_real, W_est):
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
    
    
    
def sort_W(W_real, W_est):   
    K = W_real.shape[0]
    sim = np.zeros([K,K])
    
    # Matrix of similarity 
    for i in range(K):
        for j in range(K):
            sim[i,j] = 1 - cosine(np.abs(W_est[i,:]), np.abs(W_real[j,:]))
            if math.isnan(sim[i,j]):
                sim[i,j] = 0
    # Compute the similairty for every permutation
    maxi_score = -np.inf
    for permu in itertools.permutations(range(K)):
        score = 0
        for i in range(K):
            score += sim[permu[i], i]
        if score > maxi_score:
            maxi_score = score
            maxi_permu = permu
    return W_est[maxi_permu, :]
    
