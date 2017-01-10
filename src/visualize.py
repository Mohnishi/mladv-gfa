import numpy as np
import matplotlib.pyplot as plt

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
    f = np.sum(W_est, axis = 1) == 0
    P = K - np.sum(f)
    sim = np.zeros([P,K])
    
    # Matrix of similarity
    W_tmp = W_est[np.sum(W_est, axis=1) != 0]    
    for i in range(P):
        for j in range(K):
            sim[i,j] = np.dot(np.abs(W_tmp[i,:]), np.abs(W_real[j,:])) / (np.linalg.norm(W_tmp[i,:]) * np.linalg.norm(W_real[j,:]))
    
    # Compute the similairty for every permutation 
    a = list(iter.permutations(range(K)))
    maxi_score = 0
    maxi_permu = a[0]
    for permu_all in a:
        permu = permu_all[0:P]
        score = 0
        for i in range(P):
            score += sim[i, permu[i]]
        if score > maxi_score:
            maxi_score = score
            maxi_permu = permu_all
    # 
    final_permu = np.asarray(maxi_permu)
    offset_zero = 0
    offset = 0
    for i in range(len(final_permu)):
        if f[i]:
            final_permu[i] = maxi_permu[offset_zero + P]
            offset_zero += 1
        else:
            final_permu[i] = maxi_permu[offset]
            offset += 1
    return W_est[np.argsort(final_permu), :]
    
