# -*- coding: utf-8 -*-

import numpy as np

def generation_filter_first_exp(K):
    """ Return a filter same size as alpha
    This filter could be used in the first experiment
    Size F : M x K
    """
    if K == 1:
        result = np.ones([2,1])
        result[1] = 0
        return result
    else:
        temp = generation_filter_first_exp(K-1)        
        a = np.concatenate((np.ones([2**(K-1), 1]), temp), axis = 1)
        b = np.concatenate((np.zeros([2**(K-1), 1]), temp), axis = 1)
        return np.concatenate((a,b), axis = 0)
        
def gen_filter(K):
    F = generation_filter_first_exp(K)
    result = np.zeros(F.shape)    
    offset = 0
    for i in range(F.shape[0]):
        temp = F[np.sum(F, axis=1) == K + 1 - i]
        result[offset:offset + temp.shape[0],:] = temp
        offset += temp.shape[0]
    return result[0:(2**K) - 1,:].T