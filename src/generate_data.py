# -*- coding: utf-8 -*-

# WARNING :
# The parameters of the gamma distribution used for Tau_m
# are not the right ones yet...

"""
# Example how to use :
R = 2 #rank
K = 7 #factors
Dm = [7,7,7] #groups
N = 10 #samples
X, W, Z = generation(N, K, Dm, R)
"""

import numpy as np
from numpy.random import gamma, normal, multivariate_normal

def generate_UV(M, K, R):
    """ Generation U, V
    Size U = M x R
    Size V = K x R
    """
    lambda0 = 0.1
    sigma = np.sqrt(1/lambda0)
    U = normal(0, sigma, [M, R])
    V = normal(0, sigma, [K, R])
    return U, V

def get_alpha(U, V):
    """ Compute the alpha and A = log(alpha)
    Size A = M x K
    Size alpha = M x K
    """
    m_u = np.mean(U, axis=1)
    m_v = np.mean(V, axis=1)
    m_u = m_u[:, np.newaxis]
    m_v = m_v[:, np.newaxis]
    oneK = np.ones([V.shape[0], 1])
    oneM = np.ones([U.shape[0], 1])
    A = np.dot(U, V.T) + np.dot(m_u, oneK.T) + np.dot(oneM, m_v.T)
    alpha = np.exp(A)
    return A, alpha

def get_w(D, alpha):
    """ Sampling W from Gaussian(0, 1/alpha)
    Size W = K x D
    """
    offset = 0
    W = np.ones([alpha.shape[1], sum(D)])
    for dm in range(len(D)):
        for k in range(alpha.shape[1]):
            m = np.zeros(D[dm])
            s = np.eye(D[dm]) / alpha[dm,k]
            W[k, offset:offset+D[dm]] = multivariate_normal(m, s)
        offset += D[dm]
    return W


def generate_tau(D):
    """ Sampling Tau from Gamma distribution
    Size Tau = M
    """
    p = 1   # Should be 14 but problem with sampling if 14 is used
    shape = pow(10,-p)
    rate = pow(10,-p)
    return gamma(shape, 1/rate, len(D))


def generate_x(Z, W, D, Tau, N):
    """ Sampling x from Gaussian(W*z, 1/Tau)
    Size X = D x N
    """
    offset = 0
    X = np.zeros([sum(D), N])
    for i in range(len(D)):
        m = np.dot(Z[:,i], W[:, offset:offset + D[i]])
        s = np.eye(D[i]) / Tau[i]
        X[offset:offset + D[i], :] = multivariate_normal(m, s, N).T
        offset = offset + D[i]
    return X


def generate_z(K, N):
    """ Sampling z from Gaussian(0, I)
    Size Z = K x N
    """
    return multivariate_normal(np.zeros(K), np.eye(K), N).T

def generation(N, K, D, R):
    """ Complete generation of the data
    Output :
    Size X = D x N
    Size W = D x K
    Size Z = K x N
    """
    M = len(D)
    U, V = generate_UV(M, K, R)
    A, alpha = get_alpha(U, V)
    W = get_w(D, alpha)
    Tau = generate_tau(D)
    Z = generate_z(K,N)
    X = generate_x(Z, W, D, Tau, N)
    return X, W, Z
