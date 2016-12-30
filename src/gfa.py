
import numpy as np
from numpy import exp
import scipy as sp

class GFA:

    def __init__(self, rank=4, factors=7, max_iter=100, lamb=0.1):
        self.lamb = lamb
        self.rank = rank
        self.factors = factors

    # D: m x 1 - matrix
    # X: d x n - matrix
    # sum(D) = d
    def fit(self, X, D):
        self.D = D
        self.X = X
        self.groups = len(D)
        self.U = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb),
                                  size=(self.groups, self.rank))
        self.V = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb),
                                  size=(self.factors, self.rank))
        self.mu_u = np.zeros((self.groups, 1))
        self.mu_v = np.zeros((self.factors, 1))

        self.update_W()
        self.update_alpha()

    def update_W(self):
        # temporary init values
        self.sigma_W = np.random.randn(self.groups,self.factors,sum(self.D))
        self.mu_W = np.random.randn(self.groups,self.factors,sum(self.D))

    def update_Z(self):
        pass

    def ln_alpha(self, U,V,mu_u,mu_v):
        ones_u = np.ones((1, self.factors))
        ones_v = np.ones((self.groups, 1))
        return U @ np.transpose(V) + mu_u @ ones_u + ones_v @ np.transpose(mu_v)

    def alpha(self,U,V,mu_u,mu_v):
        return exp(self.ln_alpha(U,V,mu_u,mu_v))

    def get_alpha(self):
        return self.alpha(self.U, self.V, self.mu_u, self.mu_v)

    # <W(m) W(m)T>_k,k
    def W_square_exp(self,m,k):
        acc = 0
        for i in range(self.D[m]):
            acc += self.sigma_W[m,k,i] + self.mu_W[m,k,i]**2
        return acc

    def recover_matrices(self,x):
        prev = 0
        U = x[prev:self.groups*self.rank].reshape((self.groups, self.rank))
        prev += self.groups*self.rank
        V = x[prev:prev+self.factors*self.rank].reshape((self.factors, self.rank))
        prev += self.factors*self.rank
        mu_u = x[prev:prev+self.groups].reshape((self.groups, 1))
        prev += self.groups
        mu_v = x[prev:prev+self.factors].reshape((self.factors, 1))

        return U,V,mu_u,mu_v

    def flatten_matrices(self,U,V,mu_u,mu_v):
        flattened = list(map(np.ndarray.flatten, [U,V,mu_u,mu_v]))
        return np.concatenate(flattened)

    def bound(self,x):
        U,V,mu_u,mu_v = self.recover_matrices(x)

        ln_alpha = self.ln_alpha(U,V,mu_u,mu_v)
        alpha = exp(ln_alpha)

        acc = 0
        for m in range(self.groups):
            for k in range(self.factors):
                acc += self.D[m,0] * ln_alpha[m,k]
                acc -= self.W_square_exp(m,k) * alpha[m,k]

        # lambda * (tr[UtU] + tr[VtV])
        log_pUV = self.lamb * (np.sum(np.square(U)) + np.sum(np.square(V)))

        return acc - log_pUV

    def get_A(self,U,V,mu_u,mu_v):
        return self.D @ np.ones((1, self.factors)) - self.alpha(U,V,mu_u,mu_v)

    def grad(self,x):
        U,V,mu_u,mu_v = self.recover_matrices(x)

        A = self.get_A(U,V,mu_u,mu_v)
        grad_U = A @ V + U * self.lamb
        grad_V = np.transpose(A) @ U + V * self.lamb
        grad_mu_u = A @ np.ones((self.factors,1))
        grad_mu_v = np.transpose(A) @ np.ones((self.groups,1))

        return self.flatten_matrices(grad_U, grad_V, grad_mu_u, grad_mu_v)

    def update_alpha(self):
        x0 = self.flatten_matrices(self.U, self.V, self.mu_u, self.mu_v)
        grad = self.grad(x0)
        bound = self.bound(x0)
        return bound, (self.recover_matrices(grad))

    def update_tau(self):
        pass
