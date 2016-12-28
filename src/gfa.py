
import numpy as np

class GFA:

    def __init__(self, rank=4, factors=7, max_iter=100, lamb=0.1):
        self.lamb = lamb
        self.rank = rank
        self.factors = factors

    def fit(self, X, D):
        # D = np.array([...])
        self.groups = len(D)
        self.U = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb),
                                  size=(self.groups, self.rank))
        self.V = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb),
                                  size=(self.factors, self.rank))
        self.mu_u = np.zeros((self.groups, 1))
        self.mu_v = np.zeros((self.factors, 1))

    def update_W(self):
        # self.sigma_W[group,i,j]
        # self.mu_W[group,column,i]
        pass

    def update_Z(self):
        pass

    def update_alpha(self):
        pass

    def update_tau(self):
        pass
