
import numpy as np
import scipy as sp
import scipy.optimize as opt


def split_and_reshape(flattened, *args):
    split_indices = np.cumsum([np.prod(shape) for shape in args])
    arrays = np.split(flattened, split_indices[:-1])
    return [arr.reshape(shape) for arr, shape in zip(arrays, args)]

def flatten_matrices(*args):
    return np.concatenate([M.flatten() for M in args])

class GFA:

    def __init__(self, rank=4, factors=7, max_iter=100, lamb=0.1,
                 a_tau_prior=1e-14, b_tau_prior=1e-14, optimize_method="BFGS", debug=False):
        self.lamb = lamb
        self.rank = rank
        self.factors = factors
        self.a_tau_prior = a_tau_prior
        self.b_tau_prior = b_tau_prior

        self.optimize_method = optimize_method
        self.debug = debug

    # D: m x 1 - matrix
    # X: d x n - matrix
    # Z: k x n - matrix
    # sum(D) = d
    def fit(self, X, D):
        self.init(X,D)
        # do until convergence
        self.update_W()
        self.update_Z()
        self.update_alpha()
        self.update_tau()

    def init(self, X, D):
        assert D.sum() == X.shape[0]
        self.groups = len(D)
        split_indices = np.add.accumulate(D[:-1])
        self.X = np.split(X, split_indices)
        self.D = D
        self.N = X.shape[1]

        # initialize alpha
        self.U = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb),
                                  size=(self.groups, self.rank))
        self.V = np.random.normal(loc=0, scale=np.sqrt(1/self.lamb),
                                  size=(self.factors, self.rank))
        self.mu_u = np.zeros((self.groups, 1))
        self.mu_v = np.zeros((self.factors, 1))
        self.alpha = self.get_alpha()

        # initialize q(Z)
        # TODO: investigate effect of initialization
        self.sigma_Z = np.eye(self.factors) + 0.1
        self.m_Z = np.random.randn(self.factors, self.N)

        # initialize q(tau)
        # a_tau is constant; set b_tau to a_tau so that E[tau] = 1
        self.a_tau = self.a_tau_prior + self.D * self.N / 2
        self.b_tau = self.a_tau

    def E_tau(self, m):
        """Calculate E[tau(m)]"""
        return self.a_tau[m] / self.b_tau[m]

    def E_W(self, m):
        """Calculate E[W(m)]"""
        return self.m_W[m]

    def E_WW(self, m):
        """Calculate E[W(m) W(m).T]"""
        return self.D[m] * self.sigma_W[m] + self.m_W[m] @ self.m_W[m].T

    def E_WW_diag(self):
        """Calculate diagonal of E_WW for all groups"""
        return np.array([np.diag(self.E_WW(m)) for m in range(self.groups)])

    def E_Z(self):
        """Calculate E[Z]"""
        return self.m_Z

    def E_ZZ(self):
        """Calculate E[Z Z.T]"""
        return self.N * self.sigma_Z + self.m_Z @ self.m_Z.T

    # TODO: document simplification of formulas
    def update_W(self):
        self.sigma_W = [
            np.linalg.inv(self.E_tau(m) * self.E_ZZ() + np.diag(self.alpha[m]))
            for m in range(self.groups)]
        self.m_W = [self.E_tau(m) * self.sigma_W[m] @ self.E_Z() @ self.X[m].T
                    for m in range(self.groups)]

    def update_Z(self):
        self.sigma_Z = np.linalg.inv(np.eye(self.factors) +
                                     sum(self.E_tau(m) * self.E_WW(m)
                                         for m in range(self.groups)))
        self.m_Z = self.sigma_Z @ sum(self.E_tau(m) * self.E_W(m) @ self.X[m]
                                      for m in range(self.groups))


    def ln_alpha(self, U, V, mu_u, mu_v):
        # this is equivalent to the original formula thanks to broadcasting
        return U @ V.T + mu_u + mu_v.T

    def exp_alpha(self, U, V, mu_u, mu_v):
        return np.exp(self.ln_alpha(U, V, mu_u, mu_v))

    def get_alpha(self):
        return self.exp_alpha(self.U, self.V, self.mu_u, self.mu_v)

    def recover_matrices(self, x):
        return split_and_reshape(x, (self.groups, self.rank), (self.factors, self.rank),
                                 (self.groups, 1), (self.factors, 1))

    def bound(self, x):
        U, V, mu_u, mu_v = self.recover_matrices(x)
        ln_alpha = self.ln_alpha(U, V, mu_u, mu_v)
        alpha = np.exp(ln_alpha)

        bound = (sum((self.D[m] * ln_alpha[m,:] - np.diag(self.E_WW(m)) * alpha[m,:]).sum()
                     for m in range(self.groups)) -
                 self.lamb * (np.sum(U**2) + np.sum(V**2)))

        if self.debug:
            print("Objective eval:")
            print("Bound: {}".format(-bound))
            print("Ln alpha:")
            print(ln_alpha)
        return -bound/2

    def grad(self, x):
        U, V, mu_u, mu_v = self.recover_matrices(x)

        # E_WW used in CCAGFA package at CRAN
        A = self.D[:,np.newaxis] - self.exp_alpha(U, V, mu_u, mu_v)*self.E_WW_diag()
        grad_U = -(A @ V - U * 2 * self.lamb)/2
        grad_V = -(A.T @ U - V * 2 * self.lamb)/2
        grad_mu_u = -np.sum(A,axis=1)/2
        grad_mu_v = -np.sum(A,axis=0)/2

        return flatten_matrices(grad_U, grad_V, grad_mu_u, grad_mu_v)

    def opt_debug(self,x):
        U, V, mu_u, mu_v = self.recover_matrices(x)
        print("U:\n", U)
        print("V:\n", U)
        print("mu_u\n", mu_u)
        print("mu_v\n", mu_v)
        print("Ln alpha\n", self.ln_alpha(U,V,mu_u,mu_v))

    def update_alpha(self):
        x0 = flatten_matrices(self.U, self.V, self.mu_u, self.mu_v)
        if self.debug:
            print("Values before")
            self.opt_debug(x0)

        res = opt.minimize(self.bound, x0, jac=self.grad,
                           method=self.optimize_method, options={"disp": True})
        self.U,self.V,self.mu_u,self.mu_v = self.recover_matrices(res.x)
        self.alpha = self.get_alpha()

        if self.debug:
            print("Values After")
            self.opt_debug(res.x)
            print("Gradients After")
            self.opt_debug(res.jac)

        return res # just for testing purposes

    def update_tau(self):
        self.b_tau = [self.b_tau_prior +
                      1/2 * (np.trace(self.X[m] @ self.X[m].T) +
                             np.trace(- 2*self.E_W(m) @ self.X[m] @ self.E_Z().T +
                                      self.E_WW(m) @ self.E_ZZ()))
                      for m in range(self.groups)]
