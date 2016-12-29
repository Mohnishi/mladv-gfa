
import numpy as np
import scipy as sp

class GFA:

    def __init__(self, rank=4, factors=7, max_iter=100, lamb=0.1,
                 a_tau_prior=1e-14, b_tau_prior=1e-14):
        self.lamb = lamb
        self.rank = rank
        self.factors = factors
        self.a_tau_prior = a_tau_prior
        self.b_tau_prior = b_tau_prior

    # D: m x 1 - matrix
    # X: d x n - matrix
    # Z: k x n - matrix
    # sum(D) = d
    def fit(self, X, D):
        self.D = D
        self.X = X

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

    # <W(m) W(m)T>_k,k
    def W_square_exp(self, m, k):
        acc = 0
        for i in range(self.D[m]):
            acc += self.sigma_W[m][k,i] + self.m_W[m][k,i]**2
        return acc

    def bound(self, U, V, mu_u, mu_v):
        ln_alpha = self.ln_alpha(U, V, mu_u, mu_v)
        alpha = np.exp(ln_alpha)

        acc = 0
        for m in range(self.groups):
            for k in range(self.factors):
                acc += self.D[m] * ln_alpha[m,k]
                acc -= self.W_square_exp(m,k) * alpha[m,k]

        # lambda * (tr[UtU] + tr[VtV])
        log_pUV = self.lamb * (np.sum(np.square(U)) + np.sum(np.square(V)))

        return acc - log_pUV

    def get_A(self, U, V, mu_u, mu_v):
        # add singular dimension to broadcast correctly
        return self.D[:,np.newaxis] - self.exp_alpha(U, V, mu_u, mu_v)

    def grad(self, U, V, mu_u, mu_v):
        A = self.get_A(U, V, mu_u, mu_v)
        grad_U = A @ V + U * self.lamb
        grad_V = np.transpose(A) @ U + V * self.lamb
        grad_mu_u = A @ np.ones((self.factors,1))
        grad_mu_v = np.transpose(A) @ np.ones((self.groups,1))
        return (grad_U, grad_V, grad_mu_u, grad_mu_v)

    # TODO: convert matrices to flat arrays for use in scipy.optimize
    def update_alpha(self):
        grad = self.grad(self.U, self.V, self.mu_u, self.mu_v)
        bound = self.bound(self.U, self.V, self.mu_u, self.mu_v)
        return bound, grad

    def update_tau(self):
        self.b_tau = [self.b_tau_prior +
                      1/2 * (np.trace(self.X[m] @ self.X[m].T) +
                             np.trace(- 2*self.E_W(m) @ self.X[m] @ self.E_Z().T +
                                      self.E_WW(m) @ self.E_ZZ()))
            for m in range(self.groups)]
