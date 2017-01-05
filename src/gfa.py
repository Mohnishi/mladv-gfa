import numpy as np
import scipy.special
import scipy.optimize as opt

def split_and_reshape(flattened, *args):
    """Restore matrices of shapes in 'args' from a flattened 1D vector"""
    split_indices = np.cumsum([np.prod(shape) for shape in args])
    arrays = np.split(flattened, split_indices[:-1])
    return [arr.reshape(shape) for arr, shape in zip(arrays, args)]

def flatten_matrices(*args):
    """Flatten matrices in 'args' to single 1D vector"""
    return np.concatenate([M.flatten() for M in args])

def trprod(A, B):
    """Calculates tr(AB) efficiently using the
    Hadamard product"""
    return (A.T * B).sum()

def GFA_rep(X, D, n=10, **kwargs):
    """Fits the GFA model n times and returns the best fit (maximum lower bound)"""

    models = []
    for i in range(n):
        if kwargs["debug"]:
            print("Fitting model {}...".format(i))
        g = GFA(**kwargs)
        g.fit(X,D)
        if kwargs["debug"]:
            print("Bound:", g.bound())
        models.append(g)

    max_model = max(models, key=lambda g:g.bound())
    index, model = max(enumerate(models), key=lambda x:x[1].bound())
    if kwargs["debug"]:
        print("Returning model {} at bound {}".format(index, model.bound()))
    return model

class GFA:

    def __init__(self, rank=4, factors=7, max_iter=1000, lamb=0.1,
                 a_tau_prior=1e-14, b_tau_prior=1e-14,
                 tol=1e-2, optimize_method="BFGS", debug=False):
        self.lamb = lamb
        self.rank = rank
        self.factors = factors
        self.a_tau_prior = a_tau_prior
        self.b_tau_prior = b_tau_prior

        self.optimize_method = optimize_method
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug

    def fit(self, X, D):
        """Infer latent variables from data and group divisions

        Input:
        X: data array of size d x N, where d is the amount of variables and
           N is the sample size
        D: 1D array specifying the group divisions, i.e. D = [2,3] would mean there
           are two groups, corresponding to variables 1-2 and 3-5 respectively

        Output:
        After running, the inferred parameters will be available as fields
        """

        self.init(X,D)
        self.update_params()
        prev_bound = self.bound()

        for i in range(1, self.max_iter + 1):
            self.update_params()
            new_bound = self.bound()

            if np.abs(new_bound - prev_bound) < self.tol:
                prev_bound = new_bound
                if self.debug:
                    print("Successful fit")
                break

            prev_bound = new_bound

            if (i == 1 or i % 10 == 0) and self.debug:
                print("Lower bound at iteration {}: {}".format(i, prev_bound))
        else: # nobreak
            print("Reach the maximum number of iterations")

        if self.debug:
            print("Took {} iterations".format(i))
            print("Maximal lower bound: {}".format(prev_bound))
        # for debugging calling code
        self.iters = 10

    def update_params(self):
        self.update_W()
        self.update_Z()
        self.update_alpha()
        self.update_tau()

    def init(self, X, D):
        assert D.sum() == X.shape[0]

        # normalize rows (variables) to zero mean and unit variance
        self.X_mean = X.mean(axis=1, keepdims=True)
        self.X_std = X.std(axis=1, keepdims=True)
        X = (X - self.X_mean) / self.X_std

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
        sig = np.random.randn(self.factors, self.factors)
        self.sigma_Z = sig @ sig.T
        self.m_Z = np.random.randn(self.factors, self.N)

        # initialize q(tau)
        # a_tau is constant; set b_tau to a_tau so that E[tau] = 1
        self.a_tau = self.a_tau_prior + self.D * self.N / 2
        self.b_tau = self.a_tau

    def get_W(self):
        W = np.hstack([self.E_W(m) for m in range(self.groups)])
        # scale W using saved std
        return W * self.X_std.T

    def get_Z(self):
        return self.E_Z()

    def bound(self):
        """Get current lower bound of marginal p(Y)
           (may ignore constants with respect to parameters)"""

        # calculate E[log p(X, Theta)]
        p_X = sum((self.N * self.D[m]/2 * self.E_logtau(m) - self.E_tau(m)/2 * (
                np.trace(self.X[m] @ self.X[m].T) -
                2 * np.trace(self.E_W(m).T @ self.E_Z() @ self.X[m].T) +
                np.trace(self.E_WW(m) @ self.E_ZZ()))
            for m in range(self.groups)))

        p_Z = -1/2 * np.trace(self.E_ZZ())

        p_tau = sum(((self.a_tau[m] - 1) * self.E_logtau(m)
                - self.b_tau[m]*self.E_tau(m))
            for m in range(self.groups))

        p_W = (1/2 * (self.D @ np.sum(np.log(self.alpha), axis=1)
                - np.trace(self.alpha.T @ self.E_WW_diag())))

        p_U = -self.lamb/2 * np.sum(self.U**2)
        p_V = -self.lamb/2 * np.sum(self.V**2)

        p = p_X + p_Z + p_tau + p_W + p_U + p_V

        # calculate E[-log q(Theta)] (entropy)
        ent_Z = (self.N/2 * np.log((2*np.pi*np.exp(1))**self.factors *
            np.linalg.det(self.sigma_Z)))


        ent_tau = sum((self.a_tau[m] - np.log(self.b_tau[m]) +
                scipy.special.gammaln(self.a_tau[m]) +
                (1 - self.a_tau[m])*scipy.special.digamma(self.a_tau[m])
            for m in range(self.groups)))

        ent_W = sum((self.D[m]/2 * np.log( (2*np.pi*np.exp(1))**self.factors *
                np.linalg.det(self.sigma_W[m]))
            for m in range(self.groups)))

        ent = ent_Z + ent_tau + ent_W
        return p + ent

    # NOTE: all expectations with regard to q
    def E_tau(self, m):
        """Calculate E[tau(m)]"""
        return self.a_tau[m] / self.b_tau[m]

    def E_logtau(self, m):
        """Calculate E[log tau(m)]"""
        return scipy.special.digamma(self.a_tau[m]) - np.log(self.b_tau[m])

    def E_W(self, m):
        """Calculate E[W(m)]"""
        return self.m_W[m]

    def E_WW(self, m):
        """Calculate E[W(m) W(m).T]
        Size = K x K
        """
        return self.D[m] * self.sigma_W[m] + self.m_W[m] @ self.m_W[m].T

    def E_WW_diag(self):
        """Calculate diagonal of E_WW for all groups
        Size = M x K
        """
        return np.array([np.diag(self.E_WW(m)) for m in range(self.groups)])

    def E_Z(self):
        """Calculate E[Z]"""
        return self.m_Z

    def E_ZZ(self):
        """Calculate E[Z Z.T]"""
        return self.N * self.sigma_Z + self.m_Z @ self.m_Z.T

    # TODO: document simplification of formulas
    def update_W(self):
        """Update W, i.e. update the mean m_W and covariance sigma_W
        of variational distribution

        sigma_W : M-sized vector with K x K-arrays
        m_W : M-sized vector with K x Dm-arrays, Dm = dimentionality of group
        """
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

    def bound_uv(self, x):
        """Return the lower bound as function of U,V,mu_u,mu_v
        ignoring constant terms"""

        U, V, mu_u, mu_v = self.recover_matrices(x)
        ln_alpha = self.ln_alpha(U, V, mu_u, mu_v)
        alpha = np.exp(ln_alpha)

        bound = (sum((self.D[m] * ln_alpha[m,:] - np.diag(self.E_WW(m)) * alpha[m,:]).sum()
                     for m in range(self.groups)) -
                 self.lamb * (np.sum(U**2) + np.sum(V**2)))

        return -bound/2

    def grad_uv(self, x):
        """Return the gradient of U,V,mu_u,mu_v"""

        U, V, mu_u, mu_v = self.recover_matrices(x)

        A = self.D[:,np.newaxis] - self.exp_alpha(U, V, mu_u, mu_v)*self.E_WW_diag()
        grad_U = -(A @ V - U * 2 * self.lamb)/2
        grad_V = -(A.T @ U - V * 2 * self.lamb)/2
        grad_mu_u = -np.sum(A,axis=1)/2
        grad_mu_v = -np.sum(A,axis=0)/2

        return flatten_matrices(grad_U, grad_V, grad_mu_u, grad_mu_v)

    def opt_debug(self,x):
        U, V, mu_u, mu_v = self.recover_matrices(x)
        print("U:\n", U)
        print("V:\n", V)
        print("mu_u\n", mu_u)
        print("mu_v\n", mu_v)
        print("Ln alpha\n", self.ln_alpha(U,V,mu_u,mu_v))
        print("Bound:\n", self.bound(x))

    def update_alpha(self):
        """Update alpha using joint numerical optimization over
        U, V, mu_u and mu_v

        Output:
        returns an OptimizeResult from scipy for debugging purposes
        """
        x0 = np.zeros(np.prod(self.U.shape) + np.prod(self.V.shape) +
                      np.prod(self.mu_u.shape) + np.prod(self.mu_v.shape))

        res = opt.minimize(self.bound_uv, x0, jac=self.grad_uv,
                           method=self.optimize_method)
        if not res.success and self.debug:
            print("Optmization failure")

        self.U,self.V,self.mu_u,self.mu_v = self.recover_matrices(res.x)
        self.alpha = self.get_alpha()

        return res

    def update_tau(self):
        self.b_tau = [self.b_tau_prior +
                      1/2 * (np.trace(self.X[m] @ self.X[m].T) +
                             np.trace(- 2*self.E_W(m) @ self.X[m] @ self.E_Z().T +
                                      self.E_WW(m) @ self.E_ZZ()))
                      for m in range(self.groups)]
