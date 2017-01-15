import numpy as np
import sklearn.decomposition
import gfa
import time

def infer_gfa(X,D,R,K,N):
    g = gfa.GFA_rep(X,D, n=N, debug_iter=True, debug=False, tol=1e-6,
                    max_iter=10**5, factors=K, rank=R)
    np.save("res/w_our.npy", g.get_W())
    np.save("res/bounds_our.npy", g.get_bounds())

def infer_fa(X,K):
    fac = sklearn.decomposition.FactorAnalysis(n_components=K, tol=1e-6)
    fac.fit(X.T)
    np.save("res/w_fa.npy", fac.components_)

if __name__ == '__main__':
    X = np.load("res/x.npy")
    D = np.load("res/d.npy")
    R,K,rep = np.load("res/params.npy").tolist()

    old = np.load("res/times_our.npy")

    t0 = time.time()
    infer_gfa(X,D,R,K,rep)
    end1 = t0 - time.time()

    t0 = time.time()
    infer_fa(X,K)
    end2 = t0 - time.time()

    times = np.array([end1, end2])

    np.save("res/times_our.npy", old+times)
