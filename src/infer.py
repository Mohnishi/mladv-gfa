import numpy as np
import sklearn.decomposition
import gfa

def infer_gfa(X,D,R,K,N):
    g = gfa.GFA_rep(X,D, n=N, debug_iter=True, debug=False, tol=1e-6,
                    max_iter=10**5, factors=K, rank=R)
    np.save("res/w_our.npy", g.get_W())
    np.save("res/bounds_our.npy", g.get_bounds())

def infer_fa(X,K):
    fac = sklearn.decomposition.FactorAnalysis(n_components=K)
    fac.fit(X.T)
    np.save("res/w_fa.npy", fac.components_)

if __name__ == '__main__':
    X = np.load("res/x.npy")
    D = np.load("res/d.npy")
    R,K,rep = np.load("res/params.npy").tolist()

    infer_gfa(X,D,R,K,rep)
    infer_fa(X,K)
