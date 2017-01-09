import numpy as np
import gfa
import sys

if __name__ == '__main__':
    X = np.load("res/x.npy")
    D = np.load("res/d.npy")

    n = int(sys.argv[1])

    g = gfa.GFA_rep(X,D, n=n, debug_iter=True, debug=False, tol=1e-6,
                    max_iter=10**5)
    np.save("res/w_our.npy", g.get_W())
    np.save("res/bounds_our.npy", g.get_bounds())
