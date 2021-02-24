"""NNLS"""

import numpy as np
import scipy as sp

def nnls(X, W):
    """Perform NNLS to calculate signature exposures.
    """
    H = []
    for x in X.T:
        h, _ = sp.optimize.nnls(W, x)
        H.append(h)
    H = np.array(H)
    H = H.T
    return H
