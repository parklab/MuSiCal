"""Non-negative least squares (NNLS) to calculate the signature exposures"""

import numpy as np
from scipy import optimize


def nnls(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    H = np.array([optimize.nnls(W, x)[0] for x in X.T]).T
    return H
