"""Non-negative matrix factorization"""

import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize

from .utils import beta_divergence


EPSILON = np.finfo(np.float32).eps


def _simplex_proj(y):
    D = len(y)
    u = np.sort(y)[::-1]
    tmp = u + 1/np.arange(1, D + 1)*(1 - np.cumsum(u))
    inds = np.arange(0, D)[tmp > 0] + 1
    if len(inds) == 0:
        rho = D
    else:
        rho = inds[-1]
    l = 1/rho*(1 - np.sum(u[0:rho]))
    return np.maximum(y + l, 0)


def _fit_mu(X, n_components, norm=False, max_iter=200, min_iter=100, tol=1e-4,
            conv_test_freq=10, conv_test_baseline=None, verbose=0):
    # Convert to float if they are not
    # Convert to np array in case they are not, e.g., when they are pd DataFrames.
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    if norm:
        X = normalize(X, norm='l1', axis=0)
    #
    n_features, n_samples = X.shape
    # Get co-occurrence matrix and do square root decomposition
    P = X @ X.T
    eigvalue, eigvector = np.linalg.eigh(P)
    eigvalue = eigvalue[::-1]
    eigvector = eigvector[:, ::-1]
    #B = eigvector[:, 0:n_components] @ np.diag(np.sqrt(eigvalue[0:n_components])) @ eigvector[:, 0:n_components].T
    B = eigvector[:, 0:n_components] @ np.diag(np.sqrt(eigvalue[0:n_components]))
    ### Below we solve the problem BQ = WD
    # Initialize:
    Q = np.identity(n_components)
    W = np.random.uniform(0, 1, size=(n_features, n_components))
    D = np.random.uniform(0, 1, size=(n_components, n_components))
    _X = B @ Q
    # Clip small values
    W = W.clip(EPSILON)
    D = D.clip(EPSILON)
    # Initial loss
    loss_init = beta_divergence(_X, W @ D, beta=2, square_root=False)
    # Baseline of convergence test
    if conv_test_baseline is None:
        conv_test_baseline = loss_init
    elif type(conv_test_baseline) is str and conv_test_baseline == 'min-iter':
        pass
    else:
        conv_test_baseline = float(conv_test_baseline)
    # Iteration
    loss_previous = loss_init
    converged = False
    for n_iter in range(1, max_iter + 1):
        for _ in range(0, 1):
            # Update D
            D = D * (W.T @ _X) / (W.T @ W @ D)
            D = D.clip(EPSILON)
            # Update W
            W = W * (_X @ D.T) / (W @ D @ D.T)
            W = np.array([_simplex_proj(w) for w in W.T]).T
            W = W.clip(EPSILON)
            #W = normalize(W, norm='l1', axis=0)
        # Update Q
        Q = sp.linalg.orthogonal_procrustes(B, W @ D, check_finite=False)[0]
        _X = B @ Q
        # Convergence test
        if n_iter == min_iter and conv_test_baseline == 'min-iter':
            loss = beta_divergence(_X, W @ D, beta=2, square_root=False)
            conv_test_baseline = loss
        if n_iter >= min_iter and tol > 0 and n_iter % conv_test_freq == 0:
            loss = beta_divergence(_X, W @ D, beta=2, square_root=False)
            relative_loss_change = (loss_previous - loss) / conv_test_baseline
            if (loss <= loss_previous) and (relative_loss_change <= tol):
                converged = True
            else:
                converged = False
            if verbose:
                print('Epoch %02d reached. Loss: %.3g. Previous loss: %.3g. '
                      'Baseline: %.3g. Relative loss change: %.3g' %
                      (n_iter, loss, loss_previous, conv_test_baseline, relative_loss_change))
            loss_previous = loss
        # If converged, stop
        if converged and n_iter >= min_iter:
            break

    return B, Q, W, D, n_iter, converged


class NMFCOV:
    def __init__(self,
                 X,
                 n_components,
                 norm=False,
                 max_iter=200,
                 min_iter=100,
                 tol=1e-4,
                 conv_test_freq=10,
                 conv_test_baseline=None,
                 verbose=0#,
                 #eng=None
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = n_components
        self.norm = norm
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.verbose = verbose

    def fit(self):
        B, Q, W, D, n_iter, converged = _fit_mu(X=self.X,
                                            n_components=self.n_components,
                                            norm=self.norm,
                                            max_iter=self.max_iter,
                                            min_iter=self.min_iter,
                                            tol=self.tol,
                                            conv_test_freq=self.conv_test_freq,
                                            conv_test_baseline=self.conv_test_baseline,
                                            verbose=self.verbose)
        self.B = B
        self.Q = Q
        self.W = W
        self.D = D
        self.n_iter = n_iter
        self.converged = converged
        return self
