"""Non-negative matrix factorization"""

import numpy as np
from sklearn.preprocessing import normalize

from .utils import beta_divergence
from .initialization import initialize_nmf
from .nnls import nnls


EPSILON = np.finfo(np.float32).eps


def _fit_mu(X, W, H, solver='1999-Lee', max_iter=200, min_iter=100, tol=1e-4,
            conv_test_freq=10, conv_test_baseline=None, verbose=0):
    """Multiplicative-update solver.

    MU solver for NMF, following 1999 Lee and Seung paper. Values smaller than
    EPSILON (e.g., 0) in W and H are set to EPSILON, following 2008 Gillis and
    Glineur paper.

    Parameters
    ----------
    X : array-like of shape (n_features, n_samples)
        Constant input matrix.

    W : array-like of shape (n_features, n_components)
        Initial guess.

    H : array-like of shape (n_components, n_samples)
        Initial guess.

    solver : str, '1999-Lee' | '2001-Lee'
        Algorithm.

    max_iter : int, default=200
        Maximum number of iterations.

    min_iter : int, default=100
        Minimum number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    conv_test_freq : int, default=10
        Convergence test frequency. Convergence test is performed every conv_test_freq iterations.

    conv_test_baseline : float, default=None
        Baseline for convergence test. If None, the initial loss is taken as the baseline.

    verbose : int, default=0
        Verbosity level.

    Notes
    ----------
    1. We can potentially speed up the calculation by using *= instead of explicit
    self multiplication. *= is faster for large matrices. Note that, however,
    *= updates matrices in place. So the input W and H will be updated in place,
    resulting in the initial values of W and H being lost. So we should always
    provide an np.copy instance of the input. Also, we should do in place
    calculations for W = W.clip(EPSILON) and W = normalize(W, norm='l1', axis=0)
    too. Finally, if we do everything in place, I'm not sure if we can still do
    save_iter.
    2. For the final output, we can potentially set values close to EPSILON to
    0 in W. Alternatively, we can do a final iteration which does not do clipping.
    3. When solver = '2001-Lee', it is the algorithm used by SigProfiler.
    """
    # Convert to float if they are not
    # Convert to np array in case they are not, e.g., when they are pd DataFrames.
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    if (type(W) != np.ndarray) or (not np.issubdtype(W.dtype, np.floating)):
        W = np.array(W).astype(float)
    if (type(H) != np.ndarray) or (not np.issubdtype(H.dtype, np.floating)):
        H = np.array(H).astype(float)
    n_features, n_samples = X.shape
    n_components = W.shape[1]
    # Clip small values
    W = W.clip(EPSILON)
    H = H.clip(EPSILON)
    # Initial loss
    loss_init = beta_divergence(X, W @ H, beta=1, square_root=False)
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
    if solver == '1999-Lee':
        for n_iter in range(1, max_iter + 1):
            # Update W
            W = W * ((X/(W @ H)) @ H.T)
            W = normalize(W, norm='l1', axis=0) # This is crucial
            W = W.clip(EPSILON)
            # Update H
            H = H * (W.T @ (X/(W @ H)))
            H = H.clip(EPSILON)
            # Convergence test
            if n_iter == min_iter and conv_test_baseline == 'min-iter':
                loss = beta_divergence(X, W @ H, beta=1, square_root=False)
                conv_test_baseline = loss
            if n_iter >= min_iter and tol > 0 and n_iter % conv_test_freq == 0:
                loss = beta_divergence(X, W @ H, beta=1, square_root=False)
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
    elif solver == '2001-Lee':
        for n_iter in range(1, max_iter + 1):
            # Update W
            W = W * ((X/(W @ H)) @ H.T) / np.tile(np.sum(H, 1), (n_features, 1))
            W = W.clip(EPSILON)
            # Update H
            H = H * (W.T @ (X/(W @ H))) / np.tile(np.sum(W, 0), (n_samples, 1)).T
            H = H.clip(EPSILON)
            # Convergence test
            if n_iter == min_iter and conv_test_baseline == 'min-iter':
                loss = beta_divergence(X, W @ H, beta=1, square_root=False)
                conv_test_baseline = loss
            if n_iter >= min_iter and tol > 0 and n_iter % conv_test_freq == 0:
                loss = beta_divergence(X, W @ H, beta=1, square_root=False)
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
    else:
        raise ValueError('solver must be either 1999-Lee or 2001-Lee.')

    return W, H, n_iter, converged


class NMF:
    """NMF

    Notes
    ----------
    1. We hide most optional parameters related to initialization. If one wants to change
    those optional parameters from default, they can first run initialize_nmf() with the desired
    parameters, and then supply W and H in the init=custom mode.
    2. I removed eng from __init__ and did not set eng as an attribute. Otherwise pickle will have
    a problem when saving the class instance, because pickle does not deal with matlab well.
    """
    def __init__(self,
                 X,
                 n_components,
                 init='random',
                 init_W_custom=None,
                 init_H_custom=None,
                 solver='1999-Lee',
                 max_iter=200,
                 min_iter=100,
                 tol=1e-4,
                 conv_test_freq=10,
                 conv_test_baseline=None,
                 verbose=0
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = n_components
        self.init = init
        if init_W_custom is not None:
            if (type(init_W_custom) != np.ndarray) or (not np.issubdtype(init_W_custom.dtype, np.floating)):
                init_W_custom = np.array(init_W_custom).astype(float)
        if init_H_custom is not None:
            if (type(init_H_custom) != np.ndarray) or (not np.issubdtype(init_H_custom.dtype, np.floating)):
                init_H_custom = np.array(init_H_custom).astype(float)
        self.init_W_custom = init_W_custom
        self.init_H_custom = init_H_custom
        self.solver = solver
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.verbose = verbose

    def fit(self):
        W_init, H_init = initialize_nmf(self.X, self.n_components,
                                        init=self.init,
                                        init_W_custom=self.init_W_custom,
                                        init_H_custom=self.init_H_custom)
        self.W_init = W_init
        self.H_init = H_init

        _W, _H, n_iter, converged = _fit_mu(X=self.X,
                                            W=self.W_init, H=self.H_init,
                                            solver=self.solver,
                                            max_iter=self.max_iter,
                                            min_iter=self.min_iter,
                                            tol=self.tol,
                                            conv_test_freq=self.conv_test_freq,
                                            conv_test_baseline=self.conv_test_baseline,
                                            verbose=self.verbose)
        # Normalize W and perform NNLS to recalculate H
        W = normalize(_W, norm='l1', axis=0)
        H = nnls(self.X, W)
        #
        self._W = _W
        self._H = _H
        self._reconstruction_error = beta_divergence(self.X, self._W @ self._H, beta=1, square_root=False)
        #
        self.W = W
        self.H = H
        self.reconstruction_error = beta_divergence(self.X, self.W @ self.H, beta=1, square_root=False)
        #
        self.n_iter = n_iter
        self.converged = converged
        return self
