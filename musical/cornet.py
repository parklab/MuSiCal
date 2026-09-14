"""Cornet

Cornet is implemented in Sonata (https://github.com/parklab/Sonata). This module
provides a thin wrapper around it, exposing the same interface as musical.nmf.NMF
and musical.mvnmf.MVNMF, such that DenovoSig can use the three algorithms
interchangeably.

Sonata is an optional dependency. It is imported inside Cornet.fit(), so that
MuSiCal can be imported and used without Sonata installed.
"""

import numpy as np
from sklearn.preprocessing import normalize

from .utils import beta_divergence
from .nnls import nnls


# Initialization methods supported by Sonata. MuSiCal additionally supports
# 'cluster' and 'spa', which are not available for Cornet.
INIT_METHODS_CORNET = ['custom', 'flat', 'nndsvd', 'nndsvda', 'nndsvdar',
                       'random', 'separableNMF']


class Cornet:
    """Cornet

    Parameters
    ----------
    X : array-like of shape (n_features, n_samples)
        The count matrix supplied to Cornet. When used through DenovoSig, this is
        the matrix after bootstrapping, normalization, and TMB rescaling.

    n_components : int
        Number of signatures.

    init : str
        Initialization method. Must be one of INIT_METHODS_CORNET. Note that
        'cluster' and 'spa', which are available for NMF and mvNMF in MuSiCal,
        are not supported by Cornet.

    dim_embeddings : int or None
        Dimension of the signature and sample embeddings. If None, n_components
        is used, i.e., the case of independent signatures. Smaller values enforce
        a stronger correlation structure.

    X_raw : array-like of shape (n_features, n_samples) or None
        The raw count matrix, before bootstrapping, normalization and TMB
        rescaling. H is obtained by NNLS of X_raw on the normalized signatures,
        such that H is on the same scale as the raw counts. If None, X is used.

    seed : int or None
        Seed passed to Sonata's parameter initialization.

    keep_model : bool
        If True, the underlying Sonata model object is retained as self.model.
        It is discarded by default, since it carries the full AnnData object and
        DenovoSig only requires W and H.

    Notes
    ----------
    1. All parameters are passed to Sonata explicitly, including those for which
    Sonata's default would currently be the same. This insulates MuSiCal from
    changes of the default values on the Sonata side.
    2. Sonata's fit() does not report the number of iterations performed. We
    derive n_iter and converged from the length of the recorded objective
    function history, which is appended to every conv_test_freq iterations.
    """
    def __init__(self,
                 X,
                 n_components,
                 init='random',
                 dim_embeddings=None,
                 max_iter=20000,
                 min_iter=2000,
                 tol=1e-8,
                 conv_test_freq=10,
                 X_raw=None,
                 seed=None,
                 keep_model=False,
                 verbose=0
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        if X_raw is None:
            X_raw = X
        elif (type(X_raw) != np.ndarray) or (not np.issubdtype(X_raw.dtype, np.floating)):
            X_raw = np.array(X_raw).astype(float)
        self.X_raw = X_raw
        self.n_components = n_components
        if init not in INIT_METHODS_CORNET:
            raise ValueError(
                'Invalid init for Cornet. Valid options are %r. Note that the '
                'cluster and spa initializations available for NMF and mvNMF are '
                'not supported by Cornet.' % (INIT_METHODS_CORNET,)
            )
        self.init = init
        if dim_embeddings is None:
            dim_embeddings = n_components
        self.dim_embeddings = dim_embeddings
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.seed = seed
        self.keep_model = keep_model
        self.verbose = verbose

    def fit(self):
        try:
            import sonata
            from anndata import AnnData
        except ImportError:
            raise ImportError(
                'Cornet requires Sonata, which is not installed. '
                'Install it with: pip install sonata-tools'
            )

        # Sonata expects an AnnData of shape (n_samples, n_features)
        adata = AnnData(self.X.T.copy())
        model = sonata.models.Cornet(
            n_signatures=self.n_components,
            init_method=self.init,
            dim_embeddings=self.dim_embeddings,
            min_iterations=self.min_iter,
            max_iterations=self.max_iter,
            conv_test_freq=self.conv_test_freq,
            tol=self.tol
        )
        model.fit(adata,
                  init_kwargs={'seed': self.seed},
                  history=True,
                  verbose=self.verbose)

        _W = model.signatures.T.values
        _H = model.exposures.T.values
        # Normalize W and perform NNLS to recalculate H. Note that NNLS is done
        # against X_raw, so that H is on the scale of the raw counts. This matters
        # because X is TMB rescaled, and because downstream DenovoSig.postprocess
        # evaluates all reconstruction errors against the raw count matrix.
        W = normalize(_W, norm='l1', axis=0)
        H = nnls(self.X_raw, W)
        #
        self._W = _W
        self._H = _H
        self._reconstruction_error = beta_divergence(self.X, self._W @ self._H, beta=1, square_root=False)
        #
        self.W = W
        self.H = H
        self.reconstruction_error = beta_divergence(self.X_raw, self.W @ self.H, beta=1, square_root=False)
        #
        n_conv_tests = len(model.history['objective_function'])
        self.n_iter = n_conv_tests * self.conv_test_freq
        self.converged = self.n_iter < self.max_iter
        if self.keep_model:
            self.model = model
        return self
