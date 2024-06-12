"""Non-negative matrix factorization with the generalized Kullback-Leibler divergence"""

from __future__ import annotations

import anndata as ad
import numpy as np
import salamander as sal


class NMF:
    """
    Wrapper around the implementation from salamanader.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_components: int,
        init: str = "random",
        init_W_custom: np.ndarray | None = None,
        init_H_custom: np.ndarray | None = None,
        min_iter: int = 100,
        max_iter: int = 200,
        conv_test_freq: int = 10,
        tol: float = 1e-4,
        verbose: Literal[0, 1] = 0,
    ):
        self.X = np.array(X).astype(float)
        self.n_components = n_components
        self.init = init

        if init_W_custom is not None:
            init_W_custom = np.array(init_W_custom).astype(float)
        self.init_W_custom = init_W_custom

        if init_H_custom is not None:
            init_H_custom = np.array(init_H_custom).astype(float)
        self.init_H_custom = init_H_custom

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.conv_test_freq = conv_test_freq
        self.tol = tol
        self.verbose = verbose

    def fit(self) -> NMF:
        if self.init_W_custom is not None and self.init_H_custom is not None:
            self.init = "custom"
            init_kwargs = {
                "signatures_mat": self.init_W_custom.T,
                "exposures_mat": self.init_H_custom.T,
            }
        else:
            init_kwargs = None

        model = sal.models.KLNMF(
            n_signatures=self.n_components,
            init_method=self.init,
            min_iterations=self.min_iter,
            max_iterations=self.max_iter,
            conv_test_freq=self.conv_test_freq,
            tol=self.tol,
        )
        adata = ad.AnnData(self.X.T)
        model.fit(
            adata,
            init_kwargs=init_kwargs,
            verbose=self.verbose,
            verbosity_freq=self.conv_test_freq,
        )
        W = model.asignatures.X.T
        self._W, self.W = W, W
        H = model.adata.obsm["exposures"].T
        self._H, self.H = H, H
        return self
