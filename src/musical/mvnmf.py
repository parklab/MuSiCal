"""
Minimum-volume non-negative matrix factorization with the Kullback-Leibler divergence.

TODO
----
1. Parallelize wrappedMVNMF. The problem is that, DenovoSig already parallelizes
    multiple runs of wrappedMVNMF. If inside wrappedMVNMF there is also parallelization,
    then there will be problems. I'm not sure if there is a workaround.
"""

from __future__ import annotations

import multiprocessing
import os
import warnings
from typing import Literal

import anndata as ad
import numpy as np
import salamander as sal
import scipy.stats as stats

from .utils import _samplewise_error, differential_tail_test

EPSILON = np.finfo(np.float16).eps

# fmt: off
LAMBDA_TILDE_GRID = np.array(
    [
        1e-10, 2e-10, 5e-10, 1e-9, 2e-9, 5e-9, 1e-8, 2e-8, 5e-8,
        1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
        1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2,
        1e-1, 2e-1, 5e-1, 1.0, 2.0,
    ]
)
# fmt: on


class MVNMF:
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
        lambda_tilde: float = 1e-5,
        delta: float = 1.0,
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

        self.lambda_tilde = lambda_tilde
        self.delta = delta
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.conv_test_freq = conv_test_freq
        self.tol = tol
        self.verbose = verbose

    def _get_lam(self) -> float:
        """
        Balance the volume regularization and the reconstruction error
        by scaling lambda_tilde.
        """
        klnmf_model = sal.models.KLNMF(
            n_signatures=self.n_components,
            init_method="random",
            min_iterations=1,
            max_iterations=50,
        )
        adata = ad.AnnData(self.X.T)
        klnmf_model.fit(adata)
        volume = sal.models.mvnmf.volume_logdet(klnmf_model.asignatures.X.T, self.delta)
        lam = self.lambda_tilde * klnmf_model.reconstruction_error / volume
        return lam

    def fit(self) -> MVNMF:
        if self.init_W_custom is not None and self.init_H_custom is not None:
            self.init = "custom"
            init_kwargs = {
                "signatures_mat": self.init_W_custom.T,
                "exposures_mat": self.init_H_custom.T,
            }
        else:
            init_kwargs = None

        self.lam = self._get_lam()
        model = sal.models.MvNMF(
            n_signatures=self.n_components,
            init_method=self.init,
            lam=self.lam,
            delta=self.delta,
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
        self.W = model.asignatures.X.T
        self.H = model.adata.obsm["exposures"].T
        self.loss = model.objective_function()
        self.volume = sal.models.mvnmf.volume_logdet(model.asignatures.X.T, self.delta)
        self.reconstruction_error = model.reconstruction_error
        return self


class wrappedMVNMF:
    """
    mvNMF with automatic selection of lambda_tilde.

    Notes
    -----
    1. I removed eng from __init__ and did not set eng as an attribute. Otherwise pickle will have
    a problem when saving the class instance, because pickle does not deal with matlab well.
    2. Alternative methods for selecting lambda_tilde: e.g., require that the reconstruction error is within
    (1 + thresh) * NMF reconstruction error, where thresh could be 0.1 for example.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_components: int,
        lambda_tilde_grid: np.ndarray | None = None,
        pthresh: float = 0.05,
        init: str = "random",
        init_W_custom: np.ndarray | None = None,
        init_H_custom: np.ndarray | None = None,
        delta: float = 1.0,
        min_iter: int = 100,
        max_iter: int = 200,
        tol: float = 1e-4,
        conv_test_freq: int = 10,
        ncpu: int = 1,
        noise: bool = False,  # Whether or not to add noise to the samplewise errors.
        verbose: Literal[0, 1] = 0,
    ):
        self.X = np.array(X).astype(float)
        self.n_features, self.n_samples = self.X.shape
        self.n_components = n_components

        if lambda_tilde_grid is None:
            lambda_tilde_grid = LAMBDA_TILDE_GRID
        self.lambda_tilde_grid = lambda_tilde_grid

        self.pthresh = pthresh
        self.init = init

        if init_W_custom is not None:
            init_W_custom = np.array(init_W_custom).astype(float)
        self.init_W_custom = init_W_custom

        if init_H_custom is not None:
            init_H_custom = np.array(init_H_custom).astype(float)
        self.init_H_custom = init_H_custom

        self.delta = delta
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.conv_test_freq = conv_test_freq
        self.tol = tol

        if ncpu is None:
            ncpu = os.cpu_count()
        self.ncpu = ncpu

        if type(noise) is bool:
            if noise:
                self.noise = EPSILON
            else:
                self.noise = noise
        elif np.issubdtype(type(noise), np.floating):
            self.noise = noise

        self.verbose = verbose

    def _job(self, lambda_tilde):
        np.random.seed()  # This is critical: https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing
        # For this _job(), the line above is not necessary, since there isn't any randomness in the codes below.
        # However, I think it is generally a good practice to add the seeding line in any parallel job.
        model = MVNMF(
            self.X,
            self.n_components,
            init="custom",
            init_W_custom=self.W_init,
            init_H_custom=self.H_init,
            lambda_tilde=lambda_tilde,
            delta=self.delta,
            min_iter=self.min_iter,
            max_iter=self.max_iter,
            conv_test_freq=self.conv_test_freq,
            tol=self.tol,
            verbose=0,
        )
        model.fit()
        if self.verbose:
            print(f"mvNMF with lambda_tilde = {lambda_tilde:.5f} finished.")
        return model

    def _initialize(self):
        """
        Identical initialization for all runs.
        """
        if self.init_W_custom is not None and self.init_H_custom is not None:
            self.init = "custom"
            init_kwargs = {
                "signatures_mat": self.init_W_custom.T,
                "exposures_mat": self.init_H_custom.T,
            }
        else:
            init_kwargs = {}

        signatures_mat, exposures_mat = sal.initialization.initialize.initialize_mat(
            self.X.T, n_signatures=self.n_components, method=self.init, **init_kwargs
        )
        self.W_init = signatures_mat.T
        self.H_init = exposures_mat.T

    def fit(self):
        self._initialize()

        if self.ncpu == 1:
            # We separate out ncpu == 1 case, such that in DenovoSig, we do not run into issues
            # when we create workers both outside and inside of wrappedMVNMF.
            models = []
            for lambda_tilde in self.lambda_tilde_grid:
                if self.verbose:
                    print("==============================================")
                    print("Running mvNMF with lambda_tilde = %.5g......" % lambda_tilde)
                model = MVNMF(
                    self.X,
                    self.n_components,
                    init="custom",
                    init_W_custom=self.W_init,
                    init_H_custom=self.H_init,
                    lambda_tilde=lambda_tilde,
                    delta=self.delta,
                    min_iter=self.min_iter,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    conv_test_freq=self.conv_test_freq,
                    verbose=self.verbose,
                )
                model.fit()
                models.append(model)
        else:
            workers = multiprocessing.Pool(self.ncpu)
            models = workers.map(self._job, list(self.lambda_tilde_grid))
            workers.close()
            workers.join()

        self.lam_grid = np.array([model.lam for model in models])
        self.loss_grid = np.array([model.loss for model in models])
        self.reconstruction_error_grid = np.array(
            [model.reconstruction_error for model in models]
        )
        self.model_grid = models

        ############# Select the best model ###############
        # First calculate sample-wise errors
        self.samplewise_reconstruction_errors_grid = np.array(
            [_samplewise_error(self.X, model.W @ model.H) for model in models]
        )
        # Then perform statistical tests
        # Alternative tests we can use: ks_2samp, ttest_ind (perhaps on log errors)
        self.pvalue_grid = np.array(
            [
                stats.mannwhitneyu(
                    self.samplewise_reconstruction_errors_grid[0, :],
                    self.samplewise_reconstruction_errors_grid[i + 1, :],
                    alternative="less",
                )[1]
                for i in range(0, len(self.lambda_tilde_grid) - 1)
            ]
        )
        self.pvalue_tail_grid = np.array(
            [
                differential_tail_test(
                    self.samplewise_reconstruction_errors_grid[0, :],
                    self.samplewise_reconstruction_errors_grid[i + 1, :],
                    percentile=90,
                    alternative="less",
                )[1]
                for i in range(0, len(self.lambda_tilde_grid) - 1)
            ]
        )
        # If no noise is added, the pvalues are directly used.
        if type(self.noise) is bool:
            self.pvalue_indicator_grid = self.pvalue_grid <= self.pthresh
            self.pvalue_tail_indicator_grid = self.pvalue_tail_grid <= self.pthresh
        # Otherwise, we add noise and do the tests again. We do it multiple times and take the majority vote.
        else:
            # Output a warning whenever this is done
            warnings.warn(
                "Random noise between %.3g and %.3g is added to the samplewise errors. Make sure this makes sense."
                % (-self.noise, self.noise),
                UserWarning,
            )
            self.pvalue_indicator_grid = []
            self.pvalue_tail_indicator_grid = []
            for i in range(0, len(self.lambda_tilde_grid) - 1):
                ps = []
                ps_tail = []
                for _ in range(0, 51):
                    _x1 = self.samplewise_reconstruction_errors_grid[
                        0, :
                    ] + np.random.uniform(-self.noise, self.noise, self.n_samples)
                    _x2 = self.samplewise_reconstruction_errors_grid[
                        i + 1, :
                    ] + np.random.uniform(-self.noise, self.noise, self.n_samples)
                    _offset = np.min([_x1, _x2])
                    ps.append(stats.mannwhitneyu(_x1, _x2, alternative="less")[1])
                    # We need to make everything positive to do the differential tail test.
                    if _offset < 0:
                        ps_tail.append(
                            differential_tail_test(
                                _x1 - _offset,
                                _x2 - _offset,
                                percentile=90,
                                alternative="less",
                            )[1]
                        )
                    else:
                        ps_tail.append(
                            differential_tail_test(
                                _x1, _x2, percentile=90, alternative="less"
                            )[1]
                        )
                ps = np.array(ps)
                ps_tail = np.array(ps_tail)
                self.pvalue_indicator_grid.append(
                    np.sum(ps <= self.pthresh) > np.sum(ps > self.pthresh)
                )
                self.pvalue_tail_indicator_grid.append(
                    np.sum(ps_tail <= self.pthresh) > np.sum(ps_tail > self.pthresh)
                )
            self.pvalue_indicator_grid = np.array(self.pvalue_indicator_grid)
            self.pvalue_tail_indicator_grid = np.array(self.pvalue_tail_indicator_grid)
        # Select the best model
        indicator = np.logical_or(
            self.pvalue_indicator_grid, self.pvalue_tail_indicator_grid
        )
        # indicator = np.logical_or(self.pvalue_grid <= self.pthresh, self.pvalue_tail_grid <= self.pthresh)
        if indicator.any():
            index_selected = np.argmax(indicator)
        else:  # All False
            warnings.warn(
                "No p-value is smaller than or equal to %.3g. The largest lambda_tilde is selected. Enlarge the search grid of lambda_tilde."
                % self.pthresh,
                UserWarning,
            )
            index_selected = len(self.pvalue_grid)
        # Output a warning when the selected lambda_tilde is the left edge of the grid.
        if index_selected == 0:
            warnings.warn(
                "The smallest lambda_tilde is selected. The optimal lambda_tilde might be smaller. We suggest to extend the grid to smaller lambda_tilde values to validate.",
                UserWarning,
            )
        self.lambda_tilde = self.lambda_tilde_grid[index_selected]
        self.model = models[index_selected]
        self.W = self.model.W
        self.H = self.model.H
        self.lam = self.model.lam
        self.loss = self.model.loss
        self.reconstruction_error = self.model.reconstruction_error
        return self
