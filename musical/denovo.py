"""Main class for de-novo extraction of signatures"""

import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples
import scipy.stats as stats
import warnings
import time
import multiprocessing
import os

from .nmf import NMF
from .mvnmf import MVNMF, wrappedMVNMF
from .utils import bootstrap_count_matrix, beta_divergence, _samplewise_error
from .nnls import nnls


def _gather_results(X, Ws, method='hierarchical'):
    """Gather NMF or mvNMF results

    TODO
    ----------
    1. Replicate the clustering method in SigProfilerExtractor.
    """
    n_features, n_samples = X.shape
    n_components = Ws[0].shape[1]
    ### If only one solution:
    if len(Ws) == 1:
        W = Ws[0]
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # Here we define the sil_score to be 1 when there is only one sample in each cluster.
        # This is different from the canonical definition, where it is 0.
        sil_score = np.ones(n_components)
        sil_score_mean = 1.0
        return W, H, sil_score, sil_score_mean
    ### If more than one solutions:
    ### If there is only 1 signature:
    if n_components == 1:
        W = np.mean(Ws, 0)
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # When there is only 1 cluster, we also define the sil_score to be 1.
        sil_score = np.ones(n_components)
        sil_score_mean = 1.0
        return W, H, sil_score, sil_score_mean
    ### If there are more than 1 signatures
    if method == 'hierarchical':
        sigs = np.concatenate(Ws, axis=1)
        sigs = normalize(sigs, norm='l1', axis=0)
        d = sp.spatial.distance.pdist(sigs.T, metric='cosine')
        d = d.clip(0)
        d_square_form = sp.spatial.distance.squareform(d)
        linkage = sch.linkage(d, method='average')
        cluster_membership = sch.fcluster(linkage, n_components, criterion="maxclust")
        W = []
        for i in range(0, n_components):
            W.append(np.mean(sigs[:, cluster_membership == i + 1], 1))
        W = np.array(W).T
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # Calculate sil_score
        samplewise_sil_score = silhouette_samples(d_square_form, cluster_membership, metric='precomputed')
        sil_score = []
        for i in range(0, n_components):
            sil_score.append(np.mean(samplewise_sil_score[cluster_membership == i + 1]))
        sil_score = np.array(sil_score)
        sil_score_mean = np.mean(samplewise_sil_score)
        return W, H, sil_score, sil_score_mean
    else:
        raise ValueError('Only method = hierarchical is implemented for _gather_results().')


class DenovoSig:
    """De novo signature extraction

    TODO
    ----------
    1. I'm not sure if min_n_components = 1 will be taken care of gracefully. Need to check.
    2. About selecting n_components, I'm taking the smallest n_components with a non-sigificant p-value.
        For example, if the pvalue series is [0.001, 0.001, 0.02, 0.3, 0.3, 0.01, 0.001, 0.5, 0.6] and the
        corresponding n_components are [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], then I will select n_components = 5.
        But one can argue that you should select n_components = 9. We should test on this.
    """
    def __init__(self,
                 X,
                 pthresh=0.05, # for selecting n_components
                 min_n_components=None,
                 max_n_components=None,
                 init='random',
                 method='nmf',
                 bootstrap=True,
                 n_replicates=100,
                 max_iter=1000000,
                 min_iter=10000,
                 conv_test_freq=100,
                 conv_test_baseline='min-iter',
                 tol=1e-10,
                 ncpu=1,
                 verbose=0,
                 # mvnmf specific:
                 mvnmf_hyperparameter_method='single', # single or all or fixed
                 mvnmf_lambda_tilde_grid=None,
                 mvnmf_delta=1.0,
                 mvnmf_gamma=1.0,
                 mvnmf_pthresh=0.05
                ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.pthresh = pthresh
        self.n_features, self.n_samples = self.X.shape
        if min_n_components is None:
            min_n_components = 2
        self.min_n_components = min_n_components
        if max_n_components is None:
            max_n_components = min(20, self.n_samples)
        self.max_n_components = max_n_components
        self.n_components_all = np.arange(self.min_n_components, self.max_n_components + 1)
        self.init = init
        self.method = method
        self.bootstrap = bootstrap
        self.n_replicates = n_replicates
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.tol = tol
        self.verbose=verbose
        if ncpu is None:
            ncpu = os.cpu_count()
        self.ncpu=ncpu
        # mvnmf specific
        self.mvnmf_hyperparameter_method = mvnmf_hyperparameter_method
        self.mvnmf_lambda_tilde_grid = mvnmf_lambda_tilde_grid
        self.mvnmf_delta = mvnmf_delta
        self.mvnmf_gamma = mvnmf_gamma
        self.mvnmf_pthresh = mvnmf_pthresh

    def _job(self, parameters):
        """parameters = (index_replicate, n_components, eng, lambda_tilde)

        Note that this function must be defined outside of self.fit(), otherwise we'll receive
        'cannot pickle' errors.
        """
        index_replicate, n_components, eng, lambda_tilde = parameters
        if self.method == 'nmf':
            if self.bootstrap:
                X_in = bootstrap_count_matrix(self.X)
            else:
                X_in = self.X
            model = NMF(X_in,
                        n_components,
                        init=self.init,
                        max_iter=self.max_iter,
                        min_iter=self.min_iter,
                        tol=self.tol,
                        conv_test_freq=self.conv_test_freq,
                        conv_test_baseline=self.conv_test_baseline
                       )
            model.fit(eng=eng)
            if self.verbose:
                print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
            return model
        elif self.method == 'mvnmf':
            if self.mvnmf_hyperparameter_method == 'all':
                if self.bootstrap:
                    X_in = bootstrap_count_matrix(self.X)
                else:
                    X_in = self.X
                model = wrappedMVNMF(X_in,
                                     n_components,
                                     init=self.init,
                                     max_iter=self.max_iter,
                                     min_iter=self.min_iter,
                                     tol=self.tol,
                                     conv_test_freq=self.conv_test_freq,
                                     conv_test_baseline=self.conv_test_baseline,
                                     lambda_tilde_grid=self.mvnmf_lambda_tilde_grid,
                                     pthresh=self.mvnmf_pthresh,
                                     delta=self.mvnmf_delta,
                                     gamma=self.mvnmf_gamma
                                    )
                model.fit(eng=eng)
                if self.verbose:
                    print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
                    print('Selected lambda_tilde = %.3g ' % model.lambda_tilde)
                return model
            elif self.mvnmf_hyperparameter_method == 'fixed':
                if self.bootstrap:
                    X_in = bootstrap_count_matrix(self.X)
                else:
                    X_in = self.X
                model = MVNMF(X_in,
                              n_components,
                              init=self.init,
                              max_iter=self.max_iter,
                              min_iter=self.min_iter,
                              tol=self.tol,
                              conv_test_freq=self.conv_test_freq,
                              conv_test_baseline=self.conv_test_baseline,
                              lambda_tilde=self.mvnmf_lambda_tilde_grid,
                              delta=self.mvnmf_delta,
                              gamma=self.mvnmf_gamma
                             )
                model.fit(eng=eng)
                if self.verbose:
                    print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
                return model
            elif self.mvnmf_hyperparameter_method == 'single':
                if self.bootstrap:
                    X_in = bootstrap_count_matrix(self.X)
                else:
                    X_in = self.X
                model = MVNMF(X_in,
                              n_components,
                              init=self.init,
                              max_iter=self.max_iter,
                              min_iter=self.min_iter,
                              tol=self.tol,
                              conv_test_freq=self.conv_test_freq,
                              conv_test_baseline=self.conv_test_baseline,
                              lambda_tilde=lambda_tilde,
                              delta=self.mvnmf_delta,
                              gamma=self.mvnmf_gamma
                             )
                model.fit(eng=eng)
                if self.verbose:
                    print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
                return model

    def fit(self, eng=None):
        # 1. Run NMFs
        # 2. Gather results within the same n_components
        # 3. Select n_components
        self.W_all = {}
        self.H_all = {}
        self.sil_score_all = {}
        self.sil_score_mean_all = {}
        self.reconstruction_error_all = {}
        start = time.time()
        for n_components in self.n_components_all:
            ##################################################
            ########### First run NMFs or mvNMFs #############
            ##################################################
            if self.verbose:
                print('Extracting signatures for n_components = ' + str(n_components) + '..................')
            if self.method == 'nmf':
                parameters = [(index_replicate, n_components, eng, None) for index_replicate in range(0, self.n_replicates)]
                # Note that after workers are created, modifications of global variables won't be seen by the workers.
                # Therefore, any modifications must be made before the workers are created.
                # This is why we need to recreate workers for each n_components.
                workers = multiprocessing.Pool(self.ncpu)
                models = workers.map(self._job, parameters)
                workers.close()
                workers.join()
            elif self.method == 'mvnmf':
                if self.mvnmf_hyperparameter_method == 'all':
                    parameters = [(index_replicate, n_components, eng, None) for index_replicate in range(0, self.n_replicates)]
                    workers = multiprocessing.Pool(self.ncpu)
                    models = workers.map(self._job, parameters)
                    workers.close()
                    workers.join()
                elif self.mvnmf_hyperparameter_method == 'single':
                    # Run first model, with hyperparameter selection
                    if self.bootstrap:
                        X_in = bootstrap_count_matrix(self.X)
                    else:
                        X_in = self.X
                    model = wrappedMVNMF(X_in,
                                         n_components,
                                         init=self.init,
                                         max_iter=self.max_iter,
                                         min_iter=self.min_iter,
                                         tol=self.tol,
                                         conv_test_freq=self.conv_test_freq,
                                         conv_test_baseline=self.conv_test_baseline,
                                         lambda_tilde_grid=self.mvnmf_lambda_tilde_grid,
                                         pthresh=self.mvnmf_pthresh,
                                         delta=self.mvnmf_delta,
                                         gamma=self.mvnmf_gamma
                                        )
                    model.fit(eng=eng)
                    models = [model]
                    lambda_tilde = model.lambda_tilde
                    if self.verbose:
                        print('Selected lambda_tilde = %.3g. This lambda_tilde will be used for all subsequent mvNMF runs.' % model.lambda_tilde)
                    # Run the rest of the models, using preselected hyperparameter
                    parameters = [(index_replicate, n_components, eng, lambda_tilde) for index_replicate in range(1, self.n_replicates)]
                    workers = multiprocessing.Pool(self.ncpu)
                    _models = workers.map(self._job, parameters)
                    workers.close()
                    workers.join()
                    models.extend(_models)
                elif self.mvnmf_hyperparameter_method == 'fixed':
                    if type(self.mvnmf_lambda_tilde_grid) is not float:
                        raise ValueError('When mvnmf_hyperparameter_method is set to fixed, a single float value must be provided for mvnmf_lambda_tilde_grid.')
                    parameters = [(index_replicate, n_components, eng, None) for index_replicate in range(0, self.n_replicates)]
                    workers = multiprocessing.Pool(self.ncpu)
                    models = workers.map(self._job, parameters)
                    workers.close()
                    workers.join()
            ##############################################
            ####### Gather results from all models #######
            ##############################################
            W, H, sil_score, sil_score_mean = _gather_results(self.X, [model.W for model in models])
            self.W_all[n_components] = W
            self.H_all[n_components] = H
            self.sil_score_all[n_components] = sil_score
            self.sil_score_mean_all[n_components] = sil_score_mean
            self.reconstruction_error_all[n_components] = beta_divergence(self.X, W @ H, beta=1, square_root=False)
            if self.verbose:
                print('Time elapsed: %.3g seconds.' % (time.time() - start))
        ##############################################
        ############ Select n_components #############
        ##############################################
        self.samplewise_reconstruction_errors_all = {
            n_components: _samplewise_error(self.X, self.W_all[n_components] @ self.H_all[n_components]) for n_components in self.n_components_all
        }
        # If there is only 1 n_components tested, return its result
        if len(self.n_components_all) == 1:
            warnings.warn('Only 1 n_components value is tested.',
                          UserWarning)
            self.n_components = self.n_components_all[0]
            if np.mean(self.sil_score_all[self.n_components]) >= 0.8 and np.min(self.sil_score_all[self.n_components]) >= 0.2:
                self.min_n_components_stable = self.n_components
                self.max_n_components_stable = self.n_components
            else:
                self.min_n_components_stable = None
                self.max_n_components_stable = None
            self.pvalue_all = None
            self.pvalue_all_n_components = None
        # If there are more than 1 n_components tested:
        else:
            # First get stable n_components
            candidates = []
            for n_components in self.n_components_all:
                if np.mean(self.sil_score_all[n_components]) >= 0.8 and np.min(self.sil_score_all[n_components]) >= 0.2:
                    candidates.append(n_components)
            candidates = np.array(candidates)
            # If there is only 1 stable n_components, we take it:
            if len(candidates) == 1:
                warnings.warn('Only 1 n_components value with stable solutions is found.',
                              UserWarning)
                self.n_components = candidates[0]
                self.min_n_components_stable = self.n_components
                self.max_n_components_stable = self.n_components
                self.pvalue_all = None
                self.pvalue_all_n_components = None
            else:
                # If there are no stable n_components, we do p-value tests for all n_components
                if len(candidates) == 0:
                    self.min_n_components_stable = None
                    self.max_n_components_stable = None
                    warnings.warn('No n_components values with stable solutions are found.',
                                  UserWarning)
                    self.pvalue_all_n_components = self.n_components_all
                else:
                    self.min_n_components_stable = np.min(candidates)
                    self.max_n_components_stable = np.max(candidates)
                    self.pvalue_all_n_components = np.arange(self.min_n_components_stable, self.max_n_components_stable + 1)
                self.pvalue_all = np.array([
                    stats.mannwhitneyu(self.samplewise_reconstruction_errors_all[n_components],
                                       self.samplewise_reconstruction_errors_all[n_components + 1],
                                       alternative='greater')[1] for n_components in self.pvalue_all_n_components[0:-1]
                ])
                if np.max(self.pvalue_all) <= self.pthresh:
                    index_selected = len(self.pvalue_all_n_components) - 1
                else:
                    index_selected = np.argmax(self.pvalue_all > self.pthresh)
                self.n_components = self.pvalue_all_n_components[index_selected]
        self.W = self.W_all[self.n_components]
        self.H = self.H_all[self.n_components]
        self.sil_score = self.sil_score_all[self.n_components]
        self.sil_score_mean = self.sil_score_mean_all[self.n_components]
        self.reconstruction_error = self.reconstruction_error_all[self.n_components]
        self.samplewise_reconstruction_errors = self.samplewise_reconstruction_errors_all[self.n_components]

        return self
