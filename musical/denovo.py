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
from .refit import reassign
from .validate import validate

def _gather_results(X, Ws, Hs=None, method='hierarchical', filter=False, thresh=10):
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
    # Filtering
    if filter:
        if Hs is None:
            raise ValueError('If filtering is to be performed, Hs must be supplied.')
        errors = np.array([beta_divergence(X, W @ H) for W, H in zip(Ws, Hs)])
        retained_indices = np.arange(0, len(Ws))[(errors - np.median(errors)) < thresh*stats.median_abs_deviation(errors)]
        Ws = [Ws[i] for i in retained_indices]
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
                 mvnmf_pthresh=0.05,
                 use_catalog=True,
                 catalog_name='COSMIC_v3p1_SBS_WGS',
                 thresh_match = [0.99],
                 thresh_new_sig = [0.84],
                 min_contribution = [0.1],
                 include_top = [False],
                 method_sparse = 'llh',
                 frac_thresh_base = [0.02],
                 frac_thresh_keep = [0.4],
                 frac_thresh = [0.05],
                 llh_thresh = [0.65],
                 exp_thresh = [8.],
                 features = None
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
            max_n_components = 20
        max_n_components = min(max_n_components, self.n_samples)
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
        self.use_catalog = use_catalog
        self.catalog_name = catalog_name
        self.thresh_match = thresh_match
        self.thresh_new_sig = thresh_new_sig
        self.min_contribution = min_contribution
        self.include_top = include_top
        self.method_sparse = method_sparse
        self.frac_thresh_base = frac_thresh_base
        self.frac_thresh_keep = frac_thresh_keep
        self.frac_thresh = frac_thresh
        self.llh_thresh = llh_thresh
        self.exp_thresh = exp_thresh
        self.features = features

    def _job(self, parameters):
        """parameters = (index_replicate, n_components, eng, lambda_tilde)

        Note that this function must be defined outside of self.fit(), otherwise we'll receive
        'cannot pickle' errors.
        """
        index_replicate, n_components, eng, lambda_tilde = parameters
        np.random.seed() # This is critical: https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing
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
                                     gamma=self.mvnmf_gamma,
                                     ncpu=1
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
        self.W_raw_all = {} # Save all raw results
        self.H_raw_all = {} # Save all raw results
        self.lambda_tilde_all = {} # Save lambda_tilde's used for each mvNMF run
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
                                         gamma=self.mvnmf_gamma,
                                         ncpu=self.ncpu
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
            #self.models = models
            self.W_raw_all[n_components] = [model.W for model in models] # Save all raw results
            self.H_raw_all[n_components] = [model.H for model in models] # Save all raw results
            # Save lambda_tilde's used for each mvNMF run
            if self.method == 'nmf':
                self.lambda_tilde_all[n_components] = None
            elif self.method == 'mvnmf':
                if self.mvnmf_hyperparameter_method == 'all':
                    self.lambda_tilde_all[n_components] = [model.lambda_tilde for model in models]
                elif self.mvnmf_hyperparameter_method == 'single':
                    self.lambda_tilde_all[n_components] = [lambda_tilde]*self.n_replicates
                elif self.mvnmf_hyperparameter_method == 'fixed':
                    self.lambda_tilde_all[n_components] = [self.mvnmf_lambda_tilde_grid]*self.n_replicates
            W, H, sil_score, sil_score_mean = _gather_results(
                self.X, [model.W for model in models], Hs=[model.H for model in models], filter=True
            )
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

    def set_params(self,
                   use_catalog = None,
                   catalog_name = None,
                   thresh_match = None,
                   thresh_new_sig = None,
                   min_contribution = None,
                   include_top = None,
                   method_sparse = None,
                   frac_thresh_base = None,
                   frac_thresh_keep = None,
                   frac_thresh = None,
                   llh_thresh = None,
                   exp_thresh = None):

        if use_catalog != None:
           self.use_catalog = use_catalog
        if catalog_name != None:
           self.catalog_name = catalog_name
        if thresh_match != None:
           self.thresh_match = thresh_match
        if thresh_new_sig != None:
           self.thresh_new_sig = thresh_new_sig
        if min_contribution != None:
           self.min_contribution = min_contribution
        if include_top != None:
            self.include_top = include_top
        if method_sparse != None:
            self.method_sparse = method_sparse
        if frac_thresh_base != None:
            self.frac_thresh_base = frac_thresh_base
        if frac_thresh_keep != None:
            self.frac_thresh_keep = frac_thresh_keep
        if frac_thresh != None:
            self.frac_thresh = frac_thresh
        if llh_thresh != None:
            self.llh_thresh = llh_thresh
        if exp_thresh != None:
            self.exp_thresh = exp_thresh

        return self

    def clone_model(self, X_new, grid_index = 1):
        if grid_index >= self.n_grid:
           ValueError('In clone_model grid_index is out of bounds of n_grid')

        model_new = DenovoSig(X_new,
                              min_n_components = self.n_components,
                              max_n_components = self.n_components,
                              init = self.init,
                              method = self.method,
                              bootstrap = self.bootstrap,
                              n_replicates = self.n_replicates,
                              max_iter = self.max_iter,
                              min_iter = self.min_iter,
                              conv_test_freq = self.conv_test_freq,
                              conv_test_baseline = self.conv_test_baseline,
                              tol = self.tol,
                              ncpu = self.ncpu,
                              verbose = self.verbose,
                              # mvnmf specific:
                              mvnmf_hyperparameter_method = self.mvnmf_hyperparameter_method,
                              mvnmf_lambda_tilde_grid = self.mvnmf_lambda_tilde_grid,
                              mvnmf_delta = self.mvnmf_delta,
                              mvnmf_gamma = self.mvnmf_gamma,
                              mvnmf_pthresh = self.mvnmf_pthresh,
                              use_catalog = self.use_catalog,
                              catalog_name = self.catalog_name,
                              method_sparse = self.method_sparse,
                              features = self.features)
        if self.n_grid > 1:
            model_new.set_params(frac_thresh_base = [self.frac_thresh_base_all[grid_index]],
                                  frac_thresh_keep = [self.frac_thresh_keep_all[grid_index]],
                                  frac_thresh = [self.frac_thresh_all[grid_index]],
                                  llh_thresh = [self.llh_thresh_all[grid_index]],
                                  exp_thresh = [self.exp_thresh_all[grid_index]])
        else:
            model_new.set_params(frac_thresh_base = [self.frac_thresh_base],
                                  frac_thresh_keep = [self.frac_thresh_keep],
                                  frac_thresh = [self.frac_thresh],
                                  llh_thresh = [self.llh_thresh],
                                  exp_thresh = [self.exp_thresh])

        if self.use_catalog:
            if self.n_grid > 1:
                model_new.set_params(thresh_match = [self.thresh_match_all],
                                     thresh_new_sig = [self.thresh_new_sig_all],
                                     min_contribution = [self.min_contribution_all],
                                     include_top = [self.include_top_all])
            else:
                model_new.set_params(thresh_match = [self.thresh_match],
                                     thresh_new_sig = [self.thresh_new_sig],
                                     min_contribution = [self.min_contribution],
                                     include_top = [self.include_top])
        return model_new


    def clear_grid(self):
        if hasattr(self, 'frac_thresh_base_all'):
            self.frac_thresh_base_all = None
        if hasattr(self, 'frac_thresh_keep_all'):
            self.frac_thresh_keep_all = None
        if hasattr(self, 'llh_thresh_all'):
            self.llh_thresh_all = None
        if hasattr(self, 'exp_thresh_all'):
            self.exp_thresh_all = None
        if hasattr(self, 'reconstruction_error_s_all'):
            self.reconstruction_error_s_all = None
        if hasattr(self, 'thresh_match_all'):
            self.thresh_match_all= None
        if hasattr(self, 'thresh_new_sig_all'):
            self.thresh_new_sig_all = None
        if hasattr(self, 'min_contribution_all'):
            self.min_contribution_all = None
        if hasattr(self, 'include_top_all'):
            self.include_top_all = None
        return self

    def run_reassign(self, validation_output_file = None, use_refit = False, clear = True):
        W_s, H_s, signames, reconstruction_error_s_all, n_grid, frac_thresh_base_all, frac_thresh_keep_all, frac_thresh_all, llh_thresh_all, exp_thresh_all, thresh_match_all, thresh_new_sig_all, min_contribution_all, include_top_all = reassign(self)


        # User might want to keep the same model and do a second reassignment
        # Because if the size of parameters is 1 attributes that do not end
        # all are saved this can result in having results from two different
        # reassignment calls


        if clear:
            if hasattr(self, 'W_s'):
                self.W_s = None
            if hasattr(self, 'H_s'):
                self.H_s = None
            if hasattr(self, 'signature_names'):
                self.signature_names = None
            if hasattr(self, 'reconstruction_error_s'):
                self.reconstruction_error_s = None
            if hasattr(self, 'W_s_all'):
                self.W_s_all = None
            if hasattr(self, 'H_s_all'):
                self.H_s_all = None
            if hasattr(self, 'signature_names_all'):
                self.signature_names_all = None
            self.clear_grid()

        self.n_grid = n_grid

        if n_grid == 1:
            self.W_s = W_s[0]
            self.H_s = H_s[0]
            self.signature_names = signames[0]
            self.reconstruction_error_s = reconstruction_error_s_all[0]
        else:
            self.W_s_all = W_s
            self.H_s_all = H_s
            self.signature_names_all = signames
            self.frac_thresh_base_all = frac_thresh_base_all
            self.frac_thresh_all = frac_thresh_all
            self.frac_thresh_keep_all = frac_thresh_keep_all
            self.llh_thresh_all = llh_thresh_all
            self.exp_thresh_all = exp_thresh_all
            self.reconstruction_error_s_all = reconstruction_error_s_all
            if len(thresh_match_all) > 0:
                 self.thresh_match_all = thresh_match_all
            if len(thresh_new_sig_all):
                self.thresh_new_sig_all = thresh_new_sig_all
            if len(min_contribution_all):
                self.min_contribution_all = min_contribution_all
            if len(include_top_all) > 0:
                self.include_top_all = include_top_all
        return self

    def validate_assignment(self, validation_output_file = None, use_refit = False, clear_grid = False):
        W_simul, H_simul, X_simul, best_grid_index, error_W, error_H, dist_W, dist_max, dist_max_sig_index, dist_max_all, dist_max_sig_index_all, _, _, _, _, _, _ = validate(self, validation_output_file = validation_output_file, use_refit = use_refit)
        self.W_simul = W_simul
        self.H_simul = H_simul
        self.X_simul = X_simul
        self.best_grid_index = best_grid_index
        self.error_W_simul = error_W
        self.error_H_simul = error_H
        self.dist_W_simul = dist_W
        self.dist_max_simul = dist_max
        self.dist_max_simul_sig_index = dist_max_sig_index
        self.dist_max_simul_all = dist_max_all
        self.dist_max_simul_sig_index_all = dist_max_sig_index_all

        if self.n_grid > 1:
            self.W_s = self.W_s_all[best_grid_index]
            self.H_s = self.H_s_all[best_grid_index]
            self.reconstruction_error_s = self.reconstruction_error_s_all[best_grid_index]
            self.set_params(frac_thresh_base = [self.frac_thresh_base_all[best_grid_index]],
                            frac_thresh_keep = [self.frac_thresh_keep_all[best_grid_index]],
                            frac_thresh = [self.frac_thresh_all[best_grid_index]],
                            llh_thresh = [self.llh_thresh_all[best_grid_index]],
                            exp_thresh = [self.exp_thresh_all[best_grid_index]])
            if self.use_catalog:
                self.set_params(thresh_match = [self.thresh_match_all[best_grid_index]],
                                thresh_new_sig = [self.thresh_new_sig_all[best_grid_index]],
                                min_contribution = [self.min_contribution_all[best_grid_index]],
                                include_top = [self.include_top_all[best_grid_index]])

        if clear_grid:
            self.clear_grid()
        return self
