"""Main class for de-novo extraction of signatures

TODO:
1. Universally work with pd dataframes in DenovoSig.
2. We need better structuring of the class. E.g., use @ property to protect some attributes.
"""

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
import pandas as pd
from operator import itemgetter
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import copy

from .nmf import NMF
from .mvnmf import MVNMF, wrappedMVNMF
from .utils import bootstrap_count_matrix, beta_divergence, _samplewise_error, match_catalog_pair, differential_tail_test, simulate_count_matrix
from .nnls import nnls
from .refit import reassign, assign, assign_grid
from .validate import validate
from .cluster import OptimalK, hierarchical_cluster


def _filter_results(X, Ws, Hs, method='error_distribution', thresh=0.05, percentile=90):
    """Filter NMF or mvNMF results

    TODO
    ----------
    1. Currently for filtering based on tail distributions, we use a fixed percentile as a threshold for the definition of tails.
        Alternatively, we can try to fit a gaussian mixture model to the samplewise error distribution (combined), and decide
        whether there is a cluster of samples that are not reconstructed well. If there is, then we ues the threshold defined by
        the mixture model to define tails. If there isn't, then we simply skip tail based filtering.
    """
    if len(Ws) == 1:
        return Ws, Hs, np.array([0])
    else:
        errors = np.array([beta_divergence(X, W @ H) for W, H in zip(Ws, Hs)])
        if method == 'error_distribution':
            samplewise_errors = [_samplewise_error(X, W @ H) for W, H in zip(Ws, Hs)]
            best_index = np.argmin(errors)
            pvalues = np.array([
                stats.mannwhitneyu(samplewise_errors[i],
                                   samplewise_errors[best_index],
                                   alternative='greater')[1] for i in range(0, len(Ws))
            ])
            pvalues_tail = np.array([
                differential_tail_test(samplewise_errors[i],
                                       samplewise_errors[best_index],
                                       percentile=percentile,
                                       alternative='greater')[1] for i in range(0, len(Ws))
            ])
            retained_indices = np.arange(0, len(Ws))[np.logical_and(pvalues > thresh, pvalues_tail > thresh)]
            Ws_filtered = [Ws[i] for i in retained_indices]
            Hs_filtered = [Hs[i] for i in retained_indices]
            return Ws_filtered, Hs_filtered, retained_indices
        elif method == 'error_MAE':
            retained_indices = np.arange(0, len(Ws))[(errors - np.median(errors)) <= thresh*stats.median_abs_deviation(errors)]
            Ws_filtered = [Ws[i] for i in retained_indices]
            Hs_filtered = [Hs[i] for i in retained_indices]
            return Ws_filtered, Hs_filtered, retained_indices
        elif method == 'error_min':
            retained_indices = np.arange(0, len(Ws))[errors <= (thresh + 1.0)*np.min(errors)]
            Ws_filtered = [Ws[i] for i in retained_indices]
            Hs_filtered = [Hs[i] for i in retained_indices]
            return Ws_filtered, Hs_filtered, retained_indices
        else:
            raise ValueError('Invalid method for _filter_results().')


def _gather_results(X, Ws, method='cluster_by_matching', n_components=None):
    """Gather NMF or mvNMF results

    TODO
    ----------
    1. Replicate the clustering method in SigProfilerExtractor.
    """
    n_features, n_samples = X.shape
    if n_components is None:
        n_components = Ws[0].shape[1]
    ### If only one solution:
    if len(Ws) == 1:
        W = Ws[0]
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # Here we define the sil_score to be 1 when there is only one sample in each cluster.
        # This is different from the canonical definition, where it is 0.
        # 20210415 - We change this to 0.0. When there is only 1 sample in the cluster, it means that solution is rare and thus not stable.
        # 20210719 - We'll still use 1. We'll look at n_replicates_after_filtering in _select_n_components in addition.
        sil_score = np.ones(n_components)
        sil_score_mean = 1.0
        n_support = np.ones(n_components, dtype=int)
        return W, H, sil_score, sil_score_mean, n_support
    ### If more than one solutions:
    ### If there is only 1 signature:
    if n_components == 1:
        W = np.mean(Ws, 0)
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # When there is only 1 cluster, we also define the sil_score to be 1.
        sil_score = np.ones(n_components)
        sil_score_mean = 1.0
        n_support = np.ones(n_components, dtype=int)*len(Ws)
        return W, H, sil_score, sil_score_mean, n_support
    ### If there are more than 1 signatures
    if method == 'hierarchical':
        sigs = np.concatenate(Ws, axis=1)
        sigs = normalize(sigs, norm='l1', axis=0)
        d = sp.spatial.distance.pdist(sigs.T, metric='cosine')
        d = d.clip(0)
        d_square_form = sp.spatial.distance.squareform(d)
        linkage = sch.linkage(d, method='average')
        cluster_membership = sch.fcluster(linkage, n_components, criterion="maxclust")
        if len(set(cluster_membership)) != n_components:
            cluster_membership = sch.cut_tree(linkage, n_clusters=n_components).flatten() + 1
            if len(set(cluster_membership)) != n_components:
                warnings.warn('Number of clusters output by cut_tree or fcluster is not equal to the specified number of clusters',
                              UserWarning)
        W = []
        n_support = []
        for i in range(0, n_components):
            W.append(np.mean(sigs[:, cluster_membership == i + 1], 1))
            n_support.append(np.sum(cluster_membership == (i + 1)))
        n_support = np.array(n_support)
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
        return W, H, sil_score, sil_score_mean, n_support
    elif method == 'matching':
        Ws_matched = [Ws[0]]
        for W in Ws[1:]:
            W_matched, _, _ = match_catalog_pair(Ws[0], W)
            Ws_matched.append(W_matched)
        Ws_matched = np.array(Ws_matched)
        W = np.mean(Ws_matched, axis=0)
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # Distance matrix and cluster membership
        sigs = np.concatenate(Ws_matched, axis=1)
        sigs = normalize(sigs, norm='l1', axis=0)
        d = sp.spatial.distance.pdist(sigs.T, metric='cosine')
        d = d.clip(0)
        d_square_form = sp.spatial.distance.squareform(d)
        cluster_membership = np.tile(np.arange(1, n_components + 1), len(Ws))
        # Calculate sil_score
        samplewise_sil_score = silhouette_samples(d_square_form, cluster_membership, metric='precomputed')
        sil_score = []
        for i in range(0, n_components):
            sil_score.append(np.mean(samplewise_sil_score[cluster_membership == i + 1]))
        sil_score = np.array(sil_score)
        sil_score_mean = np.mean(samplewise_sil_score)
        n_support = np.ones(n_components, dtype=int)*len(Ws)
        return W, H, sil_score, sil_score_mean, n_support
    elif method == 'cluster_by_matching':
        # pubmed: 32118208
        ### First, get all matchings
        matchings = []
        for i in range(0, len(Ws)):
            for j in range(i, len(Ws)):
                if i != j:
                    W1 = Ws[i]
                    W2 = Ws[j]
                    _, col_ind, pdist = match_catalog_pair(W1, W2, metric='cosine')
                    row_ind = np.arange(0, n_components)
                    matchings.append(
                        [pdist[row_ind, col_ind].sum(),
                         pd.DataFrame(np.array([np.arange(0, n_components), col_ind]).T, columns=[i, j])]
                    )
        matchings = sorted(matchings, key=itemgetter(0))
        ### Initialize
        result = [matchings[0][1]]
        matchings = matchings[1:]
        ### Select matchings
        while len(matchings) > 0:
            # Check stopping criterion
            if len(result) == 1 and sorted(result[0].columns.values.tolist()) == list(range(0, len(Ws))):
                break
            # Processing
            match = matchings[0]
            df = match[1]
            col1 = df.columns[0]
            col2 = df.columns[1]
            col1_in = -1
            col2_in = -1
            # Check if already in the result
            for i, item in zip(range(len(result)), result):
                if col1 in item.columns.values:
                    col1_in = i
                if col2 in item.columns.values:
                    col2_in = i
            # Add or match or pass
            if col1_in == -1 and col2_in == -1:
                result.append(df)
            elif col1_in == -1:
                df_old = result[col2_in]
                df_old = df_old.sort_values(by=col2)
                df = df.sort_values(by=col2)
                df_old[col1] = df[col1].values
                result[col2_in] = df_old
            elif col2_in == -1:
                df_old = result[col1_in]
                df_old = df_old.sort_values(by=col1)
                df = df.sort_values(by=col1)
                df_old[col2] = df[col2].values
                result[col1_in] = df_old
            else:
                if col1_in == col2_in:
                    pass
                else:
                    df_old1 = result[col1_in]
                    df_old2 = result[col2_in]
                    df_old1 = df_old1.sort_values(by=col1)
                    df = df.sort_values(by=col1)
                    df_old1[col2] = df[col2].values
                    df_old1 = df_old1.sort_values(by=col2)
                    df_old2 = df_old2.sort_values(by=col2)
                    for col in df_old2.columns:
                        if col != col2:
                            df_old1[col] = df_old2[col].values
                    result = [result[i] for i in range(0, len(result)) if i != col1_in and i != col2_in]
                    result.append(df_old1)
            #
            matchings = matchings[1:]
        if len(result) != 1:
            raise RuntimeError('Clustering by matching resulted in a result of length != 1.')
        result = result[0]
        result = result[sorted(result.columns.values.tolist())]
        ### Gather results
        Ws_matched = []
        for W, i in zip(Ws, range(0, len(Ws))):
            Ws_matched.append(W[:, result[i].values.tolist()])
        Ws_matched = np.array(Ws_matched)
        W = np.mean(Ws_matched, axis=0)
        W = normalize(W, norm='l1', axis=0)
        H = nnls(X, W)
        # Distance matrix and cluster membership
        sigs = np.concatenate(Ws_matched, axis=1)
        sigs = normalize(sigs, norm='l1', axis=0)
        d = sp.spatial.distance.pdist(sigs.T, metric='cosine')
        d = d.clip(0)
        d_square_form = sp.spatial.distance.squareform(d)
        cluster_membership = np.tile(np.arange(1, n_components + 1), len(Ws))
        # Calculate sil_score
        samplewise_sil_score = silhouette_samples(d_square_form, cluster_membership, metric='precomputed')
        sil_score = []
        for i in range(0, n_components):
            sil_score.append(np.mean(samplewise_sil_score[cluster_membership == i + 1]))
        sil_score = np.array(sil_score)
        sil_score_mean = np.mean(samplewise_sil_score)
        n_support = np.ones(n_components, dtype=int)*len(Ws)
        return W, H, sil_score, sil_score_mean, n_support
    else:
        raise ValueError('Invalid method for _gather_results().')


def _select_n_components(n_components_all, samplewise_reconstruction_errors_all, sil_score_all,
                         n_replicates, n_replicates_after_filtering_all, Ws_all=None,
                         pthresh=0.05, sil_score_mean_thresh=0.8, sil_score_min_thresh=0.2,
                         n_replicates_filter_ratio_thresh=0.2,
                         method='algorithm1', nrefs=50, max_k_all=None):
    """Select the best n_components based on reconstruction error and stability.

    Parameters
    ----------
    n_components_all: array-like
        All n_components tested.

    samplewise_reconstruction_errors_all : dict
        Dictionary of samplewise reconstruction errors.

    sil_score_all : dict
        Dictionary of signature-wise silhouette scores.

    n_replicates : int
        Number of replicates run.

    n_replicates_after_filtering_all : dict
        Dictionary of number of replicates after filtering in _gather_results()

    pthresh : float
        Threshold for p-value.

    sil_score_mean_thresh : float
        Minimum required mean sil score for a solution to be considered stable.

    sil_score_min_thresh : float
        Minimum required min sil score for a solution to be considered stable.

    n_replicates_filter_ratio_thresh : float
        Minimum required ratio of n_replicates_after_filtering/n_replicates for a solution to be considered stable.

    method : str, 'algorithm1' | 'algorithm1.1' | 'algorithm2' | 'algorithm2.1'
        Cf: selecting_n_components.pdf

    TODO
    ----------
    1. Implement the method in SigProfilerExtractor.
    """
    if method == 'consistency' and Ws_all is None:
        raise ValueError('When method is consistency, Ws_all must be provided.')

    n_components_all = np.sort(n_components_all)

    ##### Stable solutions:
    n_components_stable = []
    for n_components in n_components_all:
        if (np.mean(sil_score_all[n_components]) >= sil_score_mean_thresh and
            np.min(sil_score_all[n_components]) >= sil_score_min_thresh and
            n_replicates_after_filtering_all[n_components]/n_replicates >= n_replicates_filter_ratio_thresh):
            n_components_stable.append(n_components)
    n_components_stable = np.array(n_components_stable)

    ##### Consistent solutions
    if method == 'consistency':
        ## Default max_k_all
        if max_k_all is None:
            max_k_all = {n_components:np.max(n_components_all) + 5 for n_components in n_components_all}
        ## First, get optimal k's
        optimal_k_all = {}
        for n_components in n_components_all:
            if n_components == 1:
                optimal_k_all[n_components] = 1
            else:
                Ws = np.concatenate(Ws_all[n_components], 1)
                optimalK = OptimalK(Ws, max_k=max_k_all[n_components], nrefs=nrefs)
                optimal_k_all[n_components] = optimalK.k
        ## Select candidates
        # Candidates are those n_components whose optimal k is equal to n_components
        n_components_consistent = []
        for n_components in n_components_all:
            if n_components == optimal_k_all[n_components]:
                n_components_consistent.append(n_components)
        n_components_consistent = np.array(n_components_consistent)
    else:
        optimal_k_all = None
        n_components_consistent = None

    ##### If only 1 n_components value provided.
    if len(n_components_all) == 1:
        warnings.warn('Only 1 n_components value is tested. Selecting this n_components value.',
                      UserWarning)
        return n_components_all[0], optimal_k_all, n_components_consistent, n_components_stable, np.array([]), np.array([])

    ##### Calculate p-values:
    pvalue_all = np.array([
        stats.mannwhitneyu(samplewise_reconstruction_errors_all[n_components],
                           samplewise_reconstruction_errors_all[n_components + 1],
                           alternative='greater')[1] for n_components in n_components_all[0:-1]
    ])

    pvalue_tail_all = np.array([
        differential_tail_test(samplewise_reconstruction_errors_all[n_components],
                               samplewise_reconstruction_errors_all[n_components + 1],
                               percentile=90,
                               alternative='greater')[1] for n_components in n_components_all[0:-1]
    ])

    ##### Select n_components
    if method == 'consistency':
        if len(n_components_consistent) == 0 and len(n_components_stable) == 0:
            # This won't usually happen
            warnings.warn('No consistent n_components values are found and '
                          'no n_components values with stable solutions are found. '
                          'Selecting the smallest n_components tested.',
                          UserWarning)
            n_components_selected = n_components_all[0]
        elif len(n_components_stable) == 0:
            warnings.warn('No n_components values with stable solutions are found. '
                          'Selecting the greatest consistent n_components value.',
                          UserWarning)
            n_components_selected = n_components_consistent[-1]
        elif len(n_components_consistent) == 0:
            # This won't usually happen
            warnings.warn('No consistent n_components values are found. '
                          'Selecting the greatest n_components with stable solutions.',
                          UserWarning)
            n_components_selected = n_components_stable[-1]
        else:
            n_components_intersect = np.array(list(set(n_components_consistent).intersection(n_components_stable)))
            if len(n_components_intersect) == 0:
                warnings.warn('Intersection of stable and consistent n_components is empty. '
                              'Selecting the greatest consistent n_components value.',
                              UserWarning)
                n_components_selected = n_components_consistent[-1]
            else:
                n_components_selected = np.max(n_components_intersect)
    elif method == 'algorithm1' or method == 'algorithm1.1':
        n_components_significant = []
        for p, p_tail, n_components in zip(pvalue_all, pvalue_tail_all, n_components_all[1:]):
            if p <= pthresh or p_tail <= pthresh:
                n_components_significant.append(n_components)
        n_components_significant = np.array(n_components_significant)
        if len(n_components_stable) == 0 and len(n_components_significant) == 0:
            warnings.warn('No n_components values with stable solutions are found and '
                          'no n_components values with significant p-values are found. '
                          'Selecting the smallest n_components tested.',
                          UserWarning)
            n_components_selected = n_components_all[0]
        elif len(n_components_significant) == 0:
            warnings.warn('No n_components values with significant p-values are found. '
                          'Selecting the smallest n_components with stable solutions.',
                          UserWarning)
            n_components_selected = n_components_stable[0]
        elif len(n_components_stable) == 0:
            warnings.warn('No n_components values with stable solutions are found. '
                          'Selecting the greatest n_components with significant p-values.',
                          UserWarning)
            n_components_selected = n_components_significant[-1]
        else:
            n_components_intersect = np.array(list(set(n_components_significant).intersection(n_components_stable)))
            if len(n_components_intersect) == 0:
                if method == 'algorithm1':
                    warnings.warn('Intersection of stable and significant n_components is empty. '
                                  'Selecting the greatest n_components with significant p-values.',
                                  UserWarning)
                    n_components_selected = n_components_significant[-1]
                elif method == 'algorithm1.1':
                    warnings.warn('Intersection of stable and significant n_components is empty. '
                                  'Selecting the smallest n_components with stable solutions.',
                                  UserWarning)
                    n_components_selected = n_components_stable[0]
            else:
                n_components_selected = np.max(n_components_intersect)
    elif method == 'algorithm2' or method == 'algorithm2.1':
        n_components_nonsignificant = []
        for p, p_tail, n_components in zip(pvalue_all, pvalue_tail_all, n_components_all[0:-1]):
            if p > pthresh and p_tail > pthresh:
                n_components_nonsignificant.append(n_components)
        n_components_nonsignificant = np.array(n_components_nonsignificant)
        if len(n_components_stable) == 0 and len(n_components_nonsignificant) == 0:
            warnings.warn('No n_components values with stable solutions are found and '
                          'no n_components values with nonsignificant p-values are found. '
                          'Selecting the greatest n_components tested.',
                          UserWarning)
            n_components_selected = n_components_all[-1]
        elif len(n_components_nonsignificant) == 0:
            warnings.warn('No n_components values with nonsignificant p-values are found. '
                          'Selecting the greatest n_components with stable solutions.',
                          UserWarning)
            n_components_selected = n_components_stable[-1]
        elif len(n_components_stable) == 0:
            warnings.warn('No n_components values with stable solutions are found. '
                          'Selecting the smallest n_components with nonsignificant p-values.',
                          UserWarning)
            n_components_selected = n_components_nonsignificant[0]
        else:
            n_components_intersect = np.array(list(set(n_components_nonsignificant).intersection(n_components_stable)))
            if len(n_components_intersect) == 0:
                if method == 'algorithm2':
                    warnings.warn('Intersection of stable and nonsignificant n_components is empty. '
                                  'Selecting the smallest n_components with nonsignificant p-values.',
                                  UserWarning)
                    n_components_selected = n_components_nonsignificant[0]
                elif method == 'algorithm2.1':
                    warnings.warn('Intersection of stable and nonsignificant n_components is empty. '
                                  'Selecting the greatest n_components with stable solutions.',
                                  UserWarning)
                    n_components_selected = n_components_stable[-1]
            else:
                n_components_selected = np.min(n_components_intersect)
    else:
        raise ValueError('Invalid method for _select_n_components.')

    if n_components_selected == n_components_all[-1]:
        warnings.warn('Selected n_components is equal to the maximum n_components tested. The optimal n_components might be greater.',
                      UserWarning)
    if n_components_selected == n_components_all[0] and n_components_selected > 1:
        warnings.warn('Selected n_components is equal to the minimum n_components tested. The optimal n_components might be smaller.',
                      UserWarning)

    return n_components_selected, optimal_k_all, n_components_consistent, n_components_stable, pvalue_all, pvalue_tail_all


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
                 min_n_components=None,
                 max_n_components=None,
                 init='random',
                 method='mvnmf',
                 normalize_X=False, # whether or not to normalize the input matrix for NMF/mvNMF
                 bootstrap=True,
                 n_replicates=20,
                 max_iter=100000,
                 min_iter=10000,
                 conv_test_freq=1000,
                 conv_test_baseline='min-iter',
                 tol=1e-8,
                 ncpu=1,
                 verbose=0,
                 # Specific for result filtering:
                 filter=True,
                 filter_method='error_MAE',
                 filter_thresh=5.0,
                 filter_percentile=90,
                 # Specific for result gathering:
                 cluster_method='hierarchical',
                 # Specific for n_components selection:
                 select_method='consistency',
                 select_pthresh=0.05, # Not used by the consistency method
                 select_sil_score_mean_thresh=0.7,
                 select_sil_score_min_thresh=0.2,
                 select_n_replicates_filter_ratio_thresh=0.2,
                 # mvnmf specific:
                 mvnmf_hyperparameter_method='single', # single or all or fixed
                 mvnmf_lambda_tilde_grid=None,
                 mvnmf_delta=1.0,
                 mvnmf_gamma=1.0,
                 mvnmf_pthresh=0.05,
                 mvnmf_noise=False,
                 # Parameters below will be removed. These parameters will be put into corresponding functions.
                 # Refitting:
                 use_catalog=True,
                 catalog_name='COSMIC_v3p1_SBS_WGS',
                 thresh1_match = [0.010],
                 thresh2_match = [None],
                 thresh_new_sig = [0.8],
                 method_sparse = 'likelihood_bidirectional',
                 thresh1 = [0.001],
                 thresh2 = [None]
                ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            self.X = np.array(X).astype(float)
        self.n_features, self.n_samples = self.X.shape
        if isinstance(X, pd.DataFrame):
            self.X_df = X.astype(float)
        else:
            self.X_df = pd.DataFrame(self.X,
                                     columns=['Sample' + str(i) for i in range(1, self.n_samples + 1)],
                                     index=['Feature' + str(i) for i in range(1, self.n_features + 1)])
        self.samples = self.X_df.columns.values
        self.features = self.X_df.index.values
        if min_n_components is None:
            min_n_components = 1
        self.min_n_components = min_n_components
        if max_n_components is None:
            max_n_components = 20
        max_n_components = min(max_n_components, self.n_samples)
        self.max_n_components = max_n_components
        self.n_components_all = np.arange(self.min_n_components, self.max_n_components + 1)
        self.init = init
        self.method = method
        self.normalize_X = normalize_X
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
        # Specific for result filtering:
        self.filter=filter
        self.filter_method=filter_method
        self.filter_thresh=filter_thresh
        self.filter_percentile=filter_percentile
        # Specific for result gathering:
        self.cluster_method=cluster_method
        # Specific for n_components selection:
        self.select_method=select_method
        self.select_pthresh=select_pthresh
        self.select_sil_score_mean_thresh=select_sil_score_mean_thresh
        self.select_sil_score_min_thresh=select_sil_score_min_thresh
        self.select_n_replicates_filter_ratio_thresh=select_n_replicates_filter_ratio_thresh
        # mvnmf specific
        self.mvnmf_hyperparameter_method = mvnmf_hyperparameter_method
        self.mvnmf_lambda_tilde_grid = mvnmf_lambda_tilde_grid
        self.mvnmf_delta = mvnmf_delta
        self.mvnmf_gamma = mvnmf_gamma
        self.mvnmf_pthresh = mvnmf_pthresh
        self.mvnmf_noise = mvnmf_noise
        # Refitting
        self.use_catalog = use_catalog
        self.catalog_name = catalog_name
        self.thresh1_match = thresh1_match
        self.thresh2_match = thresh2_match
        self.thresh_new_sig = thresh_new_sig
        self.method_sparse = method_sparse
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def _job(self, parameters):
        """parameters = (index_replicate, n_components, lambda_tilde)

        Note that this function must be defined outside of self.fit(), otherwise we'll receive
        'cannot pickle' errors.
        """
        index_replicate, n_components, lambda_tilde = parameters
        np.random.seed() # This is critical: https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing
        if self.method == 'nmf':
            if self.bootstrap:
                X_in = bootstrap_count_matrix(self.X)
            else:
                X_in = self.X
            if self.normalize_X:
                X_in = normalize(X_in, norm='l1', axis=0)
            model = NMF(X_in,
                        n_components,
                        init=self.init,
                        max_iter=self.max_iter,
                        min_iter=self.min_iter,
                        tol=self.tol,
                        conv_test_freq=self.conv_test_freq,
                        conv_test_baseline=self.conv_test_baseline
                       )
            model.fit()
            if self.verbose:
                print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
            return model
        elif self.method == 'mvnmf':
            if self.mvnmf_hyperparameter_method == 'all':
                if self.bootstrap:
                    X_in = bootstrap_count_matrix(self.X)
                else:
                    X_in = self.X
                if self.normalize_X:
                    X_in = normalize(X_in, norm='l1', axis=0)
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
                                     ncpu=1,
                                     noise=self.mvnmf_noise
                                    )
                model.fit()
                if self.verbose:
                    print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
                    print('Selected lambda_tilde = %.3g ' % model.lambda_tilde)
                return model
            elif self.mvnmf_hyperparameter_method == 'fixed':
                if self.bootstrap:
                    X_in = bootstrap_count_matrix(self.X)
                else:
                    X_in = self.X
                if self.normalize_X:
                    X_in = normalize(X_in, norm='l1', axis=0)
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
                model.fit()
                if self.verbose:
                    print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
                return model
            elif self.mvnmf_hyperparameter_method == 'single':
                if self.bootstrap:
                    X_in = bootstrap_count_matrix(self.X)
                else:
                    X_in = self.X
                if self.normalize_X:
                    X_in = normalize(X_in, norm='l1', axis=0)
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
                model.fit()
                if self.verbose:
                    print('n_components = ' + str(n_components) + ', replicate ' + str(index_replicate) + ' finished.')
                return model

    def _run_jobs(self):
        self.W_raw_all = {} # Save all raw results
        self.H_raw_all = {} # Save all raw results
        self._W_raw_all = {}
        self._H_raw_all = {}
        self.lambda_tilde_all = {} # Save lambda_tilde's used for each mvNMF run
        start = time.time()
        for n_components in self.n_components_all:
            if self.verbose:
                print('Extracting signatures for n_components = ' + str(n_components) + '..................')
            if self.method == 'nmf':
                parameters = [(index_replicate, n_components, None) for index_replicate in range(0, self.n_replicates)]
                # Note that after workers are created, modifications of global variables won't be seen by the workers.
                # Therefore, any modifications must be made before the workers are created.
                # This is why we need to recreate workers for each n_components.
                workers = multiprocessing.Pool(self.ncpu)
                models = workers.map(self._job, parameters)
                workers.close()
                workers.join()
            elif self.method == 'mvnmf':
                if self.mvnmf_hyperparameter_method == 'all':
                    parameters = [(index_replicate, n_components, None) for index_replicate in range(0, self.n_replicates)]
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
                    if self.normalize_X:
                        X_in = normalize(X_in, norm='l1', axis=0)
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
                                         ncpu=self.ncpu,
                                         noise=self.mvnmf_noise
                                        )
                    model.fit()
                    models = [model]
                    lambda_tilde = model.lambda_tilde
                    if self.verbose:
                        print('Selected lambda_tilde = %.3g. This lambda_tilde will be used for all subsequent mvNMF runs.' % model.lambda_tilde)
                    # Run the rest of the models, using preselected hyperparameter
                    parameters = [(index_replicate, n_components, lambda_tilde) for index_replicate in range(1, self.n_replicates)]
                    workers = multiprocessing.Pool(self.ncpu)
                    _models = workers.map(self._job, parameters)
                    workers.close()
                    workers.join()
                    models.extend(_models)
                elif self.mvnmf_hyperparameter_method == 'fixed':
                    if type(self.mvnmf_lambda_tilde_grid) is not float:
                        raise ValueError('When mvnmf_hyperparameter_method is set to fixed, a single float value must be provided for mvnmf_lambda_tilde_grid.')
                    parameters = [(index_replicate, n_components, None) for index_replicate in range(0, self.n_replicates)]
                    workers = multiprocessing.Pool(self.ncpu)
                    models = workers.map(self._job, parameters)
                    workers.close()
                    workers.join()
            self.W_raw_all[n_components] = [model.W for model in models] # Save all raw results
            self.H_raw_all[n_components] = [model.H for model in models] # Save all raw results
            self._W_raw_all[n_components] = [model._W for model in models]
            self._H_raw_all[n_components] = [model._H for model in models]
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
            if self.verbose:
                print('Time elapsed: %.3g seconds.' % (time.time() - start))
        return self

    def postprocess(self):
        """Filter and gather raw results, and then select the best n_components"""
        ### Parameter validation
        if self.select_method == 'consistency':
            if self.cluster_method != 'hierarchical':
                warnings.warn('Select_method is consistency. However, cluster_method is %r. It is better to use cluster_method = hierarchical when select_method is consistency.' % self.cluster_method,
                              UserWarning)
        elif self.select_method in ['algorithm1', 'algorithm1.1', 'algorithm2', 'algorithm2.1']:
            pass
        else:
            raise ValueError('Invalid select_method.')
        if self.cluster_method not in ['hierarchical', 'cluster_by_matching', 'matching']:
            raise ValueError('Invalid cluster_method.')

        ### Filter and gather results
        self.W_all = {}
        self.H_all = {}
        Ws_filtered_all = {}
        self.sil_score_all = {}
        self.sil_score_mean_all = {}
        self.reconstruction_error_all = {}
        self.n_replicates_after_filtering_all = {}
        self.retained_indices_after_filtering_all = {}
        self.n_support_all = {}
        for n_components in self.n_components_all:
            # Filter
            if self.filter:
                Ws, Hs, retained_indices = _filter_results(
                    self.X, self.W_raw_all[n_components], self.H_raw_all[n_components],
                    method=self.filter_method, thresh=self.filter_thresh, percentile=self.filter_percentile
                )
            else:
                Ws = self.W_raw_all[n_components]
                Hs = self.H_raw_all[n_components]
                retained_indices = np.arange(0, self.n_replicates)
            self.retained_indices_after_filtering_all[n_components] = retained_indices
            self.n_replicates_after_filtering_all[n_components] = len(Ws)
            Ws_filtered_all[n_components] = Ws
            # Gather
            W, H, sil_score, sil_score_mean, n_support = _gather_results(self.X, Ws, method=self.cluster_method)
            self.W_all[n_components] = W
            self.H_all[n_components] = H
            self.sil_score_all[n_components] = sil_score
            self.sil_score_mean_all[n_components] = sil_score_mean
            self.n_support_all[n_components] = n_support
            self.reconstruction_error_all[n_components] = beta_divergence(self.X, W @ H, beta=1, square_root=False)
        self.samplewise_reconstruction_errors_all = {
            n_components: _samplewise_error(self.X, self.W_all[n_components] @ self.H_all[n_components]) for n_components in self.n_components_all
        }

        ### Select n_components
        self.n_components, self.optimal_k_all, self.n_components_consistent, self.n_components_stable, self.pvalue_all, self.pvalue_tail_all = _select_n_components(
            self.n_components_all,
            self.samplewise_reconstruction_errors_all,
            self.sil_score_all,
            self.n_replicates,
            self.n_replicates_after_filtering_all,
            Ws_all=Ws_filtered_all,
            pthresh=self.select_pthresh,
            sil_score_mean_thresh=self.select_sil_score_mean_thresh,
            sil_score_min_thresh=self.select_sil_score_min_thresh,
            n_replicates_filter_ratio_thresh=self.select_n_replicates_filter_ratio_thresh,
            method=self.select_method,
            nrefs=50,
            max_k_all=None
        )
        self.W = self.W_all[self.n_components]
        self.H = self.H_all[self.n_components]
        self.sil_score = self.sil_score_all[self.n_components]
        self.sil_score_mean = self.sil_score_mean_all[self.n_components]
        self.reconstruction_error = self.reconstruction_error_all[self.n_components]
        self.samplewise_reconstruction_errors = self.samplewise_reconstruction_errors_all[self.n_components]
        self.n_support = self.n_support_all[self.n_components]
        self.W_df = pd.DataFrame(self.W, columns=['Sig' + str(i) for i in range(1, self.n_components + 1)],
                                 index=self.features)
        self.signatures = self.W_df.columns.values
        self.H_df = pd.DataFrame(self.H, columns=self.samples, index=self.signatures)
        return self

    def fit(self):
        self._run_jobs()
        self.postprocess()
        return self

    def plot_selection(self, title=None, plot_pvalues=True, outfile=None, figsize=None):
        if not hasattr(self, 'W'):
            raise ValueError('The model has not been fitted.')
        sns.set_style('ticks')
        ##### Collect data
        sil_score_mean = np.array([self.sil_score_mean_all[n_components] for n_components in self.n_components_all])
        mean_sil_score = np.array([np.mean(self.sil_score_all[n_components]) for n_components in self.n_components_all])
        min_sil_score = np.array([np.min(self.sil_score_all[n_components]) for n_components in self.n_components_all])
        reconstruction_error = np.array([self.reconstruction_error_all[n_components] for n_components in self.n_components_all])
        pvalues = self.pvalue_all
        pvalues_tail = self.pvalue_tail_all
        if self.optimal_k_all is not None:
            optimal_k = np.array([self.optimal_k_all[n_components] for n_components in self.n_components_all])
        sil_score_all_df = pd.DataFrame.from_dict(self.sil_score_all, orient='index')
        sil_score_all_df.columns=[i for i in range(1, sil_score_all_df.shape[1] + 1)]

        ##### Plot
        if figsize is None:
            if self.optimal_k_all is None:
                figsize = (16, 4)
            else:
                figsize = (21, 4)
        ## Set up the axes with gridspec
        fig = plt.figure(figsize=figsize)
        if self.optimal_k_all is None:
            grid = plt.GridSpec(1, 16, hspace=0.5, wspace=4)
        else:
            grid = plt.GridSpec(1, 21, hspace=0.5, wspace=4)
        host = fig.add_subplot(grid[0, 0:8])
        plt2 = host.twinx()
        if plot_pvalues:
            plt3 = host.twinx()
        heatmap = fig.add_subplot(grid[0, 10:16])
        if self.optimal_k_all is not None:
            kplot = fig.add_subplot(grid[0, 16:21])

        ## Generate line plot
        host.set_xlabel("n components")
        host.set_ylabel("Mean silhouette score")
        plt2.set_ylabel("Reconstruction error")
        if plot_pvalues:
            plt3.set_ylabel("p-value")

        color1 = '#E94E1B'
        color2 = '#1D71B8'
        if plot_pvalues:
            color3 = '#2FAC66'

        p1, = host.plot(self.n_components_all, sil_score_mean,
                        color=color1, label="Mean silhouette score",
                        linestyle='--', marker='o')
        p2, = plt2.plot(self.n_components_all, reconstruction_error,
                        color=color2, label="Reconstruction error",
                        linestyle=':', marker='D')
        if plot_pvalues:
            p3, = plt3.plot(self.n_components_all[1:], pvalues,
                            color=color3, label="p-value",
                            linestyle='-.', marker='+', alpha=0.5)
            p3_, = plt3.plot(self.n_components_all[1:], pvalues_tail,
                             color=color3, label="p-value (tail)",
                             linestyle='-.', marker='.', alpha=0.5)

        if plot_pvalues:
            lns = [p1, p2, p3, p3_]
        else:
            lns = [p1, p2]
        host.legend(handles=lns, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.5))

        host.yaxis.label.set_color(p1.get_color())
        plt2.yaxis.label.set_color(p2.get_color())

        if plot_pvalues:
            plt3.yaxis.label.set_color(p3.get_color())

        #Adjust p-value spine position
        if plot_pvalues:
            plt3.spines['right'].set_position(('outward', 65))

        #Set ticks interval to 1
        host.xaxis.set_major_locator(ticker.MultipleLocator(1))

        #Higlight suggested signature
        host.axvspan(self.n_components - 0.25, self.n_components + 0.25, color='grey', alpha=0.3)

        #Set title
        if title is not None:
            host.set_title('Silhouette scores and reconstruction errors for ' + title)
        else:
            host.set_title('Silhouette scores and reconstruction errors')

        ## Generate heatmap
        heatmap = sns.heatmap(sil_score_all_df, vmin=0, vmax=1, cmap="YlGnBu", ax=heatmap, square=True)
        heatmap.set_xlabel("Signatures")
        heatmap.set_ylabel("n components")

        if title is not None:
            heatmap.set_title('Silhouette scores for ' + title)
        else:
            heatmap.set_title('Silhouette scores')

        ## Generate kplot
        if self.optimal_k_all is not None:
            kplot.plot(self.n_components_all, optimal_k,
                       color='k', label="Optimal k",
                       linestyle='--', marker='o')
            kplot.set_xlabel("n components")
            kplot.set_ylabel("Optimal k")
            if title is not None:
                kplot.set_title("Consistency test for " + title)
            else:
                kplot.set_title("Consistency test")
            kplot.xaxis.set_major_locator(ticker.MultipleLocator(1))
            kplot.yaxis.set_major_locator(ticker.MultipleLocator(1))
            kplot.axvspan(self.n_components - 0.25, self.n_components + 0.25, color='grey', alpha=0.3)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

    def assign(self, W_catalog, method_assign='likelihood_bidirectional',
               thresh_match=None, thresh_refit=None, thresh_new_sig=0.8, indices_associated_sigs=None):
        # Check if fit has been run
        if not hasattr(self, 'W'):
            raise ValueError('The model has not been fitted.')
        if hasattr(self, '_assign_is_run'):
            warnings.warn('self.assign has been previously called. This call will replace the previous results.',
                          UserWarning)
        if hasattr(self, '_assign_grid_is_run'):
            warnings.warn('self.assign_grid has been previously called. Therefore running self.assign may make some attributes inconsistent. Be careful.',
                          UserWarning)
        # Check input
        if not isinstance(W_catalog, pd.DataFrame):
            raise ValueError('W_catalog needs to be a pd.DataFrame.')
        if self.n_features != W_catalog.shape[0]:
            raise ValueError('W_catalog has the wrong number of features.')
        if np.sum(self.features == W_catalog.index.values) != self.n_features:
            warnings.warn('W_catalog has different feature names. The feature names of W_catalog will be converted to self.features. Make sure that the features match.',
                          UserWarning)
            W_catalog.index = self.features
        if thresh_match is None:
            thresh_match = 0.01
        if thresh_refit is None:
            thresh_refit = 0.01
        self.W_catalog = W_catalog
        self.method_assign = method_assign
        self.thresh_match = thresh_match
        self.thresh_refit = thresh_refit
        self.thresh_new_sig = thresh_new_sig
        self.indices_associated_sigs = indices_associated_sigs
        self.W_s, self.H_s, self.sig_map = assign(self.X_df, self.W_df, self.W_catalog, method=self.method_assign,
                                                  thresh_match=self.thresh_match, thresh_refit=self.thresh_refit, thresh_new_sig=self.thresh_new_sig,
                                                  indices_associated_sigs=self.indices_associated_sigs)
        self.sigs_assigned = self.H_s.index[self.H_s.sum(1) > 0].values
        self.n_sigs_assigned = len(self.sigs_assigned)
        self.W_s = pd.DataFrame.copy(self.W_s[self.sigs_assigned])
        self.H_s = pd.DataFrame.copy(self.H_s.loc[self.sigs_assigned])
        self._assign_is_run = True
        return self

    def assign_grid(self, W_catalog, method_assign='likelihood_bidirectional',
                    thresh_match_grid=None, thresh_refit_grid=None, thresh_new_sig=0.8, indices_associated_sigs=None):
        # Check if fit has been run
        if not hasattr(self, 'W'):
            raise ValueError('The model has not been fitted.')
        if hasattr(self, '_assign_is_run'):
            warnings.warn('self.assign has been previously called. Therefore running self.assign_grid may make some attributes inconsistent. Be careful.',
                          UserWarning)
        if hasattr(self, '_assign_grid_is_run'):
            warnings.warn('self.assign_grid has been previously called. This call will replace the previous results.',
                          UserWarning)
        # Check input
        if not isinstance(W_catalog, pd.DataFrame):
            raise ValueError('W_catalog needs to be a pd.DataFrame.')
        if self.n_features != W_catalog.shape[0]:
            raise ValueError('W_catalog has the wrong number of features.')
        if np.sum(self.features == W_catalog.index.values) != self.n_features:
            warnings.warn('W_catalog has different feature names. The feature names of W_catalog will be converted to self.features. Make sure that the features match.',
                          UserWarning)
            W_catalog.index = self.features
        if thresh_match_grid is None:
            thresh_match_grid = np.array([0.01])
        if thresh_refit_grid is None:
            thresh_refit_grid = np.array([0.01])
        self.W_catalog = W_catalog
        self.method_assign = method_assign
        self.thresh_match_grid = thresh_match_grid
        self.thresh_refit_grid = thresh_refit_grid
        self.thresh_new_sig = thresh_new_sig
        self.indices_associated_sigs = indices_associated_sigs
        W_s_grid_1d, H_s_grid, sig_map_grid_1d, thresh_match_grid_unique = assign_grid(
            self.X_df, self.W_df, self.W_catalog, method=self.method_assign,
            thresh_match_grid=self.thresh_match_grid, thresh_refit_grid=self.thresh_refit_grid,
            thresh_new_sig=self.thresh_new_sig,
            indices_associated_sigs=self.indices_associated_sigs,
            ncpu=self.ncpu, verbose=self.verbose
        )
        self.sig_map_grid_1d = sig_map_grid_1d
        self.W_s_grid = {}
        self.H_s_grid = {}
        self.sigs_assigned_grid = {}
        self.n_sigs_assigned_grid = {}
        for thresh_match in self.thresh_match_grid:
            for thresh_refit in self.thresh_refit_grid:
                sigs_assigned = H_s_grid[thresh_match][thresh_refit].index[H_s_grid[thresh_match][thresh_refit].sum(1) > 0].values
                self.sigs_assigned_grid[(thresh_match, thresh_refit)] = sigs_assigned
                self.W_s_grid[(thresh_match, thresh_refit)] = pd.DataFrame.copy(W_s_grid_1d[thresh_match][sigs_assigned])
                self.H_s_grid[(thresh_match, thresh_refit)] = pd.DataFrame.copy(H_s_grid[thresh_match][thresh_refit].loc[sigs_assigned])
                self.n_sigs_assigned_grid[(thresh_match, thresh_refit)] = len(sigs_assigned)
        self.thresh_match_grid_unique = thresh_match_grid_unique
        self._assign_grid_is_run = True
        return self

    def _reinstantiate(self, X_new):
        """Simply instantiate a new unfitted object with the same parameters.

        For copying the entire object, use copy.deepcopy().
        """
        model = DenovoSig(
            X_new,
            min_n_components=self.min_n_components,
            max_n_components=self.max_n_components,
            init=self.init,
            method=self.method,
            normalize_X=self.normalize_X,
            bootstrap=self.bootstrap,
            n_replicates=self.n_replicates,
            max_iter=self.max_iter,
            min_iter=self.min_iter,
            conv_test_freq=self.conv_test_freq,
            conv_test_baseline=self.conv_test_baseline,
            tol=self.tol,
            ncpu=self.ncpu,
            verbose=self.verbose,
            # Specific for result filtering:
            filter=self.filter,
            filter_method=self.filter_method,
            filter_thresh=self.filter_thresh,
            filter_percentile=self.filter_percentile,
            # Specific for result gathering:
            cluster_method=self.cluster_method,
            # Specific for n_components selection:
            select_method=self.select_method,
            select_pthresh=self.select_pthresh,
            select_sil_score_mean_thresh=self.select_sil_score_mean_thresh,
            select_sil_score_min_thresh=self.select_sil_score_min_thresh,
            select_n_replicates_filter_ratio_thresh=self.select_n_replicates_filter_ratio_thresh,
            # mvnmf specific:
            mvnmf_hyperparameter_method=self.mvnmf_hyperparameter_method,
            mvnmf_lambda_tilde_grid=self.mvnmf_lambda_tilde_grid,
            mvnmf_delta=self.mvnmf_delta,
            mvnmf_gamma=self.mvnmf_gamma,
            mvnmf_pthresh=self.mvnmf_pthresh,
            mvnmf_noise=self.mvnmf_noise
        )
        return model

    def validate(self, W_s=None, H_s=None, validate_n_replicates=1):
        """Validate a single assignment result.

        If you want to validate an external assignment result, provide W_s and H_s.
        """
        ################# Check running status and input
        if not hasattr(self, 'W'):
            raise ValueError('The model has not been fitted.')
        if W_s is not None and H_s is not None:
            if hasattr(self, 'W_s') or hasattr(self, 'H_s'):
                warnings.warn('Signature assignment W_s or H_s already present in the model. '
                              'However, external W_s and H_s are provided. The external '
                              'assignment will be validated and the original assignment will be lost. Be careful',
                              UserWarning)
            if not isinstance(W_s, pd.DataFrame):
                raise ValueError('W_s needs to be a pd.DataFrame.')
            if not isinstance(H_s, pd.DataFrame):
                raise ValueError('H_s needs to be a pd.DataFrame.')
            if W_s.index.tolist() != list(self.features):
                raise ValueError('The index of W_s does not match model.features.')
            if H_s.columns.tolist() != list(self.samples):
                raise ValueError('The columns of H_s do not match model.samples.')
            if H_s.index.tolist() != W_s.columns.tolist():
                raise ValueError('The provided W_s and H_s do not have matched signatures.')
            self.W_s = W_s
            self.H_s = H_s
        elif W_s is None and H_s is None:
            if not hasattr(self, 'W_s') or not hasattr(self, 'H_s'):
                raise ValueError('Signature assignment has not been done yet, and external assignment is not provided. '
                                 'Run assign or validate_grid first. Or provide external assignment.')
        else:
            raise ValueError('Either provide both W_s and H_s, or set both to None.')
        #####################
        self.validate_n_replicates = validate_n_replicates
        self.X_simul = []
        self.W_simul = []
        self.H_simul = []
        self.W_cos_dist = []
        self.H_frobenius_dist = []
        ##################### Run validation
        for i in range(0, self.validate_n_replicates):
            # Simulate data
            X_simul = pd.DataFrame(simulate_count_matrix(self.W_s.values, self.H_s.values),
                                   columns=self.samples, index=self.features)
            # Rerun de novo discovery
            model_simul = self._reinstantiate(X_simul)
            model_simul.min_n_components = self.n_components # Fix n_components
            model_simul.max_n_components = self.n_components # Fix n_components
            model_simul.n_components_all = np.array([self.n_components]) # Fix n_components.
            # The line above is necessary because n_components_all is defined in __init__(), which is not a good thing to do.
            # We should remove any calculation within __init__(). Then the above line will not be needed.
            if self.method == 'mvnmf': # If mvnmf, fix lambda_tilde
                model_simul.mvnmf_hyperparameter_method = 'fixed'
                model_simul.mvnmf_lambda_tilde_grid = float(self.lambda_tilde_all[self.n_components][0])
            model_simul.fit()
            # Collect data
            W_simul, col_index, pdist = match_catalog_pair(self.W, model_simul.W, metric='cosine')
            H_simul = model_simul.H[col_index, :]
            W_cos_dist = pdist[np.arange(0, self.n_components), col_index].mean()
            H_frobenius_dist = beta_divergence(self.H, H_simul, beta=2, square_root=True)
            self.X_simul.append(X_simul)
            self.W_simul.append(pd.DataFrame(W_simul, columns=self.signatures, index=self.features))
            self.H_simul.append(pd.DataFrame(H_simul, columns=self.samples, index=self.signatures))
            self.W_cos_dist.append(W_cos_dist)
            self.H_frobenius_dist.append(H_frobenius_dist)
        self.W_cos_dist_mean = np.mean(self.W_cos_dist)
        self.H_frobenius_dist_mean = np.mean(self.H_frobenius_dist)
        return self

    def validate_grid(self, validate_n_replicates=1, W_selection_method='pvalue', W_cos_dist_thresh=0.02, W_pvalue_thresh=0.05):
        """Validation on a grid.

        W_selection_method: 'pvalue' or 'cosine'
        TODO:
        1. Separate running the simulation and selecting the best grid point. So that we can redo selection without rerunning simulation. 
        """
        ################# Check running status and input
        if not hasattr(self, 'W'):
            raise ValueError('The model has not been fitted.')
        if not hasattr(self, '_assign_grid_is_run'):
            raise ValueError('Run assign_grid first.')
        if W_selection_method not in ['pvalue', 'cosine']:
            raise ValueError('Bad input for W_selection_method.')
        ################# Run validation
        self.W_selection_method = W_selection_method
        self.W_pvalue_thresh = W_pvalue_thresh
        self.validate_n_replicates = validate_n_replicates
        self.W_cos_dist_thresh = W_cos_dist_thresh
        self.X_simul_grid = {}
        self.W_simul_grid = {}
        self.H_simul_grid = {}
        self.W_cos_dist_grid = {}
        self.H_frobenius_dist_grid = {}
        self.W_cos_dist_mean_grid = {}
        self.H_frobenius_dist_mean_grid = {}
        # Note that only unique thresh_match will be run.
        for thresh_match in self.thresh_match_grid_unique:
            for thresh_refit in self.thresh_refit_grid:
                X_simul = []
                W_simul = []
                H_simul = []
                W_cos_dist = []
                H_frobenius_dist = []
                for i in range(0, self.validate_n_replicates):
                    # Simulate data
                    _X_simul = pd.DataFrame(simulate_count_matrix(self.W_s_grid[(thresh_match, thresh_refit)].values, self.H_s_grid[(thresh_match, thresh_refit)].values),
                                            columns=self.samples, index=self.features)
                    # Rerun de novo discovery
                    model_simul = self._reinstantiate(_X_simul)
                    model_simul.min_n_components = self.n_components # Fix n_components
                    model_simul.max_n_components = self.n_components # Fix n_components
                    model_simul.n_components_all = np.array([self.n_components]) # Fix n_components.
                    if self.method == 'mvnmf': # If mvnmf, fix lambda_tilde
                        model_simul.mvnmf_hyperparameter_method = 'fixed'
                        model_simul.mvnmf_lambda_tilde_grid = float(self.lambda_tilde_all[self.n_components][0])
                    model_simul.fit()
                    # Collect data
                    _W_simul, col_index, pdist = match_catalog_pair(self.W, model_simul.W, metric='cosine')
                    _H_simul = model_simul.H[col_index, :]
                    _W_cos_dist = pdist[np.arange(0, self.n_components), col_index].mean()
                    _H_frobenius_dist = beta_divergence(self.H, _H_simul, beta=2, square_root=True)
                    X_simul.append(_X_simul)
                    W_simul.append(pd.DataFrame(_W_simul, columns=self.signatures, index=self.features))
                    H_simul.append(pd.DataFrame(_H_simul, columns=self.samples, index=self.signatures))
                    W_cos_dist.append(_W_cos_dist)
                    H_frobenius_dist.append(_H_frobenius_dist)
                self.X_simul_grid[(thresh_match, thresh_refit)] = X_simul
                self.W_simul_grid[(thresh_match, thresh_refit)] = W_simul
                self.H_simul_grid[(thresh_match, thresh_refit)] = H_simul
                self.W_cos_dist_grid[(thresh_match, thresh_refit)] = W_cos_dist
                self.H_frobenius_dist_grid[(thresh_match, thresh_refit)] = H_frobenius_dist
                self.W_cos_dist_mean_grid[(thresh_match, thresh_refit)] = np.mean(W_cos_dist)
                self.H_frobenius_dist_mean_grid[(thresh_match, thresh_refit)] = np.mean(H_frobenius_dist)
        ################# Select best result
        if self.W_selection_method == 'cosine':
            # Smallest W_cos_dist
            W_cos_dist_min = np.min(list(self.W_cos_dist_mean_grid.values()))
            # Candidates: W_cos_dist within W_cos_dist_min + self.W_cos_dist_thresh
            candidate_grid_points = [key for key, value in self.W_cos_dist_mean_grid.items() if value <= W_cos_dist_min + self.W_cos_dist_thresh]
        elif self.W_selection_method == 'pvalue':
            # Element wise errors between W_simul and W_data
            elementwise_errors_W = []
            for key, value in self.W_simul_grid.items():
                W_simul = np.mean(value, 0) # We take the average signatures of multiple replicates of simulation results here.
                error = np.absolute((self.W - W_simul).flatten())
                elementwise_errors_W.append([key, np.mean(error), error])
            # Best result
            elementwise_errors_W = sorted(elementwise_errors_W, key=itemgetter(1))
            error_best = elementwise_errors_W[0][2]
            # Test differences
            self.W_simul_error_pvalue_grid = {}
            self.W_simul_error_pvalue_tail_grid = {}
            candidate_grid_points = []
            for key, _, error in elementwise_errors_W:
                pvalue = stats.mannwhitneyu(error_best, error, alternative='less')[1]
                pvalue_tail = differential_tail_test(error_best, error, percentile=90, alternative='less')[1]
                self.W_simul_error_pvalue_grid[key] = pvalue
                self.W_simul_error_pvalue_tail_grid[key] = pvalue_tail
                if pvalue > self.W_pvalue_thresh and pvalue_tail > self.W_pvalue_thresh:
                    candidate_grid_points.append(key)
        ########
        if len(candidate_grid_points) == 1:
            self.best_grid_point = candidate_grid_points[0]
        else:
            # Avoid using new signatures
            # Select those solutions where sigs_assigned does not contain any de novo signatures.
            _candidate_grid_points = [key for key in candidate_grid_points if len(set(self.sigs_assigned_grid[key]).intersection(self.signatures)) == 0]
            if len(_candidate_grid_points) == 0:
                warnings.warn('During grid search, all reasonable solutions contain new signatures. This could potentially mean that '
                              'a new signature does exist. Or, try increasing W_cos_dist_thresh or decreasing thresh_new_sig.',
                              UserWarning)
                # Keep candidate_grid_points unchanged
            else:
                candidate_grid_points = _candidate_grid_points
            ###
            if len(candidate_grid_points) == 1:
                self.best_grid_point = candidate_grid_points[0]
            else:
                # Look at number of assigned sigs
                n_sigs_assigned_min = np.min([self.n_sigs_assigned_grid[key] for key in candidate_grid_points])
                candidate_grid_points = [key for key in candidate_grid_points if self.n_sigs_assigned_grid[key] == n_sigs_assigned_min]
                if len(candidate_grid_points) == 1:
                    self.best_grid_point = candidate_grid_points[0]
                else:
                    # Look at H error
                    # Or choose the one with the strongest sparsity here.
                    tmp = [[key, self.H_frobenius_dist_mean_grid[key]] for key in candidate_grid_points]
                    tmp = sorted(tmp, key=itemgetter(1))
                    self.best_grid_point = tmp[0][0]
        self.thresh_match = self.best_grid_point[0]
        self.thresh_refit = self.best_grid_point[1]
        self.W_s = self.W_s_grid[self.best_grid_point]
        self.H_s = self.H_s_grid[self.best_grid_point]
        self.sigs_assigned = self.sigs_assigned_grid[self.best_grid_point]
        self.n_sigs_assigned = self.n_sigs_assigned_grid[self.best_grid_point]
        self.W_cos_dist = self.W_cos_dist_grid[self.best_grid_point]
        self.W_cos_dist_mean = self.W_cos_dist_mean_grid[self.best_grid_point]
        self.H_frobenius_dist = self.H_frobenius_dist_grid[self.best_grid_point]
        self.H_frobenius_dist_mean = self.H_frobenius_dist_mean_grid[self.best_grid_point]
        return self

    ###########################################################################
    ############################# Old codes below #############################
    ###########################################################################

    def set_params(self,
                   use_catalog = None,
                   catalog_name = None,
                   thresh1_match = None,
                   thresh2_match = None,
                   thresh_new_sig = None,
                   method_sparse = None,
                   thresh1 = None,
                   thresh2 = None):

        if use_catalog != None:
           self.use_catalog = use_catalog
        if catalog_name != None:
           self.catalog_name = catalog_name
        if thresh1_match != None:
           self.thresh1_match = thresh1_match
        if thresh2_match != None:
           self.thresh2_match = thresh2_match
        if thresh_new_sig != None:
           self.thresh_new_sig = thresh_new_sig
        if method_sparse != None:
            self.method_sparse = method_sparse
        if thresh1 != None:
            self.thresh1 = thresh1
        if thresh2 != None:
            self.thresh2 = thresh2

        return self

    def clone_model(self, X_new, grid_index = 1):
        print(grid_index)
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
            model_new.set_params(thresh1 = [self.thresh1_all[grid_index]],
                                 thresh2 = [self.thresh2_all[grid_index]])
        else:
            model_new.set_params(thresh1 = [self.thresh1],
                                 thresh2 = [self.thresh2])

        if self.use_catalog:
            if self.n_grid > 1:
                model_new.set_params(thresh1_match = [self.thresh1_match_all[grid_index]],
                                     thresh2_match = [self.thresh2_match_all[grid_index]],
                                     thresh_new_sig = [self.thresh_new_sig_all[grid_index]])
            else:
                model_new.set_params(thresh1_match = [self.thresh1_match],
                                    thresh2_match = [self.thresh2_match])

        return model_new


    def clear_grid(self):
        if hasattr(self, 'thresh1_all'):
            self.thresh1_all = None
        if hasattr(self, 'thresh2_all'):
            self.thresh2_all = None
        if hasattr(self, 'thresh1_match_all'):
            self.thresh1_match_all = None
        if hasattr(self, 'thresh2_match_all'):
            self.thresh2_match_all = None
        if hasattr(self, 'thresh_new_sig_all'):
            self.thresh_new_sig_all = None
        return self

    def run_reassign(self, W_catalog, signatures, clear = True):
        W_s, H_s, signames, reconstruction_error_s_all, n_grid, thresh1_all, thresh2_all, thresh1_match_all, thresh2_match_all, thresh_new_sig_all = reassign(self, W_catalog = W_catalog, signatures = signatures)


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
            self.thresh1_all = thresh1_all
            self.thresh2_all = thresh2_all
            self.thresh_new_sig_all = thresh_new_sig_all
            self.reconstruction_error_s_all = reconstruction_error_s_all
            if len(thresh1_match_all) > 0:
                 self.thresh1_match_all = thresh1_match_all
            if len(thresh2_match_all) > 0:
                 self.thresh2_match_all = thresh2_match_all
            if len(thresh_new_sig_all) > 0:
                self.thresh_new_sig_all = thresh_new_sig_all
        return self

    def validate_assignment(self, use_refit = False, clear_grid = False):

        W_simul_all, H_simul_all, X_simul_all, best_grid_index, best_grid_index_sum, best_grid_indices, best_grid_indices_sum, error_W_all, error_H_all, dist_W_all, dist_max_sig_index_all,  dist_max_all, dist_sum_all = validate(self)

        self.W_simul_all = W_simul_all
        self.H_simul_all = H_simul_all
        self.X_simul_all = X_simul_all
        self.best_grid_index = best_grid_index
        self.best_grid_index_sum = best_grid_index_sum
        self.best_grid_indices = best_grid_indices
        self.best_grid_indices_sum = best_grid_indices_sum
        self.error_W_all = error_W_all
        self.error_H_all = error_H_all
        self.dist_W_all = dist_W_all
        self.dist_max_all = dist_max_all
        self.dist_sum_all = dist_sum_all
        self.dist_max_simul_sig_index_all = dist_max_sig_index_all

        # to use maximum distance only change best_grid_index_sum to best_grid_index
        if self.n_grid > 1:
            self.W_s = self.W_s_all[best_grid_index_sum]
            self.H_s = self.H_s_all[best_grid_index_sum]
            self.signature_names = self.signature_names_all[best_grid_index_sum]
            self.reconstruction_error_s = self.reconstruction_error_s_all[best_grid_index_sum]
            self.set_params(thresh1 = [self.thresh1_all[best_grid_index_sum]],
                            thresh2 = [self.thresh2_all[best_grid_index_sum]],
                            thresh1_match = [self.thresh1_match_all[best_grid_index_sum]],
                            thresh2_match = [self.thresh2_match_all[best_grid_index_sum]],
                            thresh_new_sig = [self.thresh_new_sig_all[best_grid_index_sum]])
        self.W_simul = self.W_simul_all[best_grid_index_sum]
        self.H_simul = self.H_simul_all[best_grid_index_sum]
        self.X_simul = self.X_simul_all[best_grid_index_sum]
        self.dist_W_simul = self.dist_W_all[best_grid_index_sum]
        self.error_W_simul = self.error_W_all[best_grid_index_sum]
        self.error_H_simul = self.error_H_all[best_grid_index_sum]
        self.dist_max_simul = self.dist_max_all[best_grid_index_sum]
        self.dist_sum_simul = self.dist_sum_all[best_grid_index_sum]
        if clear_grid:
            self.clear_grid()
        return self
