"""Clustering related functions"""

import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import silhouette_samples
import scipy.stats as stats
import warnings
import pandas as pd
from operator import itemgetter
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from .plot import colorPaletteMathematica97


def _within_cluster_variation(d_square_form, cluster_membership):
    """Within cluster variation.

    Cf. https://statweb.stanford.edu/~gwalther/gap
        (Estimating the number of clusters in a data set via the gap statistic by Tibshirani et al.)

    Parameters:
    ----------
    d_square_form : array-like
        Squared form pairwise distance matrix.

    cluster_membership : array-like
        Cluster membership.
    """
    cluster_indices = sorted(list(set(cluster_membership)))
    Ds = []
    for i in cluster_indices:
        index = (cluster_membership == i)
        d_sub = d_square_form[np.ix_(index, index)]
        Ds.append(np.sum(d_sub)/2/d_sub.shape[0]) # D_r / (2 n_r)
    W = np.sum(Ds)
    return W


def hierarchical_cluster(X, k, metric='cosine', linkage_method='average'):
    """Hierarchical clustering.

    Parameters:
    ----------
    X : array-like of shape (n_features, n_samples)
        Input data matrix. Note that this is in the transposed shape of what
        scipy or sklearn normally requires, but is convenient for signature analysis.

    k : int
        Number of clusters

    metric : str
        Metric

    linkage_method : str
        Linkage method
    """
    d = sp.spatial.distance.pdist(X.T, metric=metric)
    d = d.clip(0)
    d_square_form = sp.spatial.distance.squareform(d)
    linkage = sch.linkage(d, method=linkage_method)
    cluster_membership = sch.fcluster(linkage, k, criterion="maxclust")
    return d_square_form, cluster_membership


class OptimalK:
    """Automatically select the optimal number of clusters for hierarchical clustering

    We use the gap statistic method (without log). References:
    1. See https://www.stat.cmu.edu/~ryantibs/datamining/lectures/06-clus3.pdf for a general introduction.
    2. See https://statweb.stanford.edu/~gwalther/gap for the original paper proposing the gap statistic.
        (Estimating the number of clusters in a data set via the gap statistic by Tibshirani et al.)
    3. See https://core.ac.uk/download/pdf/12172514.pdf for a proposal of removing log in the gap statistic.
        (A comparison of Gap statistic definitions with and with-out logarithm function by Mohajer et al.)
    4. See https://github.com/milesgranger/gap_statistic for a python implementation. They didn't use the original
        definition of Wk as in the Tibshirani paper. They used the definition where centroids are used. For squared
        Euclidean distances, the two definitions are equivalent. But for other metrics such as cosine, they are not.
    We also generate silhouette score based selection. Note that silhouette score is not defined for k = 1.

    TODO:
    ----------
    1. Make it more general to work with any clustering method, as in, e.g., https://github.com/milesgranger/gap_statistic.
    """
    def __init__(self,
                 X,
                 max_k=20,
                 nrefs=50,
                 metric='cosine',
                 linkage_method='average',
                 ref_method='a'
                ):
        self.X = X
        self.n_features, self.n_samples = X.shape
        self.nrefs = nrefs
        if max_k > self.n_samples:
            max_k = self.n_samples
        self.max_k = max_k
        self.ks = np.arange(1, self.max_k + 1)
        self.metric = metric
        self.linkage_method = linkage_method
        self.ref_method = ref_method
        self.select()

    def _simulate_reference_data(self, method='a'):
        """Simulate reference data according to Tibshirani et al.

        Parameters:
        ----------
        method : str, 'a' | 'b'
            Method for simulating the reference data. Methods a and b correspond to those described in
            Tibshirani et al.

        TODO:
        ----------
        1. Implement method='b'.
        2. We can also implement a method where we simply simulate each feature as a uniform distribution between 0 and 1,
            or use some Dirichlet distribution based simulation.
        """
        if method == 'a':
            a = np.min(self.X, axis=1, keepdims=True)
            b = np.max(self.X, axis=1, keepdims=True)
            self.reference_data = [np.random.random_sample(size=(self.n_features, self.n_samples)) * (b - a) + a for i in range(0, self.nrefs)]
            return self.reference_data
        elif method == 'b':
            X = self.X.T
            offset = np.mean(X, 0, keepdims=True)
            X = scale(X, with_mean=True, with_std=False) # mean-center columns
            U, D, VT = np.linalg.svd(X)
            X_prime = X @ VT.T
            a = np.min(X_prime, axis=0, keepdims=True)
            b = np.max(X_prime, axis=0, keepdims=True)
            self.reference_data = [(((np.random.random_sample(size=X.shape) * (b - a) + a) @ VT) + offset).T for i in range(0, self.nrefs)]
            return self.reference_data
        else:
            raise ValueError('Method for _simulate_reference_data can only be a or b currently.')

    @staticmethod
    def _cluster_statistic(X, max_k, metric='cosine', linkage_method='average'):
        d = sp.spatial.distance.pdist(X.T, metric=metric)
        d = d.clip(0)
        d_square_form = sp.spatial.distance.squareform(d)
        linkage = sch.linkage(d, method=linkage_method)
        Wk = []
        silscorek = []
        silscorek_percluster = {}
        for k in range(1, max_k + 1):
            cluster_membership = sch.fcluster(linkage, k, criterion="maxclust")
            Wk.append(_within_cluster_variation(d_square_form, cluster_membership))
            if k == 1:
                silscorek.append(np.nan)
                silscorek_percluster[k] = np.nan
            else:
                silscore_per_sample = silhouette_samples(d_square_form, cluster_membership, metric='precomputed')
                silscorek.append(np.mean(silscore_per_sample))
                silscorek_percluster[k] = np.array([np.mean(silscore_per_sample[cluster_membership == i]) for i in range(1, k + 1)])
        Wk = np.array(Wk)
        silscorek = np.array(silscorek)
        return Wk, silscorek, silscorek_percluster

    def select(self):
        ### First calculate statistics for data itself
        self.Wk, self.silscorek, self.silscorek_percluster = self._cluster_statistic(self.X, self.max_k, metric=self.metric, linkage_method=self.linkage_method)
        self.Wk_log = np.log(self.Wk)
        ### Then calculate statistics for reference data
        # simulate
        self._simulate_reference_data(method=self.ref_method)
        # calculate
        self.Wk_ref_all = []
        self.silscorek_ref_all = []
        self.silscorek_percluster_ref_all = []
        for data in self.reference_data:
            Wk, silscorek, silscorek_percluster = self._cluster_statistic(data, self.max_k, metric=self.metric, linkage_method=self.linkage_method)
            self.Wk_ref_all.append(Wk)
            self.silscorek_ref_all.append(silscorek)
            self.silscorek_percluster_ref_all.append(silscorek_percluster)
        self.Wk_ref_all = np.array(self.Wk_ref_all)
        self.Wk_log_ref_all = np.log(self.Wk_ref_all)
        self.silscorek_ref_all = np.array(self.silscorek_ref_all)
        # aggregate statistics
        self.Wk_ref = np.mean(self.Wk_ref_all, axis=0)
        self.Wk_log_ref = np.mean(self.Wk_log_ref_all, axis=0)
        self.silscorek_ref = np.mean(self.silscorek_ref_all, axis=0)
        self.Wk_ref_sd = np.std(self.Wk_ref_all, axis=0) * np.sqrt(1 + 1/self.nrefs)
        self.silscorek_ref_sd = np.std(self.silscorek_ref_all, axis=0) * np.sqrt(1 + 1/self.nrefs)
        self.Wk_log_ref_sd = np.std(self.Wk_log_ref_all, axis=0) * np.sqrt(1 + 1/self.nrefs)
        ### Calculate gap statistics
        self.gap_statistic = self.Wk_ref - self.Wk
        self.gap_statistic_log = self.Wk_log_ref - self.Wk_log
        ### Selecting optimal k
        # Using Wk
        candidates = self.ks[1:][(self.gap_statistic[1:] - self.Wk_ref_sd[1:] - self.gap_statistic[0:-1]) <= 0]
        self.k_gap_statistic_valid = candidates - 1
        if len(candidates) == 0:
            self.k_gap_statistic = np.nan
        else:
            self.k_gap_statistic = candidates[0] - 1
        # Using Wk_log
        candidates = self.ks[1:][(self.gap_statistic_log[1:] - self.Wk_log_ref_sd[1:] - self.gap_statistic_log[0:-1]) <= 0]
        self.k_gap_statistic_log_valid = candidates - 1
        if len(candidates) == 0:
            self.k_gap_statistic_log = np.nan
        else:
            self.k_gap_statistic_log = candidates[0] - 1
        # Using sil score
        self.k_silscore = self.ks[1:][np.argmax(self.silscorek[1:])]
        # Default
        self.k = self.k_gap_statistic
        self.k_valid = self.k_gap_statistic_valid
        ### Summary
        self.summary = pd.DataFrame({
            'k': self.ks,
            'gap': self.gap_statistic,
            'gap_log': self.gap_statistic_log,
            'sk': self.Wk_ref_sd,
            'sk_log': self.Wk_log_ref_sd,
            'diff': list(self.gap_statistic[0:-1] - self.gap_statistic[1:] + self.Wk_ref_sd[1:]) + [np.nan],
            'diff_log': list(self.gap_statistic_log[0:-1] - self.gap_statistic_log[1:] + self.Wk_log_ref_sd[1:]) + [np.nan],
            'sil_score': self.silscorek,
            'sil_score_ref': self.silscorek_ref,
            'sil_score_ref_std': self.silscorek_ref_sd,
            'k_optimal_gap': (self.ks == self.k_gap_statistic),
            'k_optimal_gap_log': (self.ks == self.k_gap_statistic_log),
            'k_optimal_silscore': (self.ks == self.k_silscore),
            'k_valid_gap': [k in self.k_gap_statistic_valid for k in self.ks],
            'k_valid_gap_log': [k in self.k_gap_statistic_log_valid for k in self.ks]
        })
        self.summary = self.summary.set_index('k')
        ### Finally cluster according to the optimal k
        _, self.cluster_membership = hierarchical_cluster(self.X, self.k, metric=self.metric, linkage_method=self.linkage_method)

    def plot(self, sil_thresh=None, outfile=None):
        mpl.rcParams['pdf.fonttype'] = 42
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        sns.set_context("notebook")
        sns.set_style("ticks")
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)

        subfig = fig.add_subplot(3, 1, 1)
        subfig.set_title("Selection using gap statistic", fontsize=14)
        subfig.spines['right'].set_visible(False)
        subfig.spines['top'].set_visible(False)
        subfig.spines['bottom'].set_color('k')
        subfig.spines['left'].set_color('k')
        for tick in subfig.get_xticklabels():
            tick.set_fontname("Arial")
        for tick in subfig.get_yticklabels():
            tick.set_fontname("Arial")
        subfig.set_xlabel("Number of clusters", fontsize=14)
        subfig.set_ylabel("Gap statistic", fontsize=14)
        subfig.errorbar(self.summary.index, self.summary['gap'], yerr=self.summary['sk'],
                        fmt='.--', markersize=10, capsize=3, capthick=2, color=colorPaletteMathematica97[0], label='All k\'s', zorder=0)
        subfig.plot(self.summary[self.summary['k_valid_gap']].index, self.summary[self.summary['k_valid_gap']]['gap'],
                    '.', markersize=10, color=colorPaletteMathematica97[2], label='Reasonable k\'s', zorder=1)
        subfig.axvspan(self.k_gap_statistic - 0.25, self.k_gap_statistic + 0.25, color='grey', alpha=0.3, label='Selected k', zorder=2)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, prop={'size': 14})
        plt.xticks(self.summary.index)
        plt.xlim(0, self.max_k + 1)

        subfig = fig.add_subplot(3, 1, 2)
        subfig.set_title("Selection using gap statistic (log)", fontsize=14)
        subfig.spines['right'].set_visible(False)
        subfig.spines['top'].set_visible(False)
        subfig.spines['bottom'].set_color('k')
        subfig.spines['left'].set_color('k')
        for tick in subfig.get_xticklabels():
            tick.set_fontname("Arial")
        for tick in subfig.get_yticklabels():
            tick.set_fontname("Arial")
        subfig.set_xlabel("Number of clusters", fontsize=14)
        subfig.set_ylabel("Gap statistic (log)", fontsize=14)
        subfig.errorbar(self.summary.index, self.summary['gap_log'], yerr=self.summary['sk_log'],
                        fmt='.--', markersize=10, capsize=3, capthick=2, color=colorPaletteMathematica97[0], label='All k\'s', zorder=0)
        subfig.plot(self.summary[self.summary['k_valid_gap_log']].index, self.summary[self.summary['k_valid_gap_log']]['gap_log'],
                    '.', markersize=10, color=colorPaletteMathematica97[2], label='Reasonable k\'s', zorder=1)
        subfig.axvspan(self.k_gap_statistic_log - 0.25, self.k_gap_statistic_log + 0.25, color='grey', alpha=0.3, label='Selected k', zorder=2)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, prop={'size': 14})
        plt.xticks(self.summary.index)
        plt.xlim(0, self.max_k + 1)

        subfig = fig.add_subplot(3, 1, 3)
        subfig.set_title("Selection using silhouette score", fontsize=14)
        subfig.spines['right'].set_visible(False)
        subfig.spines['top'].set_visible(False)
        subfig.spines['bottom'].set_color('k')
        subfig.spines['left'].set_color('k')
        for tick in subfig.get_xticklabels():
            tick.set_fontname("Arial")
        for tick in subfig.get_yticklabels():
            tick.set_fontname("Arial")
        subfig.set_xlabel("Number of clusters", fontsize=14)
        subfig.set_ylabel("Silhouette score", fontsize=14)
        subfig.plot(self.summary.index, self.summary['sil_score'], '.--', markersize=10, label='Data', color=colorPaletteMathematica97[0], zorder=1)
        subfig.errorbar(self.summary.index, self.summary['sil_score_ref'], yerr=self.summary['sil_score_ref_std'],
                        fmt='--', markersize=10, capsize=3, capthick=2, label='Reference data', color=colorPaletteMathematica97[1], zorder=2)
        for k in self.ks:
            silscores = self.silscorek_percluster[k]
            if k == 2:
                subfig.scatter(np.ones(k)*k, silscores, marker='x', color='gray', alpha=0.7, label='Data, per cluster', zorder=3)
            else:
                subfig.scatter(np.ones(k)*k, silscores, marker='x', color='gray', alpha=0.7, zorder=3)
        subfig.axvspan(self.k_silscore - 0.25, self.k_silscore + 0.25, color='gray', alpha=0.3, label='Selected k', zorder=4)
        if type(sil_thresh) is float:
            subfig.axhline(y=sil_thresh, linestyle='--', color='red', alpha=0.5, zorder=0)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, prop={'size': 14})
        #plt.legend(prop={'size': 14})
        plt.xticks(self.summary.index)
        plt.xlim(0, self.max_k + 1)
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
