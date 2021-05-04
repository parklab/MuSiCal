import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize
import scipy.stats as stats

from .cluster import OptimalK, hierarchical_cluster


def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def remove_samples_based_on_gini(H, X, gini_baseline = 0.65, gini_delta = 0.005, per_signature = True):
    gini_vec = []
    gini_vec = np.array(gini_vec)
    for h in H:
        h_norm = h/np.sum(X, axis = 0)
        gini_this = gini(h_norm)
        gini_vec = np.append(gini_vec, gini_this)

    inds_columns_to_check = np.where(gini_vec > gini_baseline)

    indices_to_remove = []
    indices_to_remove = np.array(inds_columns_to_check)

    list_indices_to_keep = {}
#    for i in np.array(inds_columns_to_check).tolist()[0]:
    for i in inds_columns_to_check[0]:
        h_norm = H[i,:]/np.sum(X, axis = 0)
        index = h_norm.size
        delta = 1
        while delta > gini_delta:
            gini_this = gini(np.sort(h_norm)[1:index])
            gini_bef = gini(np.sort(h_norm)[1:(index - 1)])
            delta = gini_this - gini_bef
            index = index - 1
            if index < np.around(h_norm.size * 0.8):
                break
        to_keep = np.where(h_norm < np.sort(h_norm)[index])
        to_remove = np.where(h_norm >= np.sort(h_norm)[index])
        list_indices_to_keep[i] = to_keep
        indices_to_remove = np.append(indices_to_remove, to_remove)

    indices_to_remove = np.unique(indices_to_remove)

    if per_signature:
        list_X = {}
        for i in inds_columns_to_check[0]:
            X_this = X[:,list_indices_to_keep[i][0]]
            list_X[i] = X_this
        return(list_X)
    else:
        X_this = np.delete(X, indices_to_remove, axis = 1)
        return(X_this)

def identify_distinct_cluster(X, H, frac_thresh=0.05):
    """Identify distinct clusters from the cohort based on exposures.

    Here 'distinct' means that there are signatures with high exposures in this cluster
    and low exposures in other samples. This function will only do 2-cluster separations.
    One can run this function iteratively until there are no separations.

    Parameters:
    ----------
    X : array-like of shape (n_features, n_samples)
        Input data matrix.

    H : array-like of shape (n_components, n_samples)
        Exposure matrix.

    frac_thresh : float, default 0.05
        Threshold below which exposure fractions will be considered zero.

    Returns:
    ----------
    k : int, 1 | 2
        Number of clusters. If 1, there isn't a distinct cluster, thus no separation is done.
        If 2, there is a distinct cluster, thus the cohort is separated into two clusters.

    clusters : list of numpy arrays
        The indices of samples in the clusters.

    Xs : list of numpy arrays
        Samples in the clusters.

    distinct : list of bool, or None
        When there is a distinct cluster, this attribute informs whether the first or the second cluster is distinct, or both.

    Notes:
    ----------
    1. Consider using .utils.differential_tail_test(). The logic in identifying cluster-specific signatures is basically
        a differential tail test.
    2. Note that compared to Doga's original code, the output now is always either 1 or 2 clusters. Also, the nonspecific signatures
        are not used anymore.
    """
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
    if (type(H) != np.ndarray) or (not np.issubdtype(H.dtype, np.floating)):
            H = np.array(H).astype(float)
    n_components, n_samples = H.shape

    ### Normalize H
    # Several differences from previous codes here:
    # 1. Previous codes modify H in place, i.e., the outside H that is passed into this function will be modified.
    #   This is not desired behavior.
    # 2. Previous codes use X to normalize H. Now H itself is used for the normalization. This should not matter too much.
    # 3. Now, after setting small exposures to 0, we renormalize so that each sample is summed to one. This is conceptually
    #   appealing if we want to use cosine distances later.
    H = normalize(H, norm='l1', axis=0)
    H[H < frac_thresh] = 0.0
    H = normalize(H, norm='l1', axis=0)

    ### Clustering
    d_square_form, cluster_membership = hierarchical_cluster(H, 2, metric='cosine', linkage_method='average')
    clust1_index = np.arange(0, n_samples)[(cluster_membership == 1)]
    clust2_index = np.arange(0, n_samples)[(cluster_membership == 2)]
    H_clust1 = H[:, clust1_index]
    H_clust2 = H[:, clust2_index]

    ### Identify cluster-specific signatures
    n_clust1 = len(clust1_index) # Number of samples in cluster 1
    n_clust2 = len(clust2_index) # Number of samples in cluster 2
    n_clust1_pos = np.sum(H_clust1 > 0, 1) # Number of samples in cluster 1 that have positive exposures of the signature, one per signature
    n_clust2_pos = np.sum(H_clust2 > 0, 1) # Number of samples in cluster 2 that have positive exposures of the signature, one per signature
    pvalue_clust1_greater = [] # p-values for whether the signature has greater exposures in cluster 1 than in cluster 2, one per signature
    pvalue_clust2_greater = [] # p-values for whether the signature has greater exposures in cluster 2 than in cluster 1, one per signature
    for h1, h2 in zip(H_clust1, H_clust2):
        try:
            p = stats.mannwhitneyu(h1, h2, alternative='greater')[1]
        except: # E.g., when the exposure of a signature is all 0 in the dataset. This could happen in later iterations of running the function.
            p = 1.0
        pvalue_clust1_greater.append(p)
        try:
            p = stats.mannwhitneyu(h2, h1, alternative='greater')[1]
        except:
            p = 1.0
        pvalue_clust2_greater.append(p)
    pvalue_clust1_greater = np.array(pvalue_clust1_greater)
    pvalue_clust2_greater = np.array(pvalue_clust2_greater)
    # A signature is considered to be specific to cluster 1, if
    # (the exposure is significantly greater in cluster 1 and less than 10% of samples in cluster 2 have positive exposures) or
    # (more than 90% of samples in cluster 1 and less than 10% of samples in cluster 2 have positive exposures)
    sigs_clust1_specific = np.arange(0, n_components)[np.logical_or(
        np.logical_and(pvalue_clust1_greater < 0.05, n_clust2_pos/n_clust2 < 0.1),
        np.logical_and(n_clust1_pos/n_clust1 > 0.9, n_clust2_pos/n_clust2 < 0.1)
    )]
    sigs_clust2_specific = np.arange(0, n_components)[np.logical_or(
        np.logical_and(pvalue_clust2_greater < 0.05, n_clust1_pos/n_clust1 < 0.1),
        np.logical_and(n_clust2_pos/n_clust2 > 0.9, n_clust1_pos/n_clust1 < 0.1)
    )]
    # If a signature has positive exposures in more than 10% of samples in both cluster 1 and 2, it is considered non-specific.
    #sigs_nonspecific = np.arange(0, n_components)[
    #    np.logical_and(n_clust1_pos/n_clust1 > 0.1, n_clust2_pos/n_clust2 > 0.1)
    #]

    ### Calculate mean exposures for cluster-specific and nonspecific signatures in the 2 clusters separately.
    if len(sigs_clust1_specific) > 0:
        mean_fraction_clust1_sigs_in_clust1 = np.mean(np.sum(H_clust1[sigs_clust1_specific, :], 0))
        mean_fraction_clust1_sigs_in_clust2 = np.mean(np.sum(H_clust2[sigs_clust1_specific, :], 0))
    else:
        mean_fraction_clust1_sigs_in_clust1 = np.nan
        mean_fraction_clust1_sigs_in_clust2 = np.nan
    if len(sigs_clust2_specific) > 0:
        mean_fraction_clust2_sigs_in_clust1 = np.mean(np.sum(H_clust1[sigs_clust2_specific, :], 0))
        mean_fraction_clust2_sigs_in_clust2 = np.mean(np.sum(H_clust2[sigs_clust2_specific, :], 0))
    else:
        mean_fraction_clust2_sigs_in_clust1 = np.nan
        mean_fraction_clust2_sigs_in_clust2 = np.nan
    #if len(sigs_nonspecific) > 0:
    #    mean_fraction_nonspecific_sigs_in_clust1 = np.mean(np.sum(H_clust1[sigs_nonspecific, :], 0))
    #    mean_fraction_nonspecific_sigs_in_clust2 = np.mean(np.sum(H_clust2[sigs_nonspecific, :], 0))
    #else:
    #    mean_fraction_nonspecific_sigs_in_clust1 = np.nan
    #    mean_fraction_nonspecific_sigs_in_clust2 = np.nan

    ### Determine whether there is a distinct cluster.
    # Cluster 1 is considered distinct, if cluster 1 specific signatures contribute less than 0.05 in cluster 2 and more than 0.3 in cluster 1.
    if ((mean_fraction_clust1_sigs_in_clust2 < 0.05 and mean_fraction_clust1_sigs_in_clust1 > 0.3) and
        (mean_fraction_clust2_sigs_in_clust1 < 0.05 and mean_fraction_clust2_sigs_in_clust2 > 0.3)):
        Xs = [X[:, clust1_index], X[:, clust2_index]]
        clusters = [clust1_index, clust2_index]
        k = 2
        distinct = [True, True]
    elif (mean_fraction_clust1_sigs_in_clust2 < 0.05 and mean_fraction_clust1_sigs_in_clust1 > 0.3):
        Xs = [X[:, clust1_index], X[:, clust2_index]]
        clusters = [clust1_index, clust2_index]
        k = 2
        distinct = [True, False]
    elif (mean_fraction_clust2_sigs_in_clust1 < 0.05 and mean_fraction_clust2_sigs_in_clust2 > 0.3):
        Xs = [X[:, clust1_index], X[:, clust2_index]]
        clusters = [clust1_index, clust2_index]
        k = 2
        distinct = [False, True]
    else:
        Xs = [X]
        clusters = [np.arange(0, n_samples)]
        k = 1
        distinct = None

    return k, clusters, Xs, distinct


def stratify_samples(X, H=None, sil_thresh=0.9,
                     max_k=20, nrefs=50, metric='cosine', linkage_method='average', ref_method='a'):
    """Stratify samples by clustering with automatic selection of cluster number.

    If H is provided, H will be used for clustering. Otherwise, X will be used.

    Parameters:
    ----------
    X : array-like of shape (n_features, n_samples)
        Input data matrix.

    H : array-like of shape (n_components, n_samples)
        Optional exposure matrix.

    sil_thresh : float, default 0.9
        Silhouette score threshold. After determining the optimal number of clusters (k) by OptimalK,
        if the optimal k is greater than 1, we check the per-cluster silhouette scores. If there is at least
        one cluster with a silhouette score > sil_thresh, we accept the clustering into k clusters. Otherwise,
        we reject the clustering, and take k = 1 instead. This helps distinguish distinct clusters such as
        MMRD vs. MMRP, versus not-so-distinct clusters such as APOBEC vs. non-APOBEC.

    max_k : int, default 20
        Maximum number of clusters to be tested.

    nrefs : int
        Used in OptimalK. Number of reference datasets to be simulated.

    Returns:
    ----------
    k : int
        The optimal number of clusters. If k == 1, then no stratification is done.

    clusters : list of numpy arrays
        The indices of samples in distinct clusters. For example, the exposures of distinct clusters
        can be obtained by [H[:, indices] for indices in clusters]

    Xs : list of numpy arrays
        Samples in distinct clusters.

    optimalK : object of OptimalK class
        Contains information about the clustering. For example, use optimalK.plot() to visualize the selection curves.

    Notes:
    ----------
    1. Currently, sil_thresh is used in the way that if there is at least one cluster with silhouette score > sil_thresh, we
        take the entire clustering returned by OptimalK. One may argue that we need k - 1 clusters with silhouette score > sil_thresh,
        i.e., at most 1 cluster with silhouette score < sil_thresh, to accept the full clustering. Otherwise a smaller k might be
        more appropriate. This is a bit more complicated. So we ignore this for now.
    """
    if H is None:
        data = normalize(X, norm='l1', axis=0)
    else:
        data = normalize(H, norm='l1', axis=0)
    n_samples = data.shape[1]
    # Clustering with automatic selection of cluster number
    optimalK = OptimalK(data, max_k=max_k, nrefs=nrefs, metric=metric, linkage_method=linkage_method, ref_method=ref_method)
    # Gather results
    k = optimalK.k # Number of clusters
    if k > 1:
        # If k > 1, we check per-cluster silhouette scores.
        # If at least one cluster has silhouette score > sil_thresh, we accept the clustering.
        # Otherwise we reject it. 
        if np.any(optimalK.silscorek_percluster[k] > sil_thresh):
            cluster_membership = optimalK.cluster_membership
            clusters = []
            Xs = []
            for i in sorted(list(set(cluster_membership))):
                indices = np.arange(0, n_samples)[cluster_membership == i]
                clusters.append(indices)
                Xs.append(X[:, indices])
        else:
            k = 1
            clusters = [np.arange(0, n_samples)]
            Xs = [X]
    else:
        k = 1
        clusters = [np.arange(0, n_samples)]
        Xs = [X]
    return k, clusters, Xs, optimalK
