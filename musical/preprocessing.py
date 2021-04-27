import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize
import scipy.stats as stats

from .cluster import OptimalK


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

def remove_distinct_cluster(H, X, frac_thresh = 0.05):
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)

    for h, i in zip(H, range(0, H.shape[1])):
        h_norm = h/np.sum(X, axis = 0)
        h_norm[h_norm < frac_thresh] = 0
        H[i,:] = h_norm

    d = sp.spatial.distance.pdist(H.T, metric='cosine')
    d.clip(0)
    d_square_form = sp.spatial.distance.squareform(d)
    linkage = sch.linkage(d, method = 'average')
    cluster_membership = np.array(sch.fcluster(linkage, 2, criterion = "maxclust"))
    n_clust1 = sum(cluster_membership == 1)
    n_clust2 = sum(cluster_membership == 2)
    pvalue_clust1_greater = []
    pvalue_clust2_greater = []
    n_clust1_pos = []
    n_clust2_pos = []

    for i in range(0, H.shape[0]):
        n_clust1_pos.append(sum(np.array(H[i, cluster_membership == 1]) > 0))
        n_clust2_pos.append(sum(np.array(H[i, cluster_membership == 2]) > 0))
        x = H[i, np.where(cluster_membership == 1)]
        y = H[i, np.where(cluster_membership == 2)]
        pvalue_clust1_greater.append(stats.mannwhitneyu(x[0], y[0], alternative = 'greater')[1])
        pvalue_clust2_greater.append(stats.mannwhitneyu(y[0], x[0], alternative = 'greater')[1])

    n_clust1_pos = np.array(n_clust1_pos)
    n_clust2_pos = np.array(n_clust2_pos)
    pvalue_clust1_greater = np.array(pvalue_clust1_greater)
    pvalue_clust2_greater = np.array(pvalue_clust2_greater)

    indices_clust1 = np.logical_and(pvalue_clust1_greater < 0.05,n_clust2_pos/n_clust2 < 0.1)
    indices_clust1 = np.logical_or(indices_clust1, np.logical_and(n_clust1_pos/n_clust1 > 0.9, n_clust2_pos/n_clust2 < 0.1))
    indices_clust2 = np.logical_and(pvalue_clust2_greater < 0.05,n_clust1_pos/n_clust1 < 0.1)
    indices_clust2 = np.logical_or(indices_clust2, np.logical_and(n_clust2_pos/n_clust2 > 0.9, n_clust1_pos/n_clust1 < 0.1))
    indices_both = np.logical_and(n_clust1_pos/n_clust1 > 0.1, n_clust2_pos/n_clust2 > 0.1)

    mean_fraction_clust1_sigs_in_clust2 = 0
    mean_fraction_clust1_sigs_in_clust1 = 0
    mean_fraction_clust2_sigs_in_clust1 = 0
    mean_fraction_clust2_sigs_in_clust2 = 0
    mean_fraction_both_in_clust1 = 0
    mean_fraction_both_in_clust2 = 0

    if sum(indices_clust1) > 0:
        mean_fraction_clust1_sigs_in_clust1 = np.mean(np.sum(H[np.where(indices_clust1),:], axis = 1)[:,np.where(cluster_membership == 1)])
        mean_fraction_clust1_sigs_in_clust2 = np.mean(np.sum(H[np.where(indices_clust1),:], axis = 1)[:,np.where(cluster_membership == 2)])
    if sum(indices_clust2) > 0:
        mean_fraction_clust2_sigs_in_clust1 = np.mean(np.sum(H[np.where(indices_clust2),:], axis = 1)[:,np.where(cluster_membership == 1)])
        mean_fraction_clust2_sigs_in_clust2 = np.mean(np.sum(H[np.where(indices_clust2),:], axis = 1)[:,np.where(cluster_membership == 2)])
    if sum(indices_both) > 0:
        mean_fraction_both_in_clust1 = np.mean(np.sum(H[np.where(indices_both),:], axis = 1)[:, np.where(cluster_membership == 1)])
        mean_fraction_both_in_clust2 = np.mean(np.sum(H[np.where(indices_both),:], axis = 1)[:, np.where(cluster_membership == 2)])

    X_run_separate = {}
    ind = 0

    if np.logical_and(mean_fraction_clust1_sigs_in_clust2 < 0.05, mean_fraction_clust1_sigs_in_clust1 > 0.3):
        X_run_separate[ind] = X[:, cluster_membership == 2]
        ind = ind + 1

    if np.logical_and(mean_fraction_clust2_sigs_in_clust1 < 0.05, mean_fraction_clust2_sigs_in_clust2 > 0.3):
        X_run_separate[ind] = X[:, cluster_membership == 1]
        ind = ind + 1

    if np.logical_or(mean_fraction_clust1_sigs_in_clust2 > 0.05, mean_fraction_clust1_sigs_in_clust1 > 0.05):
        X_run_separate[ind] = X
        ind = ind + 1
    elif np.logical_and(mean_fraction_both_in_clust1 > 0.05, mean_fraction_both_in_clust2 > 0.05):
        X_run_separate[ind] = X
        ind = ind + 1

    return(X_run_separate)


def stratify_samples(X, H=None,
                     max_k=20, nrefs=50, metric='cosine', linkage_method='average'):
    """Stratify samples by clustering with automatic selection of cluster number.

    If H is provided, H will be used for clustering. Otherwise, X will be used.

    Parameters:
    ----------
    X : array-like of shape (n_features, n_samples)
        Input data matrix.

    H : array-like of shape (n_components, n_samples)
        Optional exposure matrix.

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
    """
    if H is None:
        data = X
    else:
        data = H
    n_samples = data.shape[1]
    # Clustering with automatic selection of cluster number
    optimalK = OptimalK(data, max_k=max_k, nrefs=nrefs, metric=metric, linkage_method=linkage_method)
    # Gather results
    k = optimalK.k # Number of clusters
    cluster_membership = optimalK.cluster_membership
    clusters = []
    Xs = []
    for i in sorted(list(set(cluster_membership))):
        indices = np.arange(0, n_samples)[cluster_membership == i]
        clusters.append(indices)
        Xs.append(X[:, indices])
    return k, clusters, Xs, optimalK
