import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize
import scipy.stats as stats

from .cluster import OptimalK, hierarchical_cluster


def sort_with_indices(x):
	"""
	x has to be a numpy array
	"""
	indices_sorted = np.argsort(x)

	return x[indices_sorted], indices_sorted


def gini(x, input_sorted=True):
	"""
	Calculate the Gini coefficient of x. If x is not sorted (increasing), set input_sorted=False
	"""
	if not input_sorted:
		x = np.sort(x)
	
	n = len(x)
	aux = x * (2 * np.arange(n) - n + 1)
	scaling = 1 / (n * np.sum(x))

	return scaling * np.sum(aux)


def n_remove_gini(x, gini_delta, thresh):
	"""
	Identify how many of the largest values of x significantly contribute to its Gini coefficient.
	Do not identify more than (1 - thresh) * 100 % of x.
	x is assumed to be increasing.
	"""
	n_remove = 0
	max_n_remove = np.round((1 - thresh) * len(x))

	gini_old = gini(x)
	gini_new = gini(x[:-1])
	n_remove = 0

	while gini_old - gini_new > gini_delta and n_remove < max_n_remove:

		n_remove += 1
		gini_old = gini_new
		gini_new = gini(x[: - n_remove - 1])

	return n_remove


def remove_samples_based_on_gini(H, X, gini_baseline=.65, gini_delta=.005):
	"""
	Identify signatures with unequal exposures. A signature is said to have unequal exposures if the
	Gini coefficient of the sample exposures is higher than a given threshold.
	For these signatures, the samples causing the gini coefficient to be high are also identified.

	Input:
	------
	H: np.ndarray
		The exposure matrix of shape (n_signatures, n_samples)

	X: np.ndarray
		The mutation count matrix of shape (n_features, n_samples)

	gini_baseline: float
		Signatures with exposures having a higher Gini coefficient than 'gini_baseline' are identified
		as having unequal exposures

	gini_delta: float
		Per signature with unequal exposure, a sample is identified as a sample significanlty contributing
		the high Gini coefficient if removing it decreases the Gini coefficient by at least 'gini_delta'

	Output:
	------
	samples_to_keep: dict
		keys: indices of signatures with unequal exposures
		values: corresponding sample indices that do not (!) cause the Gini coefficient to be high

	X_to_keep: dict
		keys: indices of signatures with unequal exposures
		values: mutation count matix subsetted to the samples that do not (!) cause the Gini coefficient to be high

	samples_to_keep_all: np.ndarray
		List of sample indices not significantly causing the Gini coefficient of any signature with unequal exposure to be high
	"""
	H, X = np.array(H), np.array(X)

	n_samples = H.shape[1]

	# normalize the exposures
	H = H / np.sum(X, axis=0)

	# Gini coefficients of normalized signature exposures
	gini_coeffs = np.array([gini(np.sort(h)) for h in H])
	sigs_to_check = np.where(gini_coeffs > gini_baseline)[0]

	samples_to_keep = {}
	samples_to_remove = set()

	for sig_index in sigs_to_check:

		sorted_h, sorted_h_indices = sort_with_indices(H[sig_index,:])
		n_remove = n_remove_gini(sorted_h, gini_delta, .8)

		to_keep, to_remove = np.split(sorted_h_indices, [-n_remove]) if n_remove else (sorted_h_indices, np.empty(0))
		samples_to_keep[sig_index] = np.sort(to_keep)
		samples_to_remove |= set(to_remove)

	X_to_keep = {sig_index: X[:, samples] for sig_index, samples in samples_to_keep.items()}

	samples_to_keep_all = set(range(n_samples)) - samples_to_remove
	samples_to_keep_all = np.sort(list(samples_to_keep_all))

	results = (samples_to_keep, X_to_keep, samples_to_keep_all)

	return results


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