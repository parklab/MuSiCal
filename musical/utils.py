"""Utilities"""

import numpy as np
import sklearn.decomposition._nmf as sknmf
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import itertools
import scipy as sp
import scipy.stats as stats
from operator import itemgetter
from sklearn.preprocessing import normalize

from .nnls_sparse import nnls_sparse
from .nnls_sparse2 import SparseNNLS

##################
# Useful globals #
##################

trinucleotides_C = ["ACA",
                    "ACC",
                    "ACG",
                    "ACT",
                    "CCA",
                    "CCC",
                    "CCG",
                    "CCT",
                    "GCA",
                    "GCC",
                    "GCG",
                    "GCT",
                    "TCA",
                    "TCC",
                    "TCG",
                    "TCT"]

trinucleotides_T = ["ATA",
                    "ATC",
                    "ATG",
                    "ATT",
                    "CTA",
                    "CTC",
                    "CTG",
                    "CTT",
                    "GTA",
                    "GTC",
                    "GTG",
                    "GTT",
                    "TTA",
                    "TTC",
                    "TTG",
                    "TTT"]

snv_types_6_str = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]

snv_types_6_set = [{"C", "A"}, {"C", "G"}, {"C", "T"},
                   {"T", "A"}, {"T", "C"}, {"T", "G"}]

snv_types_96_str = (["C>A" + ":" + item for item in trinucleotides_C] +
                    ["C>G" + ":" + item for item in trinucleotides_C] +
                    ["C>T" + ":" + item for item in trinucleotides_C] +
                    ["T>A" + ":" + item for item in trinucleotides_T] +
                    ["T>C" + ":" + item for item in trinucleotides_T] +
                    ["T>G" + ":" + item for item in trinucleotides_T])

snv_types_96_list = ([["C>A", item] for item in trinucleotides_C] +
                     [["C>G", item] for item in trinucleotides_C] +
                     [["C>T", item] for item in trinucleotides_C] +
                     [["T>A", item] for item in trinucleotides_T] +
                     [["T>C", item] for item in trinucleotides_T] +
                     [["T>G", item] for item in trinucleotides_T])

# Need to update
indel_types_83_str = [
 'DEL.C.1.1',
 'DEL.C.1.2',
 'DEL.C.1.3',
 'DEL.C.1.4',
 'DEL.C.1.5',
 'DEL.C.1.6+',
 'DEL.T.1.1',
 'DEL.T.1.2',
 'DEL.T.1.3',
 'DEL.T.1.4',
 'DEL.T.1.5',
 'DEL.T.1.6+',
 'INS.C.1.0',
 'INS.C.1.1',
 'INS.C.1.2',
 'INS.C.1.3',
 'INS.C.1.4',
 'INS.C.1.5+',
 'INS.T.1.0',
 'INS.T.1.1',
 'INS.T.1.2',
 'INS.T.1.3',
 'INS.T.1.4',
 'INS.T.1.5+',
 'DEL.repeats.2.1',
 'DEL.repeats.2.2',
 'DEL.repeats.2.3',
 'DEL.repeats.2.4',
 'DEL.repeats.2.5',
 'DEL.repeats.2.6+',
 'DEL.repeats.3.1',
 'DEL.repeats.3.2',
 'DEL.repeats.3.3',
 'DEL.repeats.3.4',
 'DEL.repeats.3.5',
 'DEL.repeats.3.6+',
 'DEL.repeats.4.1',
 'DEL.repeats.4.2',
 'DEL.repeats.4.3',
 'DEL.repeats.4.4',
 'DEL.repeats.4.5',
 'DEL.repeats.4.6+',
 'DEL.repeats.5+.1',
 'DEL.repeats.5+.2',
 'DEL.repeats.5+.3',
 'DEL.repeats.5+.4',
 'DEL.repeats.5+.5',
 'DEL.repeats.5+.6+',
 'INS.repeats.2.0',
 'INS.repeats.2.1',
 'INS.repeats.2.2',
 'INS.repeats.2.3',
 'INS.repeats.2.4',
 'INS.repeats.2.5+',
 'INS.repeats.3.0',
 'INS.repeats.3.1',
 'INS.repeats.3.2',
 'INS.repeats.3.3',
 'INS.repeats.3.4',
 'INS.repeats.3.5+',
 'INS.repeats.4.0',
 'INS.repeats.4.1',
 'INS.repeats.4.2',
 'INS.repeats.4.3',
 'INS.repeats.4.4',
 'INS.repeats.4.5+',
 'INS.repeats.5+.0',
 'INS.repeats.5+.1',
 'INS.repeats.5+.2',
 'INS.repeats.5+.3',
 'INS.repeats.5+.4',
 'INS.repeats.5+.5+',
 'DEL.MH.2.1',
 'DEL.MH.3.1',
 'DEL.MH.3.2',
 'DEL.MH.4.1',
 'DEL.MH.4.2',
 'DEL.MH.4.3',
 'DEL.MH.5+.1',
 'DEL.MH.5+.2',
 'DEL.MH.5+.3',
 'DEL.MH.5+.4',
 'DEL.MH.5+.5+'
]


#####################
# Utility functions #
#####################

def beta_divergence_deprecated(A, B, beta=1, square_root=False):
    """Beta_divergence

    A and B must be float arrays.

    DEPRECATED
    ----------
    sknmf._beta_divergence has problems when beta = 1. The problem seems to be in the following line:
    sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1)),
    together with the fact that it takes 3 separate matrix sums and then do the additions. That results
    in reduced accuracy in small differences, which is really what matters for the final result. Basically,
    KL divergence is \sum(Alog(A/B) - A + B). If you do \sum(Alog(A/B)) - \sum(A) + \sum(B), it'll be less
    accurate.
    Note that using the aux matrix as in this function seems better than using sknmf._beta_divergence alone.
    But we should not use this function anyway.
    """
    aux = np.diag(np.ones(B.shape[1])).astype(float)
    return sknmf._beta_divergence(A, B, aux, beta=beta, square_root=square_root)


def beta_divergence(A, B, beta=1, square_root=False):
    """Beta_divergence

    A and B must be float arrays.

    Notes
    ----------
    For KL divergence, we do not deal with the case where B_ij = 0 but A_ij != 0,
    i.e., we'll output inf if that happens.
    For the purpose of NMF, B_ij will never be 0, because within the NMF algorithm,
    we always clip small values. So 0's will be replaced by EPSILON in W and H. As
    a result, W @ H will never have 0 entries.
    """
    if beta == 1 or beta == "kullback-leibler":
        # When B_ij = 0, KL divergence is not defined unless A_ij = 0.
        # Whenever A_ij = 0, then the contribution of the term A_ij log(A_ij/B_ij)
        # is considered as 0.
        A_data = A.ravel()
        B_data = B.ravel()
        indices = A_data > 0
        A_data = A_data[indices]
        B_data_remaining = B_data[~indices]
        B_data = B_data[indices]
        # Here we must take matrix additions first and then take sum.
        # Otherwise, the separate matrix sums will be too big and the small
        # differences will be lost, and we'll get 0.0 results.
        res = np.sum(A_data*np.log(A_data/B_data) - A_data + B_data)
        res = res + np.sum(B_data_remaining)
    elif beta == 2 or beta == 'frobenius':
        res = np.linalg.norm(A - B, ord=None) # 2-norm for vectors and frobenius norm for matrices
        res = res**2 / 2
    else:
        raise ValueError('Only beta = 1 and beta = 2 are implemented.')
    if square_root:
        res = np.sqrt(2*res)

    return res


def normalize_WH(W, H):
    normalization_factor = np.sum(W, 0)
    return W/normalization_factor, H*normalization_factor[:, None]


def match_catalog_pair(W1, W2, metric='cosine'):
    """Match a pair of signature catalogs.

    Notes
    ----------
    1. Assume a 1-to-1 match between H1 and H2. This is an assignment problem
    (See https://en.wikipedia.org/wiki/Assignment_problem).
    2. W2 will be reordered to match with W1.
    """
    if W1.shape != W2.shape:
        raise ValueError('W1 and W2 must be of the same shape.')

    pdist = pairwise_distances(W1.T, W2.T, metric=metric)
    W2_reordered_indices = linear_sum_assignment(pdist)[1]
    return W2[:, W2_reordered_indices], W2_reordered_indices, pdist


def bootstrap_count_matrix(X):
    #n_features, n_samples = X.shape
    X_bootstrapped = []
    for x in X.T:
        N = int(round(np.sum(x)))
        p = np.ravel(x/np.sum(x))
        X_bootstrapped.append(np.random.multinomial(N, p))
        #indices = np.random.choice(n_features, size=N, replace=True, p=x/np.sum(x))
        #X_bootstrapped.append([np.sum(indices == i)
        #                       for i in range(0, n_features)])
    X_bootstrapped = np.array(X_bootstrapped).T
    return X_bootstrapped


def simulate_count_matrix(W, H, method='multinomial'):
    #n_features, n_components = W.shape
    #_, n_samples = H.shape

    # Just in case W and H are not properly normalized
    #W, H = normalize_WH(W, H)

    if method == 'multinomial':
        #X_simulated = []
        #for h in H.T:
        #    x = np.zeros(n_features, dtype=int)
        #    for i in range(0, n_components):
        #        N = int(round(h[i]))
        #        indices = np.random.choice(n_features, size=N, replace=True, p=W[:, i])
        #        x += np.array([np.sum(indices == j) for j in range(0, n_features)])
        #    X_simulated.append(x)
        #X_simulated = np.array(X_simulated).T
        X_simulated = bootstrap_count_matrix(W @ H)
    else:
        raise ValueError(
            'Invalid method parameter: got %r instead of one of %r' %
            (method, {'multinomial'}))

    return X_simulated


def _samplewise_error(X, X_reconstructed, beta=1, square_root=False):
    errors = []
    for x, x_reconstructed in zip(X.T, X_reconstructed.T):
        errors.append(beta_divergence(x, x_reconstructed, beta=beta, square_root=square_root))
    errors = np.array(errors)
    return errors


def match_signature_to_catalog(w, W_catalog, thresh=0.99, min_contribution = 0.1, include_top=True):
    """Match a single signature to possibly multiple signatures in the catalog.

    Parameters
    ----------
    w : array-like of shape (n_features,)
        The signature to be matched.

    W_catalog : array-like of shape (n_features, n_sigs).
        The signature catalog.

    thresh : float
        If cosine similarity to any signature in W_catalog is >= thresh, then w is matched to that signature.
        If cosine similarity to all signatures in W_catalog is < thresh, then doublets will be considered.
        If cosine similarity to any doublet is >= thresh, then w is matched to that doublet.
        If cosine similarity to all doublets is < thresh, then triplets will be considered.

    include_top : bool
        If true, then the top matched signature will always be included when doublets or triplets are considered.
        Otherwise, all possible doublets or triplets will be considered when applicable.


    Returns
    ----------
    match : tuple
        Index/indices of matched signature(s).

    cos : float
        Cosine similarity.

    coef : array-like or None
        When w is matched to doublets or triplets, coef is the NNLS coef for reconstructing w.
        When w is matched to a single signature, coef is None.
    """
    n_features, n_sigs = W_catalog.shape
    ###################################################
    ###################### Singles ####################
    ###################################################
    ### Calculate cosine similarity to each signature in the catalog
    data = []
    for i, w_catalog in zip(range(0, n_sigs), W_catalog.T):
        data.append([i, 1 - sp.spatial.distance.cosine(w, w_catalog)])
    data = sorted(data, key=itemgetter(1), reverse=True)
    match = (data[0][0],)
    cos = data[0][1]
    coef = None
    # If the top matched signature has cosine similarity >= thresh, then return this signature.
    if cos >= thresh:
        return match, cos, coef

    ###################################################
    ##################### Doublets ####################
    ###################################################
    ### Otherwise, consider doublets
    # First, construct the combinations of signatures to be tested.
    sigs = range(0, n_sigs) # Indices of all signatures in the catalog.
    top = data[0][0] # Index of the top matched signature.
    combs = [] # All signature combinations to be tested.
    if include_top:
        combs.extend([(top,)]) # First, include singlets
        sigs_notop = [sig for sig in sigs if sig != top] # Indices of all signatures excluding the top matched signature.
        combs.extend([(top, sig) for sig in sigs_notop])
    else:
        combs.extend([(sig,) for sig in sigs]) # First, include singlets
        combs.extend(list(itertools.combinations(sigs, 2)))
    # Then, perform NNLS on all combinations to be tested, and select the best combination based on residual error.
    data = []
    for item in combs:
        x, resid = sp.optimize.nnls(W_catalog[:, list(item)], w)
        if np.min(x) < min_contribution:
            continue
        data.append([item, x, resid])
    data = sorted(data, key=itemgetter(2))
    if len(data) > 0:
        match = data[0][0]
        coef = data[0][1]
        cos = 1 - sp.spatial.distance.cosine(w, np.dot(W_catalog[:, list(match)], coef))
        # If cosine similarity >= thresh, return the best doublet
        if cos >= thresh:
            return match, cos, coef

    ###################################################
    ##################### Triplets ####################
    ###################################################
    ### Otherwise, consider triplets
    #combs = []
    if include_top:
        sigs_notop = [sig for sig in sigs if sig != top] # Indices of all signatures excluding the top matched signature.
        combs.extend([(top,) + item for item in list(itertools.combinations(sigs_notop, 2))])
    else:
        combs.extend(list(itertools.combinations(sigs, 3)))
    data = []
    for item in combs:
        x, resid = sp.optimize.nnls(W_catalog[:, list(item)], w)
        if np.min(x) < min_contribution:
            continue
        data.append([item, x, resid])
    data = sorted(data, key=itemgetter(2))
    if len(data) > 0:
        match = data[0][0]
        coef = data[0][1]
        cos = 1 - sp.spatial.distance.cosine(w, np.dot(W_catalog[:, list(match)], coef))
        return match, cos, coef
    return (), np.nan, None


def match_signature_to_catalog_nnls_sparse(w, W_catalog, N=10000, method='llh_stepwise',
                                           frac_thresh_base=0.02, frac_thresh_keep=0.4,
                                           frac_thresh=0.05, llh_thresh=0.65, exp_thresh=8.):
    x = np.rint(w*N).astype(int)

    h = nnls_sparse(x, W_catalog, method=method,
                    frac_thresh_base=frac_thresh_base, frac_thresh_keep=frac_thresh_keep,
                    frac_thresh=frac_thresh, llh_thresh=llh_thresh, exp_thresh=exp_thresh)
    match = np.arange(0, W_catalog.shape[1])[h > 0]
    coef, _ = sp.optimize.nnls(W_catalog[:, match], w)
    cos = 1 - sp.spatial.distance.cosine(w, W_catalog[:, match] @ coef)
    return tuple(match), cos, coef

def match_signature_to_catalog_nnls_sparse2(w, W_catalog, method='likelihood_bidirectional',
                                            thresh1 = 0.001, thresh2 = None):
    
    sparse_method = SparseNNLS(method = method,
                               thresh1 = thresh1,
                               thresh2 = thresh2)
    sparse_method.fit(X= w, W = W_catalog)
    h = np.transpose(np.array(sparse_method.H))
    match = np.arange(0, W_catalog.shape[1])[np.where(h > 0)[1]]
    coef, _ = sp.optimize.nnls(W_catalog[:, match], w)
    cos = 1 - sp.spatial.distance.cosine(w, W_catalog[:, match] @ coef)
    return tuple(match), cos, coef


def tag_similar_signatures(W, metric = 'cosine'):
    pdist = pairwise_distances(W.T, metric = metric)
    n_signatures = W.shape[1]
    similar_signatures = []
    for i in  range(0, n_signatures):
        inds = np.where(pdist[i, :] < 0.05)
        similar_signatures[i] = inds
    return similar_signatures

def save_signature_exposure_tables(model):
    model.W


def differential_tail_test(a, b, percentile=90, alternative='two-sided'):
    """Test if distribution tails are different (pubmed: 18655712)

    Parameters
    ----------
    a, b : array-like
        a, b must be positive.

    percentile : float
        Percentile threshold above which data points are considered tails.

    alternative : {'two-sided', 'less', 'greater'}
        Defines the alternative hypothesis. For example, when set to 'greater',
        the alternative hypothesis is that the tail of a is greater than the tail
        of b.
    """
    a = np.array(a)
    b = np.array(b)
    if len(a) != len(b):
        warnings.warn('Lengths of a and b are different. The differential tail test could lose power.',
                      UserWarning)
    comb = np.concatenate([a, b])
    thresh = np.percentile(comb, percentile)
    za = a * (a > thresh)
    zb = b * (b > thresh)
    # If za and zb contain identical values, e.g., both za and zb are all zeros.
    #if len(za) == len(zb) and (np.sort(za) == np.sort(zb)).all():
    if len(set(np.concatenate((za, zb)))) == 1:
        if alternative == 'two-sided':
            return np.nan, 1.0
        else:
            return np.nan, 0.5
    statistic, pvalue = stats.mannwhitneyu(za, zb, alternative=alternative)
    return statistic, pvalue


def smallest_singular_value(X, norm=None, **kwargs):
    """Calculate the smallest singular value of a matrix.

    **kwargs are the arguments for sklearn.preprocessing.normalize().
    """
    if norm is None:
        u, s, vh = np.linalg.svd(X)
    else:
        u, s, vh = np.linalg.svd(normalize(X, norm=norm, **kwargs))
    return s[-1]


def parallelotope_volume(X):
    """Calculate the parallelotope volume.

    Parameters
    ----------
    X : array, shape (m, n)
        We consider rows of X to be edges of the parallelotope in n dimensional
        space.

        - If rank X != m, we return 0 with a warning that rows of X are not
            linear independent.

        - If rank X == m, and m == n, we use the parallelotope formed by rows
            of X directly.

        - If rank X == m, and m < n, we complete the parallelotope using
            orthonormal basis of the null space (kernel) of X.

    NOTE:
    This is copied from the old SigExplorer. If we want to use it on the signature
    matrix W, we should supply W.T.
    """
    m, n = X.shape
    r = np.linalg.matrix_rank(X)
    if r != m:
        warnings.warn("Rank (= %d) of the input matrix is not equal to the "
                      "row number (= %d). Thus rows of the input matrix are "
                      "not linear independent" % (r, m),
                      UserWarning)
        return 0
    else:
        if m == n:
            v = np.abs(np.linalg.det(normalize(X)))
        else:
            v = np.abs(
                np.linalg.det(
                    normalize(
                        np.concatenate(
                            (X, sp.linalg.null_space(X).T), axis=0
                        ), axis=1
                    )
                )
            )
    return v


def classification_statistics(confusion_matrix=None, P=None, PP=None, All=None):
    """
    Parameters:
    ----------
    confusion_matrix : np.ndarray
        Cannot be pd.DataFrame
    P : list
        Real positives
    PP : list
        Predicted positives
    All : list
        All labels, i.e., P + N
    """
    if confusion_matrix is None:
        if P is None or PP is None or All is None:
            raise ValueError('When confusion matrix is not provided, P, PP, and All must be provided.')
        P = set(P)
        PP = set(PP)
        All = set(All)
        ###
        N = All - P
        PN = All - PP
        TP = PP.intersection(P)
        TN = PN.intersection(N)
        ###
        nP = len(P)
        nN = len(N)
        nPP = len(PP)
        nPN = len(PN)
        nTP = len(TP)
        nFP = nPP - nTP
        nTN = len(TN)
        nFN = nPN - nTN
        confusion_matrix = np.array([[nTP, nFN], [nFP, nTN]])
    else:
        if P is not None or PP is not None or All is not None:
            warnings.warn('Confusion matrix is provided. The provided P, PP, or All are ignored.',
                          UserWarning)
        if confusion_matrix.shape != (2, 2):
            raise ValueError('Confusion matrix is not of the correct shape.')
        nTP = confusion_matrix[0, 0]
        nFN = confusion_matrix[0, 1]
        nFP = confusion_matrix[1, 0]
        nTN = confusion_matrix[1, 1]
        nP = nTP + nFN
        nN = nFP + nTN
        nPP = nTP + nFP
        nPN = nFN + nTN

    ######################################
    statistics = {
        "P": nP,
        "N": nN,
        "PP": nPP,
        "PN": nPN,
        "TP": nTP,
        "TN": nTN,
        "FP": nFP,
        "FN": nFN,
        "ConfusionMatrix": confusion_matrix
    }

    # Power = sensitivity = recall = true positive rate = TP/P = TP/(TP + FN)
    if nP == 0:
        statistics["Power"] = np.nan
        statistics["Sensitivity"] = np.nan
        statistics["Recall"] = np.nan
        statistics["TPR"] = np.nan
    else:
        statistics["Power"] = nTP/nP
        statistics["Sensitivity"] = nTP/nP
        statistics["Recall"] = nTP/nP
        statistics["TPR"] = nTP/nP
    # FDR = FP/(FP + TP)
    if nFP + nTP == 0:
        statistics["FDR"] = np.nan
    else:
        statistics["FDR"] = nFP/(nFP + nTP)
    # precision = TP/(TP + FP) = 1 - FDR
    statistics["Precision"] = 1 - statistics["FDR"]
    # False positive rate = FP/N = FP/(FP + TN)
    if nN == 0:
        statistics["FPR"] = np.nan
    else:
        statistics["FPR"] = nFP/nN
    # Specificity = true negative rate = selectivity = TN/N
    if nN == 0:
        statistics["Specificity"] = np.nan
        statistics["TNR"] = np.nan
    else:
        statistics["Specificity"] = nTN/nN
        statistics["TNR"] = nTN/nN
    # False negative rate = FN/P = 1 - TPR
    statistics["FNR"] = 1 - statistics['TPR']
    # Accuracy = (TP + TN)/(P + N)
    if nP + nN == 0:
        statistics["Accuracy"] = np.nan
    else:
        statistics["Accuracy"] = (nTP + nTN)/(nP + nN)
    # Balanced accuracy = (TPR + TNR)/2
    statistics['BAcc'] = (statistics['TPR'] + statistics['TNR'])/2
    # F1 score = 2 * precision * recall / (precision + recall)
    if statistics['Precision'] + statistics['Recall'] > 0:
        statistics['F1'] = 2 * statistics['Precision'] * statistics['Recall'] / (statistics['Precision'] + statistics['Recall'])
    else:
        statistics['F1'] = np.nan
    # Mathew's correlation coefficient
    # https://bmcgenomics.biomedcentral.com/track/pdf/10.1186/s12864-019-6413-7.pdf
    if np.sum(statistics['ConfusionMatrix'] == 0) == 4:
        statistics['MCC'] = np.nan
    elif np.sum(statistics['ConfusionMatrix'] == 0) == 3:
        if nTP != 0 or nTN != 0:
            statistics['MCC'] = 1
        elif nFP != 0 or nFN != 0:
            statistics['MCC'] = -1
        else: # Won't happen
            statistics['MCC'] = np.nan
    else:
        if (nTP + nFP)*(nTP + nFN)*(nTN + nFP)*(nTN + nFN) == 0:
            statistics['MCC'] = 0
        else:
            statistics['MCC'] = (nTP * nTN - nFP * nFN)/np.sqrt((nTP + nFP)*(nTP + nFN)*(nTN + nFP)*(nTN + nFN))
    # Normalized MCC
    statistics['nMCC'] = (statistics['MCC'] + 1)/2

    return statistics
