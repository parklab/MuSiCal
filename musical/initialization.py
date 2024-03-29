"""Initialization for NMF"""

import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
import sklearn.decomposition._nmf as sknmf
import warnings

from .nnls import nnls


def _spa(M, r):
    """Python version of SPA algorithm

    TODO
    ----------
    1. Currently this code is following Algorithm 1 in 2013_Gillis_Fast and Robust Recursive Algorithms for Separable Nonnegative Matrix Factorization.pdf.
        The matlab code is more complete in several senses, e.g., there is an early stopping criterion.
        We need to read the matlab code more carefully and see if we need to implement those too. Basically
        we need to see if we should write the python code so that it is more similar to the matlab code.
        I think in most cases the current python code should work just fine. I've done several tests showing that
        the current python code produces equivalent results to the matlab code.
    """
    m, n = M.shape
    # Normalize so that columns sum to 1
    M = normalize(M, norm='l1', axis=0)
    R = np.copy(M)
    J = []
    I = np.eye(m)
    for j in range(1, r + 1):
        normR = np.sum(R**2, 0)
        jstar = np.argmax(normR)
        uj = np.reshape(R[:, jstar], (-1, 1)) # column vector
        R = (I - (uj @ uj.T)/np.sum(uj**2)) @ R
        J.append(jstar)
    return np.array(J)


def _init_spa(X, n_components, normalize_X=True, normalize_W=True):
    n_components = int(n_components)
    if normalize_X:
        X = normalize(X, norm='l1', axis=0) # This is redundant normalization.
    K = _spa(X, n_components)
    W = X[:, K]
    if normalize_W:
        W = normalize(W, norm='l1', axis=0)
    H = nnls(X, W)
    return W, H, K


def _init_cluster(X, n_components, metric='cosine', linkage='average',
                  max_ncluster=100, min_nsample=1):
    n_features, n_samples = X.shape
    XT_norm = normalize(X, norm='l1', axis=0).T
    d = sp.spatial.distance.pdist(XT_norm, metric=metric)
    d = d.clip(0)
    linkage = sch.linkage(d, method=linkage)
    for ncluster in range(n_components, np.min([n_samples, max_ncluster]) + 1):
        cluster_membership = sch.fcluster(linkage, ncluster, criterion='maxclust')
        if len(set(cluster_membership)) != ncluster:
            cluster_membership = sch.cut_tree(linkage, n_clusters=ncluster).flatten() + 1
            if len(set(cluster_membership)) != ncluster:
                warnings.warn('Number of clusters output by cut_tree or fcluster is not equal to the specified number of clusters',
                              UserWarning)
        W = []
        for i in range(1, ncluster + 1):
            if np.sum(cluster_membership == i) >= min_nsample:
                W.append(np.mean(XT_norm[cluster_membership == i, :], 0))
        W = np.array(W).T
        if W.shape[1] == n_components:
            break
    if W.shape[1] != n_components:
        raise RuntimeError('Initialization with init=cluster failed.')
    W = normalize(W, norm='l1', axis=0)
    H = nnls(X, W)
    return W, H, cluster_membership



def initialize_nmf(X, n_components, init='random', init_normalize_W=None,
                   init_refit_H=None,
                   init_cluster_metric='cosine',
                   init_cluster_linkage='average',
                   init_cluster_max_ncluster=100, init_cluster_min_nsample=1,
                   init_W_custom=None, init_H_custom=None):
    """Algorithms for NMF initialization

    Parameters
    ----------
    X : array-like of shape (n_features, n_samples)
        Constant input matrix.

    n_components : int
        The number of components (i.e., signatures).

    init : str
        Algorithm for the initialization.
        Valid options:

        - 'random': fill in the matrices with random numbers uniformly distributed
            between 0 and 1.

        - 'nndsvd':

        - 'nndsvda':

        - 'nndsvdar':

        - 'cluster':

        - 'spa':

        - 'custom':

    init_normalize_W : bool
        Whether or not to L1 normalize each signature.

    init_refit_H : bool
        Whether or not to refit H using NNLS. If True, H will be refitted after possible W normalization.

    """
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    n_features, n_samples = X.shape
    ################################################
    ############## Check parameters ################
    ################################################
    # This is necassary when n_components is of type numpy.int64, which is not supported by matlab.
    n_components = int(n_components)
    if init not in ['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'cluster', 'spa', 'custom']:
        raise ValueError('Invalid init parameter.')

    if init_normalize_W is None:
        if init in ['cluster', 'spa']:
            init_normalize_W = True
        else:
            init_normalize_W = False
    elif type(init_normalize_W) is bool:
        if init_normalize_W and init == 'custom':
            warnings.warn('init_normalize_W is set to True and init is custom. This might change init_W_custom.',
                          UserWarning)
        if (not init_normalize_W) and init in ['spa']:
            warnings.warn('init is %r but init_normalize_W is set to False. Normalizing W is recommended.' % init,
                          UserWarning)
        if (not init_normalize_W) and init == 'cluster':
            warnings.warn('init is cluster and init_normalize_W is set to False. W will be normalized regardless of init_normalize_W when init is cluster.',
                          UserWarning)
            init_normalize_W = True
    else:
        raise TypeError('Invalid parameter type for init_normalize_W.')

    if init_refit_H is None:
        if init in ['cluster', 'spa']:
            init_refit_H = True
        else:
            init_refit_H = False
    elif type(init_refit_H) is bool:
        if init_refit_H and init == 'custom':
            warnings.warn('init_refit_H is set to True and init is custom. This might change init_H_custom.',
                          UserWarning)
        if init_refit_H and init in ['random', 'nndsvd', 'nndsvda', 'nndsvdar']:
            warnings.warn('init_refit_H is set to True and init is %r. This might not be intended.' % init,
                          UserWarning)
        if (not init_refit_H) and init in ['cluster', 'spa']:
            warnings.warn('init_refit_H is set to False and init is %r. H will be calculated by NNLS refitting anyway.' % init,
                          UserWarning)
            init_refit_H = True
    else:
        raise TypeError('Invalid parameter type for init_refit_H.')

    if init == 'custom':
        if init_W_custom is None or init_H_custom is None:
            raise ValueError('init is set to custom. Must provide init_W_custom and init_H_custom.')
        else:
            if (type(init_W_custom) != np.ndarray) or (not np.issubdtype(init_W_custom.dtype, np.floating)):
                init_W_custom = np.array(init_W_custom).astype(float)
            if (type(init_H_custom) != np.ndarray) or (not np.issubdtype(init_H_custom.dtype, np.floating)):
                init_H_custom = np.array(init_H_custom).astype(float)
            if init_W_custom.shape != (n_features, n_components):
                raise ValueError('init_W_custom is of wrong shape.')
            if init_H_custom.shape != (n_components, n_samples):
                raise ValueError('init_H_custom is of wrong shape.')

    ######################################################
    ##################### Algorithms #####################
    ######################################################

    if init == 'random':
        W = np.random.uniform(0, 1, size=(n_features, n_components))
        H = np.random.uniform(0, 1, size=(n_components, n_samples))
        if init_normalize_W:
            W = normalize(W, norm='l1', axis=0)
        if init_refit_H:
            H = nnls(X, W)
        return W, H
    elif init == 'nndsvd' or init == 'nndsvda' or init == 'nndsvdar':
        W, H = sknmf._initialize_nmf(X, n_components, init=init)
        if init_normalize_W:
            W = normalize(W, norm='l1', axis=0)
        if init_refit_H:
            H = nnls(X, W)
        return W, H
    elif init == 'cluster':
        W, H, _ = _init_cluster(X, n_components, metric=init_cluster_metric,
                                linkage=init_cluster_linkage,
                                max_ncluster=init_cluster_max_ncluster,
                                min_nsample=init_cluster_min_nsample)
        return W, H
    elif init == 'spa':
        W, H, _ = _init_spa(X, n_components, normalize_X=True,
                            normalize_W=init_normalize_W)
        return W, H
    elif init == 'custom':
        W = init_W_custom
        H = init_H_custom
        if init_normalize_W:
            W = normalize(W, norm='l1', axis=0)
        if init_refit_H:
            H = nnls(X, W)
        return W, H
