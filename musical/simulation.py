"""Routines for simulation studies"""

import numpy as np
import scipy as sp
import warnings
import pandas as pd

from .utils import simulate_count_matrix

def simulate_LDA(alpha, n_samples, n_mutations, n_components, n_features=96, W=None, beta=None,
                 samples=None, features=None, sigs=None, adapt_alpha=False):
    """Simulate a dataset based on the LDA generative model.
    """
    ##### Parameter validation
    if type(alpha) is float:
        alpha = np.ones(n_components) * alpha
    elif type(alpha) is np.ndarray:
        if len(alpha) != n_components:
            raise ValueError('When alpha is a np.ndarray, it should be of length n_components.')
        alpha = alpha.astype(float)
    else:
        raise ValueError('alpha must be a float or a numpy.ndarray.')

    if type(n_mutations) is int or type(n_mutations) is float:
        n_mutations = np.random.poisson(n_mutations, size=n_samples)
    elif type(n_mutations) is np.ndarray:
        n_mutations = n_mutations.astype(int)
        if len(n_mutations) != n_samples:
            raise ValueError('When n_mutations is a numpy.ndarray, its length must be n_samples.')
    else:
        raise ValueError('n_mutations must be a float, int, or a numpy.ndarray.')

    ##### Simulate signatures if it is not provided.
    if W is not None:
        if (type(W) != np.ndarray) or (not np.issubdtype(W.dtype, np.floating)):
            W = np.array(W).astype(float)
        if W.shape[1] != n_components:
            raise ValueError('The number of signatures in the provided W is different from n_components.')
        if W.shape[0] != n_features:
            raise ValueError('The number of features in the provided W is different from n_features.')
        if beta is not None:
            warnings.warn('Both W and beta are provided. W will be used and beta will be ignored.',
                          UserWarning)
    else:
        # For simulating signatures, we'll always assume symmetric Dirichlet distributions.
        if beta is None:
            beta = 10**np.random.uniform(np.log10(0.02), np.log10(5.0), n_components)
        elif type(beta) is float:
            beta = np.ones(n_components) * beta
        elif type(beta) is np.ndarray:
            if len(beta) != n_components:
                raise ValueError('When beta is a np.ndarray, it should be of length n_components.')
            beta = beta.astype(float)
        else:
            raise ValueError('beta must be None, float, or np.ndarray.')
        W = []
        for item in beta:
            W.append(np.random.dirichlet(np.ones(n_features)*item, 1).flatten())
        W = np.array(W)
        W = W.T

    ##### Adapt alpha to the signatures
    # The same concentration parameter will be kept (i.e., sum of alpha components)
    # But each component of alpha will be scaled according to the 1/std of the signatures
    if adapt_alpha:
        weight = 1/np.std(W, 0, ddof=1)
        weight = weight/np.sum(weight)
        alpha = np.sum(alpha) * weight

    ##### Simulate exposures
    H = np.random.dirichlet(alpha, n_samples).T
    # Match to n_mutations
    H = H * n_mutations

    ##### Simulate X
    X = simulate_count_matrix(W, H)

    ##### Get pd dataframes
    if samples is None:
        samples = ['Sample' + str(i) for i in range(1, n_samples+1)]
    if features is None:
        features = ['Feature' + str(i) for i in range(1, n_features+1)]
    if sigs is None:
        sigs = ['Sig' + str(i) for i in range(1, n_components+1)]
    X = pd.DataFrame(X, columns=samples, index=features)
    W = pd.DataFrame(W, columns=sigs, index=features)
    H = pd.DataFrame(H, columns=samples, index=sigs)

    return W, H, X
