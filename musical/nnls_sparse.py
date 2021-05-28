"""Main class for the refitting problem"""

import numpy as np
import pandas as pd
import scipy as sp
from .nnls import nnls
from sklearn.metrics import pairwise_distances
import scipy.stats as stats


#def _llh_multinomial(x, p, epsilon=0.00001):
def _llh_multinomial(x, p, epsilon=1e-16):
    """Log likelihood of multinomial distribution: logP(x|p).

    Parameters
    ----------
    x : 1-d numpy array
        Observations (counts)
    p : 1-d numpy array
        Event probabilities. p should be summed to 1.

    Notes
    ----------
    1. Result is sum_i x_i log(p_i), ignoring the constant factor independent
        of p.
    2. Scipy.stats.multinomial.logpmf does not work, because it will give -inf
        for cases where there are zeros in p.
    """
    p = p.astype(float)  # In case p is of type int
    p[p == 0] = epsilon  # Consider p = p.clip(epsilon)
    return np.sum(x*np.log(p))

def _lh_multinomial(x, ps, offset=True, normalize=True):
    """Multinomial likelihood for a fixed x using a set of p's.

    Parameters
    ----------
    x : 1-d numpy array
        Observations (counts)
    ps : 2-d numpy array
        A set of event probabilities. Each row of ps is p. Each p should be
        summed to 1.
    offset : boolean
        If offset, subtract the log likelihoods by the maximum one.
    normalize : boolean
        If normalize, divide by sum of likelihoods.

    Notes
    ----------
    1. The resulted likelihoods are good for calculating likelihood ratios of
    multonimial models with different p's, since, 1) the constant factor
    independent of p is ignored in _llh_multinomial, and 2) a constant is
    subtracted from log likelihood.
    """
    llhs = np.array([_llh_multinomial(x, p) for p in ps])
    #llhs = np.array([stats.multinomial.logpmf(x, int(np.sum(x)), p) for p in ps])
    if offset:
        lhs = np.exp(llhs - np.max(llhs))
    else:
        lhs = np.exp(llhs)
    if normalize:
        lhs = lhs/np.sum(lhs)
    return lhs

def _redefine_frac_thresh(frac_thresh, h_fracs, frac_diff=0.05):
    """Rescue cases where no signatures pass frac_thresh.

    Parameters
    ----------
    frac_thresh : float
    h_fracs : 1-d numpy array
        Signature exposure fractions for a single sample.
    frac_diff : float
        Default 0.05

    Notes
    ----------
    Six examples: frac_thresh = 0.2 originally.
    Example 1: h_fracs = [0.41, 0.39, 0.1, 0.1]
        Output will be 0.2
    Example 2: h_fracs = [0.41, 0.29, 0.1, 0.1, 0.1]
        Output will be 0.2
    Example 3: h_fracs = [0.22, 0.18, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]                          i  _
          Output will be 0.17
    Example 4: h_fracs = [0.22, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06]
        Output will be 0.2
    Example 5: h_fracs = [0.19, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06]
        Output will be 0.14
    Example 6: h_fracs = [0.19, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        Output will be 0.14
    """
    frac_max = np.max(h_fracs)
    if frac_max > frac_thresh:
        frac_max_index = np.argmax(h_fracs)
        frac_second_max = np.max(np.delete(h_fracs, frac_max_index))
        if frac_max - frac_second_max < frac_diff:
            frac_thresh = min(frac_thresh, frac_max - frac_diff)
    else:
        frac_thresh = frac_max - frac_diff

    return frac_thresh


def _nnls_sparse_delta(x, h, W, delta=0.):
    """Auxiliary function to perform sparse NNLS with a simple cutoff.

    Parameters
    ----------
    x : 1-d numpy array, shape (n_features,)
        Count vector of a single sample
    w : 1-d numpy array, shape (n_components,)
        Initial exposure vector of a single sample
    H : 2-d numpy array, shape (n_components, n_features)
        Signature matrix
    delta : float
        Threshold. Signatures with exposure fraction <= delta in the original
        w will be set to 0 exposure.
    """
    n = np.sum(x)
    h_fracs = h/n
    inds_zero = np.where(h_fracs <= delta)[0]
    inds_sig = np.where(h_fracs > delta)[0]
    if len(inds_zero) > 0:
        h[inds_zero] = 0.
    h[inds_sig] = sp.optimize.nnls(W[:, inds_sig], x)[0]
    h_fracs = h/n
    return h, h_fracs, inds_zero, inds_sig

def nnls_sparse(x, W, method='llh',
                frac_thresh_base=0.02, frac_thresh_keep=0.4,
                frac_thresh=0.05, llh_thresh=0.65, exp_thresh=8.):

    n_components = W.shape[1]
    n = np.sum(x)
    h, _ = sp.optimize.nnls(W, x)
    if method in ['llh', 'cut', 'llh_stepwise']:
        h, h_fracs, inds_zero, inds_sig = _nnls_sparse_delta(x, h, W, delta = frac_thresh_base)

    inds_all = np.arange(0, n_components)


    if method == 'llh':
        x_nnls = np.matmul(W, h)
        if len(inds_sig) > 1:
            lhs_nonzero = []
            for ind in inds_sig:
                x_nnls2 = np.matmul(W[:, inds_sig[inds_sig != ind]],
                                    sp.optimize.nnls(W[:, inds_sig[inds_sig != ind]], x)[0])
                ps = np.array([x_nnls/np.sum(x_nnls),
                               x_nnls2/np.sum(x_nnls2)])
                lhs = _lh_multinomial(x, ps)
                lhs_nonzero.append(lhs[0])

            lhs_nonzero = np.array(lhs_nonzero)
            # Select final set of nonzero signatures
            inds_sig_tmp = []
            for ind, lh, frac in zip(inds_sig, lhs_nonzero,
                                     h_fracs[inds_sig]):
                if lh > llh_thresh or frac > frac_thresh_keep:
                    inds_sig_tmp.append(ind)
            inds_sig = np.sort(np.array(inds_sig_tmp))
            inds_zero = np.array([i for i in inds_all
                                  if i not in inds_sig])

        # Final NNLS
        if len(inds_zero) > 0:
            h[inds_zero] = 0.
        h[inds_sig] = sp.optimize.nnls(W[:, inds_sig], x)[0]
        h_fracs = h/n

    elif method == 'cut':
        sig_maxs = np.max(W, axis = 0)
        frac_thresh = _redefine_frac_thresh(frac_thresh, h_fracs)
        h, h_fracs, inds_zero, inds_sig = _nnls_sparse_delta(
            x, h, W, frac_thresh)
        # Loop until all the cuts are satisfied by all signatures
        while (
            np.sum(
                np.logical_or(
                    h_fracs[inds_sig] <= frac_thresh,
                    h[inds_sig] <= exp_thresh/sig_maxs[inds_sig]
                    )
            ) > 0
        ):
            frac_thresh = _redefine_frac_thresh(frac_thresh, h_fracs)
            inds_sig = inds_sig[
                np.logical_and(
                    h_fracs[inds_sig] > frac_thresh,
                    h[inds_sig] > exp_thresh/sig_maxs[inds_sig]
                )
            ]
            inds_zero = np.array([i for i in inds_all
                                  if i not in inds_sig])
            if len(inds_zero) > 0:
                h[inds_zero] = 0.
            h[inds_sig] = sp.optimize.nnls(W[:, inds_sig], x)[0]
            h_fracs = h/n

    elif method == 'llh_stepwise':
        x_nnls = W @ h
        inds_current = np.copy(inds_sig)
        if len(inds_sig) > 1:
            while len(inds_current) > 1:
                likelihood_ratios = []
                xs = []
                for ind in inds_current:
                    x_nnls2 = W[:, inds_current[inds_current != ind]] @ sp.optimize.nnls(W[:, inds_current[inds_current != ind]], x)[0]
                    xs.append(x_nnls2)
                    ps = np.array([x_nnls/np.sum(x_nnls), x_nnls2/np.sum(x_nnls2)])
                    lhs = _lh_multinomial(x, ps)
                    likelihood_ratios.append(lhs[0])
                #print(inds_current, likelihood_ratios)
                if np.min(likelihood_ratios) > llh_thresh:
                    break
                else:
                    index_remove = inds_current[np.argmin(likelihood_ratios)]
                    inds_current = np.copy(inds_current[inds_current != index_remove])
                    x_nnls = xs[np.argmin(likelihood_ratios)]

            inds_sig = np.copy(inds_current)
            inds_zero = np.array([i for i in inds_all if i not in inds_sig])
        if len(inds_zero) > 0:
            h[inds_zero] = 0.
        h[inds_sig] = sp.optimize.nnls(W[:, inds_sig], x)[0]
        h_fracs = h/n

    elif method == 'cut_naive':
        h, h_fracs, inds_zero, inds_sig = _nnls_sparse_delta(x, h, W, delta = frac_thresh)

    return h
