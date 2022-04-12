"""Refitting and matching

TODO:
1. Change the option indices_associated_sigs=None to something like connected_sigs=False.
I.e., automatically set associated signatures. Or, at the same time allow user to specify their own
association list. If they do not specify, use our default list. This change should be done after we revise
nnls_sparse.py, where we replace indices_associated_sigs by names of associated sigs. That'll be easier to do.
"""


import numpy as np
import scipy as sp
import pandas as pd
import warnings

from .nnls import nnls
from .nnls_sparse import SparseNNLS, SparseNNLSGrid
from .utils import match_signature_to_catalog_nnls_sparse, beta_divergence, get_sig_indices_associated, SIGS_ASSOCIATED_DICT, SIGS_ASSOCIATED
from .catalog import load_catalog


def refit(X, W, method='likelihood_bidirectional', thresh=None,
          connected_sigs=False):
    """Wrapper around SparseNNLS for refitting

    Note that only one parameter thresh1 is allowed here.
    Both X and W should be pd.DataFrame.
    If connected_sigs is set to True, we'll not fill in missing connected sigs, although a warning will be printed.
    So make sure W contains all connected signatures if connected_sigs is set to True.
    """
    # Check input
    if X.shape[0] != W.shape[0]:
        raise ValueError('X and W have different numbers of channels.')
    if (X.index == W.index).sum() != X.shape[0]:
        raise ValueError('X and W have different indices.')
    # SparseNNLS
    if connected_sigs:
        indices_associated_sigs, _ = get_sig_indices_associated(W.columns.values, W.columns.values)
        # Give some informative warnings
        missing_sigs = []
        W_sigs = W.columns.values
        for key in W_sigs:
            if key in SIGS_ASSOCIATED_DICT.keys():
                for sig in SIGS_ASSOCIATED_DICT[key]:
                    if sig not in W_sigs:
                        missing_sigs.append(sig)
        if len(missing_sigs) > 0:
            warnings.warn(('In refit: connected_sigs is set to True. The input W contains signatures with connected signatures. ' +
                           'However, W is missing some connected signatures. Specifically, W is missing: ' +
                           ','.join(missing_sigs) + '. Please fill in these missing sigs in W or make sure this is indeed what is wanted.'),
                          UserWarning)
    else:
        indices_associated_sigs = None
    model = SparseNNLS(method=method, thresh1=thresh, indices_associated_sigs=indices_associated_sigs)
    model.fit(X, W)
    return model.H, model

def refit_grid(X, W, method='likelihood_bidirectional', thresh_grid=None, ncpu=1, verbose=0,
               connected_sigs=False):
    """Refitting on a grid of thresholds.
    """
    # Check input
    if X.shape[0] != W.shape[0]:
        raise ValueError('X and W have different numbers of channels.')
    if (X.index == W.index).sum() != X.shape[0]:
        raise ValueError('X and W have different indices.')
    # SparseNNLSGrid
    if thresh_grid is None:
        thresh_grid = np.array([0.001])
    # connected_sigs
    if connected_sigs:
        indices_associated_sigs, _ = get_sig_indices_associated(W.columns.values, W.columns.values)
        # Give some informative warnings
        missing_sigs = []
        W_sigs = W.columns.values
        for key in W_sigs:
            if key in SIGS_ASSOCIATED_DICT.keys():
                for sig in SIGS_ASSOCIATED_DICT[key]:
                    if sig not in W_sigs:
                        missing_sigs.append(sig)
        if len(missing_sigs) > 0:
            warnings.warn(('In refit: connected_sigs is set to True. The input W contains signatures with connected signatures. ' +
                           'However, W is missing some connected signatures. Specifically, W is missing: ' +
                           ','.join(missing_sigs) + '. Please fill in these missing sigs in W or make sure this is indeed what is wanted.'),
                          UserWarning)
    else:
        indices_associated_sigs = None
    model = SparseNNLSGrid(method=method, thresh1_grid=thresh_grid, ncpu=ncpu, verbose=verbose, indices_associated_sigs=indices_associated_sigs)
    model.fit(X, W)
    # Results
    H_grid = {}
    if model.thresh2_grid[0] is None:
        for thresh in thresh_grid:
            H_grid[thresh] = model.H_grid[(thresh, None)]
    elif model.thresh2_grid[0] == 0.0:
        for thresh in thresh_grid:
            H_grid[thresh] = model.H_grid[(thresh, 0.0)]
    else:
        raise ValueError('thresh2 is modified unexpectedly.')
    return H_grid, model

def _get_W_s(W, W_catalog, H_reduced, cos_similarities, thresh_new_sig):
    """Auxiliary function for matching"""
    inds_new_sig = np.arange(0, W.shape[1])[np.array(cos_similarities) < thresh_new_sig]
    inds_not_new_sig = np.arange(0, W.shape[1])[np.array(cos_similarities) >= thresh_new_sig]
    if len(inds_new_sig) > 0 and len(inds_not_new_sig) > 0:
        W_s = pd.DataFrame.copy(W.iloc[:, inds_new_sig])
        # Let's not rename these new signatures and keep their original name in W, so that we can distinguish them.
        #W_s.columns = ['Sig_N' + str(i) for i in range(1, len(inds_new_sig) + 1)]
        sig_map = pd.DataFrame(np.identity(len(inds_new_sig)), columns=W_s.columns, index=W_s.columns)
        H_tmp = H_reduced.iloc[:, inds_not_new_sig]
        W_s = pd.concat([W_s, W_catalog[H_tmp.index[H_tmp.sum(1) > 0]]], axis=1)
        sig_map = pd.concat([sig_map, H_tmp.loc[H_tmp.index[H_tmp.sum(1) > 0]]]).fillna(0.0)
    elif len(inds_new_sig) > 0 and len(inds_not_new_sig) == 0:
        W_s = pd.DataFrame.copy(W.iloc[:, inds_new_sig])
        sig_map = pd.DataFrame(np.identity(len(inds_new_sig)), columns=W_s.columns, index=W_s.columns)
    elif len(inds_new_sig) == 0 and len(inds_not_new_sig) > 0:
        W_s = pd.DataFrame.copy(W_catalog[H_reduced.index])
        sig_map = H_reduced
    return W_s, sig_map

def _clear_W_s(W, W_s, sig_map, min_sum_0p01=0.15, min_sig_contrib_ratio=0.25):
    """
    TODO:
    1. The 0.01 threshold may need to be adjusted for signature definitions with different numbers of channels.
    """
    for i in range(sig_map.shape[1]):
        set_to_zero = False
        for j in range(sig_map.shape[0]):
            fij = sig_map.iat[j,i]
            if fij < 1 and fij != 0:
                # Scale the signature according to the contribution it makes
                contrib = np.dot(W_s[sig_map.index[j]], fij)
                sum_above_0p01 = np.sum(contrib[contrib > 0.01])
                max_sig_contrib_ratio = max(contrib)/max(W[sig_map.columns[i]])
                if sum_above_0p01 < min_sum_0p01 and max_sig_contrib_ratio < min_sig_contrib_ratio:
                    set_to_zero = True
                    sig_map.iat[j,i] = 0
        if set_to_zero:
            weights, _ = sp.optimize.nnls(np.array(W_s.loc[:,sig_map.index[sig_map.loc[:, sig_map.columns[i]] > 0 ]]), np.array(W.loc[:,sig_map.columns[i]]))
            sig_map.loc[sig_map.loc[:, sig_map.columns[i]] > 0, sig_map.columns[i]] = weights
    sig_map = sig_map[sig_map.sum(1) > 0]
    W_s = W_s[sig_map.index]
    return W_s, sig_map

def _add_missing_connected_sigs(W_s, W_catalog):
    missing_sigs = []
    W_s_sigs = W_s.columns.values
    W_catalog_sigs = W_catalog.columns.values
    for key in W_s_sigs:
        if key in SIGS_ASSOCIATED_DICT.keys():
            for sig in SIGS_ASSOCIATED_DICT[key]:
                if sig not in W_s_sigs and sig in W_catalog_sigs:
                    missing_sigs.append(sig)
    W_s = pd.concat([W_s, W_catalog[missing_sigs]], axis=1)
    return W_s

def match(W, W_catalog, thresh_new_sig=0.8, method='likelihood_bidirectional', thresh=None,
          connected_sigs=False, clear_W_s=True):
    """Wrapper around SparseNNLS for matching

    Note that only one parameter thresh1 is allowed here.
    Both W and W_catalog should be pd.DataFrame.

    TODO:
    1. Directly using indices_associated_sigs does not make sense for matching.
    We should do this outside of SparseNNLS, i.e., after SparseNNLS, we check the
    set of matched signatures, and include additional associated signatures when necessary.
    But for now, this is okay. It does not affect the final set of matched signatures.
    """
    # Check input
    if W.shape[0] != W_catalog.shape[0]:
        raise ValueError('W and W_catalog have different numbers of channels.')
    if (W.index == W_catalog.index).sum() != W.shape[0]:
        raise ValueError('W and W_catalog have different indices.')
    if len(set(W.columns).intersection(W_catalog.columns)) > 0:
        raise ValueError('W and W_catalog cannot contain signatures with the same name.')
    if W.shape[0] == 83:
        if clear_W_s:
            clear_W_s = False
            warnings.warn('The signatures might be ID signatures. Therefore clear_W_s is turned off.',
                          UserWarning)
    # SparseNNLS
    model = SparseNNLS(method=method, thresh1=thresh, indices_associated_sigs=None)
    model.fit(W, W_catalog)
    # Identify new signatures not in the catalog.
    W_s, sig_map = _get_W_s(W, W_catalog, model.H_reduced, model.cos_similarities, thresh_new_sig)
    if clear_W_s:
        W_s, sig_map = _clear_W_s(W, W_s, sig_map)
    # If connected_sigs, we add missing connected signatures to W_s
    if connected_sigs:
        W_s = _add_missing_connected_sigs(W_s, W_catalog)
    return W_s, sig_map, model

def match_grid(W, W_catalog, thresh_new_sig=0.8, method='likelihood_bidirectional', thresh_grid=None, ncpu=1, verbose=0,
               connected_sigs=False, clear_W_s=True):
    """Matching on a grid of thresholds.
    """
    # Check input
    if W.shape[0] != W_catalog.shape[0]:
        raise ValueError('W and W_catalog have different numbers of channels.')
    if (W.index == W_catalog.index).sum() != W.shape[0]:
        raise ValueError('W and W_catalog have different indices.')
    if len(set(W.columns).intersection(W_catalog.columns)) > 0:
        raise ValueError('W and W_catalog cannot contain signatures with the same name.')
    if W.shape[0] == 83:
        if clear_W_s:
            clear_W_s = False
            warnings.warn('The signatures might be ID signatures. Therefore clear_W_s is turned off.',
                          UserWarning)
    # SparseNNLSGrid
    if thresh_grid is None:
        thresh_grid = np.array([0.001])
    model = SparseNNLSGrid(method=method, thresh1_grid=thresh_grid, ncpu=ncpu, verbose=verbose, indices_associated_sigs=None)
    model.fit(W, W_catalog)
    # Identify new signatures not in the catalog.
    thresh2 = model.thresh2_grid[0]
    if thresh2 is None:
        pass
    elif thresh2 == 0.0:
        pass
    else:
        raise ValueError('thresh2 is modified unexpectedly.')
    W_s_grid = {}
    sig_map_grid = {}
    for thresh in thresh_grid:
        key = (thresh, thresh2)
        W_s, sig_map = _get_W_s(W, W_catalog, model.H_reduced_grid[key], model.cos_similarities_grid[key], thresh_new_sig)
        if clear_W_s:
            W_s, sig_map = _clear_W_s(W, W_s, sig_map)
        ## Add missing connected signatures
        if connected_sigs:
            W_s = _add_missing_connected_sigs(W_s, W_catalog)
        W_s_grid[thresh] = W_s
        sig_map_grid[thresh] = sig_map
    return W_s_grid, sig_map_grid, model

def assign(X, W, W_catalog,
           method='likelihood_bidirectional',
           thresh_match=None,
           thresh_refit=None,
           thresh_new_sig=0.8,
           connected_sigs=False,
           clear_W_s=True):
    """Assign = match + refit.

    The same method will be used for both match and refit.
    Match and refit can have different thresholds. But only one threshold is allowed for each.
    If you want to skip matching, set thresh_new_sig to a value > 1.
    """
    W_s, sig_map, _ = match(W, W_catalog, thresh_new_sig=thresh_new_sig, method=method, thresh=thresh_match, connected_sigs=connected_sigs, clear_W_s=clear_W_s)
    H_s, _ = refit(X, W_s, method=method, thresh=thresh_refit, connected_sigs=connected_sigs)
    return W_s, H_s, sig_map

def assign_grid(X, W, W_catalog, method='likelihood_bidirectional',
                thresh_match_grid=None, thresh_refit_grid=None,
                thresh_new_sig=0.8, connected_sigs=False, clear_W_s=True,
                ncpu=1, verbose=0):
    """Match and refit on a grid"""
    if thresh_match_grid is None:
        thresh_match_grid = np.array([0.001])
    if thresh_refit_grid is None:
        thresh_refit_grid = np.array([0.001])
    # First, matching on a grid
    W_s_grid_1d, sig_map_grid_1d, _ = match_grid(W, W_catalog, thresh_new_sig=thresh_new_sig, method=method,
                                                 thresh_grid=thresh_match_grid, ncpu=ncpu, verbose=verbose,
                                                 connected_sigs=connected_sigs, clear_W_s=clear_W_s)
    # Second, refitting on a grid.
    # When a matching result is already calculated before, do not do refitting again.
    H_s_grid = {}
    calculated_matching_results = {}
    thresh_match_grid_unique = []
    for thresh_match in thresh_match_grid:
        sigs = tuple(sorted(W_s_grid_1d[thresh_match].columns.tolist()))
        if sigs in calculated_matching_results.keys():
            H_s_grid[thresh_match] = H_s_grid[calculated_matching_results[sigs]]
        else:
            H_s_grid_1d, _ = refit_grid(X, W_s_grid_1d[thresh_match],
                                        method=method, thresh_grid=thresh_refit_grid, ncpu=ncpu, verbose=verbose,
                                        connected_sigs=connected_sigs)
            H_s_grid[thresh_match] = H_s_grid_1d
            calculated_matching_results[sigs] = thresh_match
            thresh_match_grid_unique.append(thresh_match)
    return W_s_grid_1d, H_s_grid, sig_map_grid_1d, np.array(thresh_match_grid_unique)
