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
from .nnls import nnls
from .nnls_sparse import SparseNNLS, SparseNNLSGrid
from .utils import match_signature_to_catalog_nnls_sparse, beta_divergence, get_sig_indices_associated
from .catalog import load_catalog


def refit(X, W, method='likelihood_bidirectional', thresh=None,
          indices_associated_sigs=None):
    """Wrapper around SparseNNLS for refitting

    Note that only one parameter thresh1 is allowed here.
    Both X and W should be pd.DataFrame.
    """
    # Check input
    if X.shape[0] != W.shape[0]:
        raise ValueError('X and W have different numbers of channels.')
    if (X.index == W.index).sum() != X.shape[0]:
        raise ValueError('X and W have different indices.')
    # SparseNNLS
    model = SparseNNLS(method=method, thresh1=thresh, indices_associated_sigs=indices_associated_sigs)
    model.fit(X, W)
    return model.H, model

def refit_grid(X, W, method='likelihood_bidirectional', thresh_grid=None, ncpu=1, verbose=0,
               indices_associated_sigs=None):
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

def _clear_W_s(W, W_s, sig_map, min_sum_0p01 = 0.15, min_sig_contrib_ratio = 0.25):
    for i in range(sig_map.shape[1]):
        set_to_zero = False
        for j in range(sig_map.shape[0]):
            fij = sig_map.iat[j,i]
            if fij < 1 and fij != 0:
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
    
def match(W, W_catalog, thresh_new_sig=0.8, method='likelihood_bidirectional', thresh=None,
          indices_associated_sigs=None):
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
    # SparseNNLS
    model = SparseNNLS(method=method, thresh1=thresh, indices_associated_sigs=indices_associated_sigs)
    model.fit(W, W_catalog)
    # Identify new signatures not in the catalog.
    W_s, sig_map = _get_W_s(W, W_catalog, model.H_reduced, model.cos_similarities, thresh_new_sig)
    W_s, sig_map = _clear_W_s(W, W_s, sig_map)
    return W_s, sig_map, model

def match_grid(W, W_catalog, thresh_new_sig=0.8, method='likelihood_bidirectional', thresh_grid=None, ncpu=1, verbose=0,
               indices_associated_sigs=None):
    """Matching on a grid of thresholds.
    """
    # Check input
    if W.shape[0] != W_catalog.shape[0]:
        raise ValueError('W and W_catalog have different numbers of channels.')
    if (W.index == W_catalog.index).sum() != W.shape[0]:
        raise ValueError('W and W_catalog have different indices.')
    if len(set(W.columns).intersection(W_catalog.columns)) > 0:
        raise ValueError('W and W_catalog cannot contain signatures with the same name.')
    # SparseNNLSGrid
    if thresh_grid is None:
        thresh_grid = np.array([0.001])
    model = SparseNNLSGrid(method=method, thresh1_grid=thresh_grid, ncpu=ncpu, verbose=verbose, indices_associated_sigs=indices_associated_sigs)
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
        W_s, sig_map = _clear_W_s(W, W_s, sig_map)
        W_s_grid[thresh] = W_s
        sig_map_grid[thresh] = sig_map
    return W_s_grid, sig_map_grid, model

def assign(X, W, W_catalog,
           method='likelihood_bidirectional',
           thresh_match=None,
           thresh_refit=None,
           thresh_new_sig=0.8,
           indices_associated_sigs=None):
    """Assign = match + refit.

    The same method will be used for both match and refit.
    Match and refit can have different thresholds. But only one threshold is allowed for each.
    If you want to skip matching, set thresh_new_sig to a value > 1.
    """
    W_s, sig_map, _ = match(W, W_catalog, thresh_new_sig=thresh_new_sig, method=method, thresh=thresh_match, indices_associated_sigs=indices_associated_sigs)
    H_s, _ = refit(X, W_s, method=method, thresh=thresh_refit, indices_associated_sigs=indices_associated_sigs)
    return W_s, H_s, sig_map

def assign_grid(X, W, W_catalog, method='likelihood_bidirectional',
                thresh_match_grid=None, thresh_refit_grid=None,
                thresh_new_sig=0.8, indices_associated_sigs=None,
                ncpu=1, verbose=0):
    """Match and refit on a grid"""
    if thresh_match_grid is None:
        thresh_match_grid = np.array([0.001])
    if thresh_refit_grid is None:
        thresh_refit_grid = np.array([0.001])
    # First, matching on a grid
    W_s_grid_1d, sig_map_grid_1d, _ = match_grid(W, W_catalog, thresh_new_sig=thresh_new_sig, method=method,
                                                 thresh_grid=thresh_match_grid, ncpu=ncpu, verbose=verbose,
                                                 indices_associated_sigs=indices_associated_sigs)
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
                                        indices_associated_sigs=indices_associated_sigs)
            H_s_grid[thresh_match] = H_s_grid_1d
            calculated_matching_results[sigs] = thresh_match
            thresh_match_grid_unique.append(thresh_match)
    return W_s_grid_1d, H_s_grid, sig_map_grid_1d, np.array(thresh_match_grid_unique)





###############################################################################
################################ Old codes below ##############################
###############################################################################

def refit_matrix(X, W, method = 'likelihood_bidirectional',
                 thresh1 = 0.001,
                 thresh2 = None,
                 indices_associated_sigs = None ):
   """
   Refitting each column of the matrix either using nnls or nnls sparse
   If H is None an initial with NNLS an initial H is calculated and
   baseline threshold is applied on this H (frac_thresh_base) otherwise
   the user provided H, which in reassign is the denovo result, serves this purpose

   Parameters
   ----------
   method : str
       It can be llh "for" likelihood, "cut" for cut-based selection and
       "none" for simple nnls
   see nnls_sparse for other parameters
   """
   # we could consider moving nnls function in here from nnls.py. I added it here

   H = []
   if method == None:
       for x in X.T:
           h, _ = sp.optimize.nnls(W, x)
           H.append(h)
       H = np.array(H)
       X_reconstructed = np.array([W.values @ h for h in H.T.values]).T

   else:
       sparse_method = SparseNNLS(method = method,
                                  thresh1 = thresh1,
                                  thresh2 = thresh2,
                                  indices_associated_sigs = indices_associated_sigs)
       sparse_method.fit(X, W)
       H = sparse_method.H
       X_reconstructed = sparse_method.X_reconstructed
   reconstruction_error = beta_divergence(X, np.array(X_reconstructed), beta=1, square_root=False)

   H = np.array(H)

   return H, reconstruction_error

def param_search_same_length(v1, v2, v3 = None):
    """
    Allow a grid search over the parameters of same numpy array length or all the parameters except
    one being constant, in which case the other parameters are converted into arrays of same
    length as the variable one with a constant value
    """
    s1 = len(v1)
    s2 = len(v2)
    s3 = None
    same_size_match_params = (s1 == s2)
    if v3 != None:
        s3 = len(v3)
        same_size_match_params = (s1 == s2 == s3)
    else:
        s3 = None
    n_params = 0

    if same_size_match_params:
        n_params = s1
    elif (s2 == 1) & (s1 != 1) & (s3 == None):
        v2 = np.repeat(v2, s1)
        n_params = s1
    elif (s1 == 1) & (s2 != 1) & (s3 == None):
        v1 = np.repeat(v1, s2)
        n_params = s2
    elif (s1 == s2 == 1) & (s3 != 1):
        v1 = np.repeat(v1, s3)
        v2 = np.repeat(v2, s3)
        n_params = s3
    elif (s1 == s3 == 1) & (s2 != 1):
        v1 = np.repeat(v1, s2)
        v3 = np.repeat(v3, s2)
        n_params = s2
    elif (s2 == s3 == 1) & (s1 != 1):
        v2 = np.repeat(v2, s1)
        v3 = np.repeat(v3, s1)
        n_params = s1
    else:
        raise ValueError('parameters should be either same size or only one of them longer than one')

    if s3 == None:
        return v1, v2, n_params
    else:
        return v1, v2, v3, n_params

def get_decomposed_W(W, W_catalog, signatures, thresh1, thresh2, thresh_new_sig):
    inds = [] # indices of the catalog to which W model matches to
    inds_w_model_new_sig = [] # indices of W from the model that are not matched to the catalog
    ind_w_model = 0

    # for each signature in the signature matrix match to the catalog if the cosine similarity
    # is smaller than the value thresh_new_sig use the denovo signature instead
    for w in W.T:
        match_inds, cos, coef = match_signature_to_catalog_nnls_sparse(w = w, W_catalog = W_catalog, thresh1 = thresh1, thresh2 = thresh2, method = 'likelihood_bidirectional')
        if match_inds == ():
            inds_w_model_new_sig.append(ind_w_model)
        else:
            match_inds = np.asarray(match_inds)
            if cos < thresh_new_sig:
                inds_w_model_new_sig.append(ind_w_model)
            else:
                inds = np.append(inds, match_inds)
        ind_w_model = ind_w_model + 1


    unique_inds = np.unique(inds).astype(int)

    inds_w_model_new_sig = np.array(inds_w_model_new_sig)
    if(len(inds_w_model_new_sig) > 0):
        W_s = W[:, inds_w_model_new_sig]
        signames = ["Sig_N" + str(i) for i in range(0, inds_w_model_new_sig.size)]
        signames = np.array(signames)
        if(unique_inds.size > 0):
            W_s = np.concatenate((W_s, W_catalog[:, unique_inds]), axis = 1)
            signames = np.append(signames, signatures[unique_inds])
    else:
        W_s = W_catalog[:, unique_inds]
        signames = signatures[unique_inds]
        signames = np.array(signames)

    return W_s, signames

def reassign(model, W_catalog, signatures, force_assign_associated = False):  # maybe we should move this function into denovo.py

    # We need to write here an if to give error if parameters are not set

    use_catalog = model.use_catalog
    thresh1_match = model.thresh1_match
    thresh2_match = model.thresh2_match
    method_sparse = model.method_sparse
    thresh1 = model.thresh1
    thresh2 = model.thresh2
    thresh1_match = model.thresh1_match
    thresh2_match = model.thresh2_match
    thresh_new_sig = model.thresh_new_sig

    """
    Example for running without denovo calculation
    model = musical.DenovoSig(X, min_n_components=min_n_components, max_n_components = max_n_components, init='nndsvdar', method='nmf', n_replicates=10, ncpu=2)
    model.W = W_user # Set the signature matrix
    model.run_reassign(W_catalog, signatures)

    Reassignment is done by first determining the signatures that will be used in refitting
    and then using them to refit X.

    Parameters are preset in the DenovoSig object, below are the explanation of the ones
    this function uses

    Parameters
    ----------
    use catalog : boolean
        if False : Signatures discovered de-novo are used
        if True : First signatures are matched to catalog and a new matrix of signatures
        are calculated using match_signature_to_catalog function

    thresh1_match, thresh2_match: Used to define the SparseNNLS object to match denovo signature to the catalog
    thresh_new_sig : 1-d numpy array
        If the cosine of the composite signature is smaller than this value the signature is considered
        to be new, has no matching to the catalog and
    thresh1, thresh2: Used to define the SparseNNLS object used in the refitting of signatures to the spectrum of each sample

    - Parameters that are used when matching to catalog (see match_signature_to_catalog function):
    thresh_match : 1-numpy array
    include top : boolean

    - Parameters that are used when introducing sparsity (see refit function)
    frac_thresh_base : 1-d numpy array
    frac_thresh_keep :  1-d numpy array
    frac_thresh : 1-d numpy array
    llh_thresh : 1-d numpy array
    exp_thresh : 1-d numpy array
    """

    thresh1, thresh2, n_params_sparse = param_search_same_length(thresh1, thresh2)
    thresh1_match, thresh2_match, thresh_new_sig, n_params_match = param_search_same_length(thresh1_match, thresh2_match, thresh_new_sig)


    W = model.W

    X = model.X

    # here add error statement if W and W_catalog do not match in size
    signames = {}
    W_s = {}
    H_s = {}
    index_param = 0

    reconstruction_error_s_all = []

    thresh1_all = []
    thresh2_all = []

    # these are returned empty if use_catalog is False
    thresh1_match_all = []
    thresh2_match_all = []
    thresh_new_sig_all = []

    if model.features is not None:
        inds = []
        for item in model.features:
            inds.append(catalog.features.index(item))
        W_catalog = W_catalog[inds, :]

    for i in np.arange(0, n_params_sparse): # loop over parameters used in nnls_sparse
        if use_catalog:
            for j in np.arange(0, n_params_match): # loop over parameters used in catalog match
                W_s_this, signames_this = get_decomposed_W(W, W_catalog, signatures, thresh1_match[j], thresh2_match[j], thresh_new_sig[j])

                if force_assign_associated:
                    indices_associated_sigs, signames_this_tmp = get_sig_indices_associated(signames_this, signatures)
                    signames_this_tmp = np.array(signames_this_tmp)
                    if signames_this.size != signames_this_tmp.size:
                        signames_this = signames_this_tmp
                        W_s_this = W_catalog[:,[index for index,item in enumerate(signatures) if item in signames_this]]

                    if signames_this.size == signames_this_tmp.size:
                        if not (signames_this == signames_this_tmp).all():
                            signames_this = signames_this_tmp
                            W_s_this = W_catalog[:,[index for index,item in enumerate(signatures) if item in signames_this]]
                else:
                    indices_associated_sigs = None
                H_s_this, reco_error = refit_matrix(X, W_s_this,
                                                    method = method_sparse,
                                                    thresh1 = thresh1[i],
                                                    thresh2 = thresh2[i],
                                                    indices_associated_sigs = indices_associated_sigs )

                # a flat paramater array following the same indices as W_s and H_s are generated
                # to be used in validation step to pick the best parameters once the best W_s and H_s are determined
                thresh1_all = np.append(thresh1_all, thresh1[i])
                thresh2_all = np.append(thresh2_all, thresh2[i])
                thresh1_match_all = np.append(thresh1_match_all, thresh1_match[j])
                thresh2_match_all = np.append(thresh2_match_all, thresh2_match[j])
                thresh_new_sig_all = np.append(thresh_new_sig_all, thresh_new_sig[j])
                reconstruction_error_s_all = np.append(reconstruction_error_s_all, reco_error)

                # add W_s for this paramater set to the list of W_s and same for H_s
                W_s[index_param] = W_s_this
                H_s[index_param] = H_s_this
                signames[index_param] = signames_this
                index_param = index_param + 1


        else:
            # without catalog W of the model is used
            W_s_this = W
            W_s_this = np.array(W_s_this)
            signames_this = ["Sig_D" + str(i) for i in range(0, W_s_this.shape[1])]

            H_s_this, reco_error = refit_matrix(X, W_s_this,
                                               method = method_sparse,
                                               thresh1 = thresh1[i],
                                               thresh2 = thresh2[i])
            reconstruction_error_s_all = np.append(reconstruction_error_s_all, reco_error)
            W_s[index_param] = W_s_this
            H_s[index_param] = H_s_this
            signames[index_param] = signames_this
            index_param = index_param + 1

    n_grid = index_param
    for i in range(n_grid):
        W_s[i] = W_s[i][:, np.sum(H_s[i], axis = 1) > 0]
        signames[i] = signames[i][ np.sum(H_s[i], axis = 1) > 0]
        H_s[i] = H_s[i][np.sum(H_s[i], axis = 1) > 0, :]


    return W_s, H_s, signames, reconstruction_error_s_all, n_grid, thresh1_all, thresh2_all, thresh1_match_all, thresh2_match_all, thresh_new_sig_all
