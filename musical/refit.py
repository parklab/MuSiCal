"""Main class for the refitting problem"""

import numpy as np
from .nnls import nnls
from .utils import match_signature_to_catalog, beta_divergence
from .nnls_sparse import nnls_sparse
from .catalog import load_catalog

def refit_matrix(X, W, method = 'llh',
                 frac_thresh_base = 0.02,
                 frac_thresh_keep = 0.3,
                 frac_thresh = 0.05,
                 llh_thresh = 0.65,
                 exp_thresh = 8.):
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
   for x in X.T:
       if method == 'none':
           h, _ = sp.optimize.nnls(W, x)
       else:
           h = nnls_sparse(x = x, W = W, method = method,
                          frac_thresh_base = frac_thresh_base,
                          frac_thresh_keep = frac_thresh_keep,
                          frac_thresh = frac_thresh,
                          llh_thresh = llh_thresh,
                          exp_thresh = exp_thresh)
       H.append(h)
   H = np.array(H)
   H = H.T
   reconstruction_error = beta_divergence(X, W @ H, beta=1, square_root=False)

   return H, reconstruction_error

def param_search_same_length(v1, v2, v3, v4, v5 = None):
    """
    Allow a grid search over the parameters of same numpy array length or all the parameters except
    one being constant, in which case the other parameters are converted into arrays of same
    length as the variable one with a constant value
    """
    s1 = len(v1)
    s2 = len(v2)
    s3 = len(v3)
    s4 = len(v4)
    same_size_match_params = (s1 == s2 == s3 == s4)
    if v5 != None:
        s5 = len(v5)
        same_size_match_params = (s1 == s2 == s3 == s4 == s5)
    else:
        s5 = None
    n_params = 0

    if same_size_match_params:
        n_params = s1
    elif (s2 == s3 == s4 ==  1) & (s1 != 1) & (s5 == None):
        v2 = np.repeat(v2, s1)
        v3 = np.repeat(v3, s1)
        v4 = np.repeat(v4, s1)
        n_params = s1
    elif (s1 == s3 == s4 == 1) & (s2 != 1) & (s5 == None):
        v1 = np.repeat(v1, s2)
        v3 = np.repeat(v3, s2)
        v4 = np.repeat(v4, s2)
        n_params = s2
    elif (s1 == s2 == s4 == 1) & (s3 != 1) & (s5 == None):
        v1 = np.repeat(v1, s3)
        v2 = np.repeat(v2, s3)
        v4 = np.repeat(v4, s3)
        n_params = s3
    elif (s1 == s2 == s3 == 1) & (s4 != 1) & (s5 == None):
        v1 = np.repeat(v1, s4)
        v2 = np.repeat(v2, s4)
        v3 = np.repeat(v3, s4)
        n_params = s4
    elif (s2 == s3 == s4 == s5 == 1) & (s1 != 1):
        v2 = np.repeat(v2, s1)
        v3 = np.repeat(v3, s1)
        v4 = np.repeat(v4, s1)
        v5 = np.repeat(v5, s1)
        n_params = s1
    elif (s1 == s3 == s4 == s5 == 1) & (s2 != 1):
        v1 = np.repeat(v1, s2)
        v3 = np.repeat(v3, s2)
        v4 = np.repeat(v4, s2)
        v5 = np.repeat(v5, s2)
        n_params = s2
    elif (s1 == s2 == s4 == s5 == 1) & (s3 != 1):
        v1 = np.repeat(v1, s3)
        v2 = np.repeat(v2, s3)
        v4 = np.repeat(v4, s3)
        v5 = np.repeat(v5, s3)
        n_params = s3
    elif (s1 == s2 == s3 == s5 == 1) & (s4 != 1):
        v1 = np.repeat(v1, s4)
        v2 = np.repeat(v2, s4)
        v3 = np.repeat(v3, s4)
        v5 = np.repeat(v5, s4)
        n_params = s4
    elif (s1 == s2 == s3 == s4 == 1) & (s5 != 1):
        v1 = np.repeat(v1, s5)
        v2 = np.repeat(v2, s5)
        v3 = np.repeat(v3, s5)
        v4 = np.repeat(v4, s5)
        n_params = s5
    else:
        raise ValueError('parameters should be either same size or only one variable')

    if s5 == None:
        return v1, v2, v3, v4, n_params
    else:
        return v1, v2, v3, v4, v5, n_params



def get_decomposed_W(W, W_catalog, signatures, thresh_match, thresh_new_sig, min_contribution, include_top):
    inds = [] # indices of the catalog to which W model matches to
    inds_w_model_new_sig = [] # indices of W from the model that are not matched to the catalog
    ind_w_model = 0

    # for each signature in the signature matrix match to the catalog if the cosine similarity
    # is smaller than the value thresh_new_sig use the denovo signature instead
    for w in W.T:
        match_inds, cos, coef = match_signature_to_catalog(w, W_catalog, thresh = thresh_match, min_contribution = min_contribution, include_top = include_top)
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

def reassign(model):  # maybe we should move this function into denovo.py

    # We need to write here an if to give error if parameters are not set

    use_catalog = model.use_catalog #True,
    catalog_name = model.catalog_name #'COSMIC_v3p1_SBS_WGS',
    thresh_match = model.thresh_match #0.99,
    thresh_new_sig = model.thresh_new_sig #0.84,
    min_contribution = model.min_contribution #0.1,
    include_top = model.include_top #False,
    method_sparse = model.method_sparse #'llh'
    frac_thresh_base = model.frac_thresh_base #0.02,
    frac_thresh_keep = model.frac_thresh_keep #0.3,
    frac_thresh = model.frac_thresh #0.05,
    llh_thresh = model.llh_thresh #0.65,
    exp_thresh = model.exp_thresh #8.

    """
    Example for running without denovo calculation
    model = musical.DenovoSig(X, min_n_components=min_n_components, max_n_components = max_n_components, init='nndsvdar', method='nmf', n_replicates=10, ncpu=2)
    model.W = W_user # Set the signature matrix
    model.run_reassign()

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

    min_contribution : 1-d numpy array
        If the coefficient of a single signature is smaller than this value it will not be considered
        as valid composite signature
    thresh_new_sig : 1-d numpy array
        If the cosine of the composite signature is smaller than this value the signature is considered
        to be new, has no matching to the catalog and
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

    frac_thresh_base, frac_thresh_keep, frac_thresh, llh_thresh, exp_thresh, n_params_sparse = param_search_same_length(frac_thresh_base, frac_thresh_keep, frac_thresh, llh_thresh, exp_thresh)
    thresh_match, thresh_new_sig, min_contribution, include_top, n_params_match = param_search_same_length(thresh_match, thresh_new_sig, min_contribution, include_top)

    catalog = load_catalog(catalog_name)
    W_catalog = np.array(catalog.W)
    signatures = np.array(catalog.signatures)

    W_catalog = W_catalog[:,[index for index,item in enumerate(signatures) if item != "SBS40"]]
    signatures = signatures[[index for index,item in enumerate(signatures) if item != "SBS40"]]

    W = model.W

    X = model.X

    # here add error statement if W and W_catalog do not match in size
    signames = {}
    W_s = {}
    H_s = {}
    index_param = 0

    reconstruction_error_s_all = []

    frac_thresh_base_all = []
    frac_thresh_keep_all = []
    frac_thresh_all = []
    llh_thresh_all = []
    exp_thresh_all = []

    # these are returned empty if use_catalog is False
    thresh_match_all = []
    thresh_new_sig_all = []
    min_contribution_all = []
    include_top_all = []

    if model.features is not None:
        inds = []
        for item in model.features:
            inds.append(catalog.features.index(item))
        W_catalog = W_catalog[inds, :]

    for i in np.arange(0, n_params_sparse): # loop over parameters used in nnls_sparse
        if use_catalog:
            for j in np.arange(0, n_params_match): # loop over parameters used in catalog match
                W_s_this, signames_this = get_decomposed_W(W, W_catalog, signatures, thresh_match[j], thresh_new_sig[j], min_contribution[j], include_top[j])


                H_s_this, reco_error = refit_matrix(X, W_s_this,
                                                    method = method_sparse,
                                                    frac_thresh_base = frac_thresh_base[i],
                                                    frac_thresh_keep = frac_thresh_keep[i],
                                                    frac_thresh = frac_thresh[i],
                                                    llh_thresh = llh_thresh[i],
                                                    exp_thresh = exp_thresh[i])

                # a flat paramater array following the same indices as W_s and H_s are generated
                # to be used in validation step to pick the best parameters once the best W_s and H_s are determined
                frac_thresh_base_all = np.append(frac_thresh_base_all, frac_thresh_base[i])
                frac_thresh_keep_all = np.append(frac_thresh_keep_all, frac_thresh_keep[i])
                frac_thresh_all = np.append(frac_thresh_all, frac_thresh[i])
                llh_thresh_all = np.append(llh_thresh_all, llh_thresh[i])
                exp_thresh_all = np.append(exp_thresh_all, exp_thresh[i])
                thresh_match_all = np.append(thresh_match_all, thresh_match[j])
                thresh_new_sig_all = np.append(thresh_new_sig_all, thresh_new_sig[j])
                min_contribution_all = np.append(min_contribution_all, min_contribution[j])
                include_top_all = np.append(include_top_all,include_top[j])
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
                                               frac_thresh_base = frac_thresh_base[i],
                                               frac_thresh_keep = frac_thresh_keep[i],
                                               frac_thresh = frac_thresh[i],
                                               llh_thresh = llh_thresh[i],
                                               exp_thresh = exp_thresh[i])
            frac_thresh_base_all = np.append(frac_thresh_base_all, frac_thresh_base[i])
            frac_thresh_keep_all = np.append(frac_thresh_keep_all, frac_thresh_keep[i])
            frac_thresh_all = np.append(frac_thresh_all, frac_thresh[i])
            llh_thresh_all = np.append(llh_thresh_all, llh_thresh[i])
            exp_thresh_all = np.append(exp_thresh_all, exp_thresh[i])
            reconstruction_error_s_all = np.append(reconstruction_error_s_all, reco_error)
            W_s[index_param] = W_s_this
            H_s[index_param] = H_s_this
            signames[index_param] = signames_this
            index_param = index_param + 1

    n_grid = index_param
    return W_s, H_s, signames, reconstruction_error_s_all, n_grid, frac_thresh_base_all, frac_thresh_keep_all, frac_thresh_all, llh_thresh_all, exp_thresh_all, thresh_match_all, thresh_new_sig_all, min_contribution_all, include_top_all
