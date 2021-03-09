"""Main class for the refitting problem"""

import numpy as np
from .nnls import nnls
from .utils import match_signature_to_catalog
from .nnls_sparse import nnls_sparse
from .catalog import load_catalog

def refit_matrix(X, W, H = None,
                 method = 'llh',
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

   if method == 'none':
      H_s = nnls(X, W)

   H_s = []
   if H is None: 
       for x in X.T:
           h_s = nnls_sparse(x = x, W = W, method = method)
           H_s.append(h_s)
   else:    
       for x, h in zip(X.T, H.T):
           h_s = nnls_sparse(x = x, W = W, h = h, method = method)
           H_s.append(h_s)
   H_s = np.array(H_s) 
   return H_s.T

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

    if v5 == None:
        return v1, v2, v3, v4, n_params
    else:
        return v1, v2, v3, v4, v5, n_params
                 
def reassign(model):
 
    # We need to write here an if to give error if parameters are not set
   
    use_catalog = model.use_catalog #True, 
    catalog_name = model.catalog_name #'COSMIC_v3p1_SBS_WGS',
    thresh_match = model.thresh_match #0.99,
    thresh_new_sig = model.thresh_new_sig #0.84,
    min_contribution = model.min_contribution #0.1,
    include_top = model.include_top #False, 
    use_denovo_H = model.use_denovo_H #False, 
    method_sparse = model.method_sparse #'llh'
    frac_thresh_base = model.frac_thresh_base #0.02, 
    frac_thresh_keep = model.frac_thresh_keep #0.3, 
    frac_thresh = model.frac_thresh #0.05, 
    llh_thresh = model.llh_thresh #0.65,
    exp_thresh = model.exp_thresh #8.

    """
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
    use_denovo_H : boolean 
        Determines whether the H from denovo calculation will be used to restrict the signatures
        that are considered in the NNLS calculation. If True the relative H values of each
        signature is required to be greater than frac_thresh_base before the calculation
    frac_thresh_base : 1-d numpy array
    frac_thresh_keep :  1-d numpy array
    frac_thresh : 1-d numpy array
    llh_thresh : 1-d numpy array
    exp_thresh : 1-d numpy array
    """

    frac_thresh_base, frac_thresh_keep, frach_thresh, llh_thresh, exp_thresh, n_params_sparse = param_search_same_length(frac_thresh_base, frac_thresh_keep, frac_thresh, llh_thresh, exp_thresh)
    thresh_match, thresh_new_sig, min_contribution, include_top, n_params_match = param_search_same_length(thresh_match, thresh_new_sig, min_contribution, include_top)

    W_catalog = np.array(load_catalog(catalog_name).W)
    W = model.W
    H = model.H
    X = model.X

    W_s = {}
    H_s = {}
    index_param = 0

    frac_thresh_base_all = []
    frac_thresh_keep_all = []
    frac_thresh_all = []
    llh_thresh_all = []
    exp_thresh_all = []
     
    # Define the signatures that will be used for refitting
    if use_catalog:
        thresh_match_all = []
        thresh_new_sig_all = []
        min_contribution_all = []
        include_top_all = []
     

    for i in np.arange(0, n_params_sparse): # loop over parameters used in nnls_spars
        if use_catalog:
            for j in np.arange(0, n_params_match): # loop over parameters used in catalog match
                inds = []
                cosines = []
                coefs = []
                inds_w_model = []
                ind_w_model = 0

                # for each signature in the signature matrix match to the catalog if the cosine similarity
                # is smaller than the value thresh_new_sig use the denovo signature instead 
                for w in model.W.T:
                    match_inds, cos, coef = match_signature_to_catalog(w, W_catalog, thresh = thresh_match[j], min_contribution = min_contribution[j], include_top = include_top[j])                    
                    match_inds = np.asarray(match_inds)
                    if cos < thresh_new_sig[j]: 
                        # instead of adding the catalog index use -900 
                        # to keep track of signatures that are not matched
                        inds = np.append(inds, -900) 
                        cosines = np.append(cosines, 1)
                        coefs = np.append(coefs, 1)
                        inds_w_model = np.append(inds_w_model, ind_w_model)
                    else:
                        inds = np.append(inds, match_inds)
                        cosines = np.append(cosines, np.repeat(cos, len(coef)))            
                        coefs = np.append(coefs, coef)
                        inds_w_model = np.append(inds_w_model, np.repeat(ind_w_model, len(coef)))
                    ind_w_model = ind_w_model + 1 
        
                inds_w_model = np.array([int(index) for index in inds_w_model]) # it was saving them as double for some reason
                inds = np.array([int(index) for index in inds]) # it was saving them as double for some reason

                # more than one denovo signature might be matched to the same signature in the catalog                
                unique_inds = np.unique(inds, axis=0)

                if(np.asarray(np.where(inds == -900)).size > 0):
                    # if there are new signatures first add them to the columns/rows of W/H
                    W_s_this = W[:, inds_w_model[inds == -900]] 
                    unique_inds = unique_inds[unique_inds != -900]
                    if use_denovo_H:
                        H_s_ini_this = H[inds_w_model[inds == -900],:]         
                else:
                    W_s_this = []
                    if use_denovo_H:
                         H_s_ini_this = [] # for keeping track of how the denovo exposure should be shared across signatures after matching to multiple

                if W_s_this is None: 
                    W_s_this = W_catalog[:, unique_inds]
                else:
                    W_s_this = np.concatenate((W_s_this, W_catalog[:, unique_inds]), axis = 1)

                if use_denovo_H:
                    # for each unique signature in the catalog more than one denovo signature may be matched
                    # here we add up exposures of different denovo signatures weighted with their NNLS coefficient
                    for ind_this in unique_inds: 
                        inds_w_model_this = inds_w_model[inds == ind_this]
                        coefs_this = coefs[inds == ind_this]
                        h = np.sum(np.transpose(coefs_this) * H[inds_w_model_this,:], axis = 0)
                        if H_s_ini_this is None:
                            H_s_ini_this = [h]
                        else:
                            H_s_ini_this = np.concatenate((H_s_ini_this, [h]), axis = 0)
                    H_s_ini_this = H_s_ini_this
                else:
                     H_s_ini_this = None

                H_s_this = refit_matrix(X, W_s_this, H_s_ini_this, 
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
                # add W_s for this paramater set to the list of W_s and same for H_s
                W_s[index_param] = W_s_this
                H_s[index_param] = H_s_this
                index_param = index_param + 1


        else:
            # without catalog W and H (if use_denovo is True) are from denovo model are used
            W_s_this = model.W
            if use_denovo_H: 
                H_s_ini_this = model.H
            else:
                H_s_ini_this = None

            W_s_this = np.array(W_s_this)
            H_s_ini_this = np.array(H_s_ini_this).T

            H_s_this = refit_matrix(X, W_s_this, H_s_ini_this,
                                        method = method_sparse,
                                        frac_thresh_base = frac_thresh_base[i],
                                        frac_thresh_keep = frac_thresh_keep[i],
                                        frac_thresh = frac_thresh[i],
                                        llh_thresh = llh_thresh[i],
                                        exp_thresh = exp_thresh[i])
            index_param = index_param + 1
            frac_thresh_base_all = np.append(frac_thresh_base_all, frac_thresh_base[i])
            frac_thresh_keep_all = np.append(frac_thresh_keep_all, frac_thresh_keep[i])
            frac_thresh_all = np.append(frac_thresh_all, frac_thresh[i])
            llh_thresh_all = np.append(llh_thresh_all, llh_thresh[i])
            exp_thresh_all = append(exp_thresh_all, exp_thresh[i])
            W_s[index_param] = W_s_this
            H_s[index_param] = H_s_this
            index_param = index_param + 1
    return W_s, H_s, frac_thresh_base_all, frac_thresh_keep_all, frac_thresh_all, llh_thresh_all, exp_thresh_all, thresh_match_all, thresh_new_sig_all, min_contribution_all, include_top_all 
 
   
    


