import numpy as np
import pickle
from .utils import beta_divergence, simulate_count_matrix, match_catalog_pair

def validate(model, 
             metric_dist = 'cosine' # for use_refit 'euclidean' might be better
        ): 

    """
    This function validates the reassignment (H_s and W_s matrices) in a model
    using simulations. 
    
    If multiple parameters were tested for sparsity or matching to the catalog
    simulations are ran for all solutions. The simulated X is then processed
    through denovo calculation in the identical was as the data has been processed.
    The resulting W and H from simulations and data are compared using different 
    measures.

    You can validate a table without running refitting by simply
    creating a model and setting W_s and H_s values to a table of your choosing
    """

    dist_W_all = {}
    dists_per_sig_all = {}
    W_simul_all = {}
    H_simul_all = {}
    X_simul_all = {}

    error_W_all = []
    error_H_all = []
    dist_max_all = []
    dist_sum_all = []
    dist_max_sig_index_all = []
    nsig = []
    nsig = np.array(nsig)
    
    # fix lambda 
    model.mvnmf_hyperparameter_method = 'fixed'
    model.mvnmf_lambda_tilde_grid = np.float(model.lambda_tilde_all[model.n_components][0])

    if model.n_grid > 1:
        index = 0
        for i in range(model.n_grid): 
            W_s_this = model.W_s_all[i]
            H_s_this = model.H_s_all[i]
            pdist = {}
            dist_per_sig_this = {}
            X_simul_this = {}
            W_simul_this = {}
            H_simul_this = {}
            nsig = np.append(nsig, model.W_s_all[i].shape[1])

            for j in range(3): # for the moment repetitions are hardcoded
                X_simul_this[j] = simulate_count_matrix(W_s_this, H_s_this)

                model_simul = model.clone_model(X_simul_this[j], grid_index = i)
                model_simul.fit()
                W_simul_reordered, reordered_indices, pdist_this = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
                model_simul.W = W_simul_reordered
                model_simul.H = model_simul.H[reordered_indices,:]

                W_simul_this[j] = model_simul.W
                H_simul_this[j] = model_simul.H
                    
                index = index + 1

            W_simul_comb = np.array([np.array(W_simul_this[0]), np.array(W_simul_this[1]), np.array(W_simul_this[2])])
            H_simul_comb = np.array([np.array(H_simul_this[0]), np.array(H_simul_this[1]), np.array(H_simul_this[2])])
            X_simul_comb = np.array([np.array(X_simul_this[0]), np.array(X_simul_this[1]), np.array(X_simul_this[2])])

                                           
            X_simul_all[i] = np.average(X_simul_comb, axis = 0)
            W_simul_all[i] = np.average(W_simul_comb, axis = 0)
            H_simul_all[i] = np.average(H_simul_comb, axis = 0)

            _, _, pdist_comb = match_catalog_pair(model.W, W_simul_all[i], metric = metric_dist)

            dist_W_all[i] = pdist_comb
            dists_per_sig_all[i] = np.diagonal(pdist_comb)

            error_W_all.append(beta_divergence(model.W, W_simul_all[i], beta = 2)) 
            error_H_all.append(beta_divergence(model.H, H_simul_all[i], beta = 2))
                
            inds_max = np.where(np.max(dists_per_sig_all[i]) == dists_per_sig_all[i])
            dist_max_all.append(np.max(dists_per_sig_all[i]))
            dist_sum_all.append(np.sum(dists_per_sig_all[i]))
            dist_max_sig_index_all.append(inds_max)
        
        min_dist = min(dist_max_all)
        min_dist_sum = min(dist_sum_all)

        
        # if possible avoid assigning new signatures if the solution without new signatures replace best index
        best_grid_indices = np.where(dist_max_all < min_dist + 0.02)[0] # should we convert this into a parameter or keep it fixed
        best_grid_indices_sum = np.where(dist_sum_all < min_dist_sum + 0.02 * model.W.shape[1])[0] # should we convert this into a parameter or keep it fixed
        
        indices_without_new_sigs = np.where(np.char.find('Sig_N', model.signature_names_all[i]) == -1)[0]
        if len(indices_without_new_sigs) > 0:        
            best_grid_indices = [index for item,index in enumerate(best_grid_indices) if item in indices_without_new_sigs]
        if len(indices_without_new_sigs) > 0:
            best_grid_indices_sum = [index for item,index in enumerate(best_grid_indices_sum) if item in indices_without_new_sigs]
            
        min_nsig = np.min(nsig[best_grid_indices])        
        best_grid_indices = np.array(best_grid_indices)[np.array(nsig)[np.array(best_grid_indices)] == min_nsig]
        
        min_nsig_sum = np.min(nsig[best_grid_indices_sum])        
        best_grid_indices_sum =  np.array(best_grid_indices_sum)[np.array(nsig)[np.array(best_grid_indices_sum)] == min_nsig_sum]
            
        # check the error of H
        min_error_H = np.min(np.array(error_H_all)[best_grid_indices])
        best_grid_index = np.array(best_grid_indices)[np.array(error_H_all)[best_grid_indices] == min_error_H]

        best_grid_index = np.asscalar(best_grid_index)

        min_error_H_sum = np.min(np.array(error_H_all)[best_grid_indices_sum])
        best_grid_index_sum = np.array(best_grid_indices_sum)[np.array(error_H_all)[best_grid_indices_sum] == min_error_H_sum]
        best_grid_index_sum = np.asscalar(best_grid_index_sum)

    else:  # if there was no grid search
        best_grid_index = None
        best_grid_index_sum = None
        best_grid_indices = None
        best_grid_indices_sum = None
        W_s = model.W_s
        H_s = model.H_s
        pdist = {}
        H_simul_this = {}
        W_simul_this = {}
        X_simul_this = {}
        for j in range(3): # for the moment repetitions are hardcoded
            X_simul_this[j] = simulate_count_matrix(W_s, H_s)
            model_simul = model.clone_model(X_simul_this[j], grid_index = 1) 
            model_simul.fit()
            W_simul_reordered, reordered_indices, pdist[j] = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
            model_simul.W = W_simul_reordered
            model_simul.H = model_simul.H[reordered_indices,:]
            W_simul_this[j] = model_simul.W
            H_simul_this[j] = model_simul.H

       
        W_simul_comb = [W_simul_this[0], W_simul_this[1], W_simul_this[2]]
        H_simul_comb = [H_simul_this[0], H_simul_this[1], H_simul_this[2]]
        X_simul_comb = [X_simul_this[0], X_simul_this[1], X_simul_this[2]]
            
        X_simul_all[0] = np.average(X_simul_comb, axis = 0)
        W_simul_all[0] = np.average(W_simul_comb, axis = 0)
        H_simul_all[0] = np.average(H_simul_comb, axis = 0)

        _, _, pdist_comb = match_catalog_pair(model.W, W_simul_all[0], metric = metric_dist)
        dist_W_all[0] = pdist_comb
        dists_per_sig_comb = np.diagonal(pdist_comb)
                
        inds_max = np.where(np.max(dists_per_sig_comb) == dists_per_sig_comb)
        dist_max_sig_index_all.append(inds_max)
        dist_max_all.append(np.max(dists_per_sig_comb))
        dist_sum_all.append(np.sum(dists_per_sig_comb))
        dist_W_all[0] = pdist_comb
        error_W_all.append(beta_divergence(model.W, W_simul_all[0], beta = 2))
        error_H_all.append(beta_divergence(model.H, H_simul_all[0], beta = 2))
        best_grid_index = 0
        best_grid_index_sum = 0
        
    return W_simul_all, H_simul_all, X_simul_all, best_grid_index, best_grid_index_sum, best_grid_indices, best_grid_indices_sum, error_W_all, error_H_all, dist_W_all, dist_max_sig_index_all,  dist_max_all, dist_sum_all

