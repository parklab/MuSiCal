import numpy as np
import pickle
from .utils import beta_divergence, simulate_count_matrix, match_catalog_pair

def validate(model, 
             use_refit = False, 
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
    save_models = False

    dist_W_all = {}
    dists_per_sig_all = {}
    W_simul_all = {}
    H_simul_all = {}
    X_simul_all = {}

    error_W_all = []
    error_H_all = []
    dist_max_all = []
    dist_max_sig_index_all = []
    
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
            
            for j in range(3): # for the moment repetitions are hardcoded
                X_simul_this[j] = simulate_count_matrix(W_s_this, H_s_this)

                model_simul = model.clone_model(X_simul_this[j], grid_index = i)
                if use_refit:
                    model_simul.W = model.W_s_all[i]
                    model_simul.run_reassign()
                    H_simul_reordered, reordered_indices, pdist_this = match_catalog_pair(model.H_s_all[i].T, model_simul.H_s.T, metric = metric_dist)
                    W_simul_this[j] = model_simul.W_s
                    H_simul_this[j] = model_simul.H_s
                else:
                    model_simul.fit()
                    W_simul_reordered, reordered_indices, pdist_this = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
                    model_simul.W = W_simul_reordered
                    model_simul.H = model_simul.H[reordered_indices,:]

                    W_simul_this[j] = model_simul.W
                    H_simul_this[j] = model_simul.H
                    
                if save_models: 
                    list_models_simul[index] = model_simul
  
                pdist[j] = pdist_this
                dist_per_sig_this[j] = np.diagonal(pdist_this)
                index = index + 1

            pdist_comb = [pdist[0], pdist[1], pdist[2]]
            dists_per_sig_comb = [dist_per_sig_this[0], dist_per_sig_this[1], dist_per_sig_this[2]]
            W_simul_comb = np.array([np.array(W_simul_this[0]), np.array(W_simul_this[1]), np.array(W_simul_this[2])])
            H_simul_comb = np.array([np.array(H_simul_this[0]), np.array(H_simul_this[1]), np.array(H_simul_this[2])])
            X_simul_comb = np.array([np.array(X_simul_this[0]), np.array(X_simul_this[1]), np.array(X_simul_this[2])])

                               
            dist_W_all[i] = np.average(pdist_comb, axis = 0)
            dists_per_sig_all[i] = np.average(dists_per_sig_comb, axis = 0)
            
            X_simul_all[i] = np.average(X_simul_comb, axis = 0)
            W_simul_all[i] = np.average(W_simul_comb, axis = 0)
            H_simul_all[i] = np.average(H_simul_comb, axis = 0)
            if use_refit:
                error_W_all.append(beta_divergence(model.W_s_all[i], W_simul_all[i], beta = 2))
                error_H_all.append(beta_divergence(model.H_s_all[i], H_simul_all[i], beta = 2))
            else:
                error_W_all.append(beta_divergence(model.W, W_simul_all[i], beta = 2)) 
                error_H_all.append(beta_divergence(model.H, H_simul_all[i], beta = 2))
                
            inds_max = np.where(np.max(dists_per_sig_all[i]) == dists_per_sig_all[i])
            dist_max_all.append(np.sum(dists_per_sig_all[i]))
            dist_max_sig_index_all.append(inds_max)
        
        min_dist = min(dist_max_all)
        
#        best_grid_index = dist_max_all.index(min(dist_max_all))
        
        # if possible avoid assigning new signatures if the solution without new signatures replace best index
        best_grid_indices = np.where(dist_max_all < min_dist + 0.02 * model.W.shape[1])[0] # should we convert this into a parameter or keep it fixed
        indices_without_new_sigs = np.where(np.char.find('Sig_N0', model.signature_names_all[i]) == -1)[0]
        
        
        best_indices_without_new_sigs = [index for item,index in enumerate(best_grid_indices) if item in indices_without_new_sigs]
                
        if len(best_indices_without_new_sigs) > 0:
            best_grid_indices = best_indices_without_new_sigs
            
        # check the error of H
        if use_refit:
            best_grid_index = best_grid_indices[dist_max_all[best_grid_indices] == min(dist_max_all)]
        else:
            min_error_H = np.min(np.array(error_H_all)[best_grid_indices])
            best_grid_index = np.array(best_grid_indices)[np.array(error_H_all)[best_grid_indices] == min_error_H]

        best_grid_index = np.asscalar(best_grid_index)
        
        W_simul = W_simul_all[best_grid_index]
        H_simul = H_simul_all[best_grid_index]
        X_simul = X_simul_all[best_grid_index]
        error_W = error_W_all[best_grid_index]
        error_H = error_H_all[best_grid_index]
        dist_W = dist_W_all[best_grid_index]
        dist_max = dist_max_all[best_grid_index]
        dist_max_sig_index = dist_max_sig_index_all[best_grid_index]        


    else:  # if there was no grid search
        best_grid_index = None
        W_s = model.W_s
        H_s = model.H_s
        pdist = {}
        H_simul_this = {}
        W_simul_this = {}
        X_simul_this = {}
        for j in range(3): # for the moment repetitions are hardcoded
            X_simul_this[j] = simulate_count_matrix(W_s, H_s)
            model_simul = model.clone_model(X_simul, grid_index = 1) 
            if use_refit:
                model_simul.W = model.W_s
                model_simul.run_reassign()
                H_simul_reordered, reordered_indices, pdist[j] = match_catalog_pair(model.H_s.T, model_simul.H_s.T, metric = metric_dist)
                W_simul_this[j] = model_simul.W_s
                H_simul_this[j] = model_simul.H_s
            else:
                model_simul.fit()
                W_simul_reordered, reordered_indices, pdist[j] = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
                model_simul.W = W_simul_reordered
                model_simul.H = model_simul.H[reordered_indices,:]
                W_simul_this[j] = model_simul.W
                H_simul_this[j] = model_simul.H

            if use_refit:
                _,_,pdist[j] = match_catalog_pair(model.H_s.T, model_simul.H_s.T, metric = metric_dist)
            else: 
                _,_,pdist[j] = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)


        pdist_comb = [pdist[0], pdist[1], pdist[2]]
        dist_W = np.diagonal(pdist_comb)        
        dists_per_sig_comb = np.diagonal(np.average(pdist_comb, axis = 2))
        
        W_simul_comb = [W_simul_this[0], W_simul_this[1], W_simul_this[2]]
        H_simul_comb = [H_simul_this[0], H_simul_this[1], H_simul_this[2]]
        X_simul_comb = [X_simul_this[0], X_simul_this[1], X_simul_this[2]]
            
        X_simul = np.average(X_simul_comb, axis = 2)
        W_simul = np.average(W_simul_comb, axis = 2)
        H_simul = np.average(H_simul_comb, axis = 2)
                
        inds_max = np.where(np.max(dists_per_sig_comb) == dists_per_sig_comb)
        dist_max = np.sum(dists_per_sig_comb)
        dist_max_sig_index = inds_max
        dist_W = pdist
        if use_refit:
            error_W = beta_divergence(model.W_s, W_simul, beta = 2)
            error_H = beta_divergence(model.H_s, H_simul, beta = 2)
        else:
            error_W = beta_divergence(model.W, W_simul, beta = 2)
            error_H = beta_divergence(model.H, H_simul, beta = 2)
        
    return W_simul, H_simul, X_simul, best_grid_index, best_grid_indices, error_W, error_H, dist_W, dist_max, dist_max_sig_index,  dist_max_all, dist_max_sig_index_all, W_simul_all, H_simul_all, X_simul_all, error_W_all, error_H_all, dist_W_all
