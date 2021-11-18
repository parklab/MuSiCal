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
        for i in range(0, model.n_grid):
            W_s_this = model.W_s_all[i]
            H_s_this = model.H_s_all[i]
            print(W_s_this.shape)
            print(H_s_this.shape)
            
            X_simul_this = simulate_count_matrix(W_s_this, H_s_this)
            print(X_simul_this.shape)
            model_simul = model.clone_model(X_simul_this, grid_index = i)
            if use_refit:
                model_simul.W = model.W_s_all[i]
                model_simul.run_reassign()
                H_simul_reordered, reordered_indices, pdist = match_catalog_pair(model.H_s_all[i].T, model_simul.H_s.T, metric = metric_dist)
                error_W_all.append(beta_divergence(model.W_s_all[i], model_simul.W_s, beta=1))
                error_H_all.append(beta_divergence(model.H_s_all[i], model_simul.H_s, beta=1))
            else:
                model_simul.fit()
                W_simul_reordered, reordered_indices, pdist = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
                model_simul.W = W_simul_reordered
                model_simul.H = model_simul.H[reordered_indices,:]
                error_W_all.append(beta_divergence(model.W, model_simul.W, beta=1))
                error_H_all.append(beta_divergence(model.H, model_simul.H, beta=1))

            if save_models: 
                list_models_simul[i] = model_simul

            # pdist is the distance between the signatures if use_refit = False
            # otherwise it is the distance between the exposures
            if use_refit:
                _,_,pdist = match_catalog_pair(model.H_s.T, model_simul.H_s.T, metric = metric_dist)
            else: 
                _,_,pdist = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
            dists_per_sig_this = np.diagonal(pdist)
            inds_max = np.where(np.max(dists_per_sig_this) == dists_per_sig_this)
            dist_max_all.append(np.max(dists_per_sig_this))
            dist_max_sig_index_all.append(inds_max)

            dist_W_all[i] = pdist 
            dists_per_sig_all[i] = dists_per_sig_this

            X_simul_all[i] = X_simul_this
            if use_refit:
                W_simul_all[i] = model_simul.W_s
                H_simul_all[i] = model_simul.H_s
            else:
                W_simul_all[i] = model_simul.W
                H_simul_all[i] = model_simul.H
  
        best_grid_index = dist_max_all.index(min(dist_max_all))
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
        X_simul = simulate_count_matrix(W_s, H_s)
        model_simul = model.clone_model(X_simul, grid_index = 1) 
        if use_refit:
            model_simul.W = model.W_s
            model_simul.run_reassign()
            H_simul_reordered, reordered_indices, pdist = match_catalog_pair(model.H_s.T, model_simul.H_s.T, metric = metric_dist)
            error_W = beta_divergence(model.W_s, model_simul.W_s, beta=1)
            error_H = beta_divergence(model.H_s, model_simul.H_s, beta=1)
            W_simul = model_simul.W_s
            H_simul = model_simul.H_s
        else:
            model_simul.fit()
            W_simul_reordered, reordered_indices, pdist = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
            model_simul.W = W_simul_reordered
            model_simul.H = model_simul.H[reordered_indices,:]
            error_W = beta_divergence(model.W, model_simul.W, beta=1)
            error_H = beta_divergence(model.H, model_simul.H, beta=1)
            W_simul = model_simul.W
            H_simul = model_simul.H

        if use_refit:
            _,_,pdist = match_catalog_pair(model.H_s.T, model_simul.H_s.T, metric = metric_dist)
        else: 
             _,_,pdist = match_catalog_pair(model.W, model_simul.W, metric = metric_dist)
        dists_per_sig_this = np.diagonal(pdist)
        inds_max = np.where(np.max(dists_per_sig_this) == dists_per_sig_this)
        dist_max = np.max(dists_per_sig_this)
        dist_max_sig_index = inds_max
        dist_W = pdist
    return W_simul, H_simul, X_simul, best_grid_index, error_W, error_H, dist_W, dist_max, dist_max_sig_index,  dist_max_all, dist_max_sig_index_all, W_simul_all, H_simul_all, X_simul_all, error_W_all, error_H_all, dist_W_all
