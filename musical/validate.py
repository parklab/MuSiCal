import numpy as np
import pickle
from .utils import beta_divergence, simulate_count_matrix, match_catalog_pair


def validate(model, validation_output_file, measure_dist = 'cosine'):

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
    if validation_output_file is not None:
       list_models_simul = []
       save_models = True
    else:
       save_models = False
    dist_W_all = {}
    W_simul_all = {}
    H_simul_all = {}
    X_simul_all = {}
    error_W_all = []
    error_H_all = []
    if model.n_grid > 1: 
        for i in range(0, model.n_grid):
            W_s_this = model.W_s_all[i]
            H_s_this = model.H_s_all[i]
            X_simul_this = simulate_count_matrix(W_s_this, H_s_this)
            model_simul = model.clone_model(X_simul_this, grid_index = i)
            model_simul.fit()
            W_simul_reordered, reordered_indices, pdist = match_catalog_pair(model.W, model_simul.W, measure = measure_dist)
            model_simul.W = W_simul_reordered
            model_simul.H = model_simul.H[reordered_indices,:]
            error_W_all.append(beta_divergence(model.W, model_simul.W, beta=1))
            error_H_all.append(beta_divergence(model.H, model_simul.H, beta=1))
            if save_models: 
                list_models_simul[i] = model_simul
            dist_W_all[i] = pdist
            X_simul_all[i] = X_simul_this
            W_simul_all[i] = model_simul.W
            H_simul_all[i] = model_simul.H
  
        best_grid_index = error_W_all.index(min(error_W_all))
        W_simul = W_simul_all[best_grid_index]
        H_simul = H_simul_all[best_grid_index]
        X_simul = X_simul_all[best_grid_index]
        error_W = error_W_all[best_grid_index]
        error_H = error_H_all[best_grid_index]
        dist_W = dist_W_all[best_grid_index]

        if save_models: # if output file is specified saves the list of simulated data 
            with open(validation_output_file, 'wb') as f:
                pickle.dump(list_models_simul, f, pickle.HIGHEST_PROTOCOL)

    else:  # if there was no grid search
        W_s = model.W_s
        H_s = model.H_s
        X_simul = simulate_count_matrix(W_s, H_s)
        model_simul = model.clone_model( X_simul, grid_index = 1) 
        model_simul.fit()
        W_simul_reordered, reordered_indices, pdist = match_catalog_pair(model.W, model_simul.W)
        model_simul.W = W_simul_reordered
        model_simul.H = model_simul.H[reordered_indices,:]
        error_W = beta_divergence(model.W, model_simul.W, beta=1)
        error_H = beta_divergence(model.H, model_simul.H, beta=1)
        W_simul = model_simul.W
        H_simul = model_simul.H

    return W_simul, H_simul, X_simul, best_grid_index, error_W, error_H, dist_W, W_simul_all, H_simul_all, X_simul_all, error_W_all, error_H_all, dist_W_all
