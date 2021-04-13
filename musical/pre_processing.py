import numpy as np

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def remove_samples_based_on_gini(H, X, gini_baseline = 0.65, gini_delta = 0.005, per_signature = True):
    gini_vec = []
    gini_vec = np.array(gini_vec)
    for h in H:
        h_norm = h/np.sum(X, axis = 0)
        gini_this = gini(h_norm)
        gini_vec = np.append(gini_vec, gini_this)

    inds_columns_to_check = np.where(gini_vec > gini_baseline)

    indices_to_remove = []
    indices_to_remove = np.array(inds_columns_to_check)

    list_indices_to_keep = {}
#    for i in np.array(inds_columns_to_check).tolist()[0]:
    for i in inds_columns_to_check[0]:
        h_norm = H[i,:]/np.sum(X, axis = 0)
        index = h_norm.size
        delta = 1
        while delta > gini_delta:
            gini_this = gini(np.sort(h_norm)[1:index])
            gini_bef = gini(np.sort(h_norm)[1:(index - 1)])
            delta = gini_this - gini_bef
            index = index - 1
            if index < np.around(h_norm.size * 0.8): 
                break
        to_keep = np.where(h_norm < np.sort(h_norm)[index])
        to_remove = np.where(h_norm >= np.sort(h_norm)[index])
        list_indices_to_keep[i] = to_keep
        indices_to_remove = np.append(indices_to_remove, to_remove)

    indices_to_remove = np.unique(indices_to_remove)

    if per_signature:
        list_X = {}
        for i in inds_columns_to_check[0]:
            X_this = X[:,list_indices_to_keep[i][0]]
            list_X[i] = X_this
        return(list_X)
    else:
        X_this = np.delete(X, indices_to_remove, axis = 1)
        return(X_this)


    
