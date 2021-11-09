"""Sparse nnls"""

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import pairwise_distances
import scipy.stats as stats
import warnings
from sklearn.preprocessing import normalize


def _fill_vector(x, indices, L):
    if len(x) != len(indices):
        raise ValueError('x and indices are not of the same length.')
    x_filled = np.zeros(L)
    x_filled[indices] = x
    return x_filled


def nnls_thresh_naive(x, W, thresh=0.05, thresh_agnostic=0.0):
    """Naive thresholded nnls

    An initial NNLS is first done. Based on the initial result,
    signatures with normalized exposure < thresh or with absolute exposure
    < thresh_agnostic/(max of signature) are removed. Then a final NNLS is done
    to recalculate the exposure.
    """
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ###
    if thresh_agnostic > 0:
        W_max = np.max(W, 0)
        h, _ = sp.optimize.nnls(W, x)
        h_normalized = h/np.sum(h)
        indices_retained = indices_all[(h_normalized >= thresh) & (h >= thresh_agnostic/W_max)]
    else:
        h, _ = sp.optimize.nnls(W, x)
        h_normalized = h/np.sum(h)
        indices_retained = indices_all[h_normalized >= thresh]
    ###
    if len(indices_retained) == 0:
        indices_retained = np.array([np.argmax(h)])
    h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    h = _fill_vector(h, indices_retained, n_sigs)
    return h


def nnls_thresh(x, W, thresh=0.05, thresh_agnostic=0.0):
    """Thresholded nnls

    An initial NNLS is first done. Based on the initial result,
    signatures with normalized exposure < thresh or with absolute exposure
    < thresh_agnostic/(max of signature) are removed. Then this procedure is
    repeated with only the retained signatures, until all signatures satisfy
    the condition that normalized exposxure >= thresh and absolute exposure >=
    thresh_agnostic/(max of signature).
    """
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ###
    if thresh_agnostic > 0:
        W_max = np.max(W, 0)
        h, _ = sp.optimize.nnls(W, x)
        h_normalized = h/np.sum(h)
        indices_retained = indices_all[(h_normalized >= thresh) & (h >= thresh_agnostic/W_max)]
        if len(indices_retained) == 0:
            indices_retained = np.array([np.argmax(h)])
        while True:
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            h_normalized = h/np.sum(h)
            indices_retained_next = indices_retained[(h_normalized >= thresh) & (h >= thresh_agnostic/W_max[indices_retained])]
            if len(indices_retained_next) == 0:
                indices_retained_next = np.array([indices_retained[np.argmax(h)]])
            if np.array_equal(indices_retained_next, indices_retained):
                break
            indices_retained = indices_retained_next
    else:
        h, _ = sp.optimize.nnls(W, x)
        h_normalized = h/np.sum(h)
        indices_retained = indices_all[h_normalized >= thresh]
        if len(indices_retained) == 0:
            indices_retained = np.array([np.argmax(h)])
        while True:
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            h_normalized = h/np.sum(h)
            indices_retained_next = indices_retained[h_normalized >= thresh]
            if len(indices_retained_next) == 0:
                indices_retained_next = np.array([indices_retained[np.argmax(h)]])
            if np.array_equal(indices_retained_next, indices_retained):
                break
            indices_retained = indices_retained_next
    h = _fill_vector(h, indices_retained, n_sigs)
    return h


def _multinomial_loglikelihood(x, p, epsilon=1e-16, per_trial=True):
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
    p = p.clip(epsilon)
    p = p/np.sum(p)
    if per_trial:
        return np.sum(x*np.log(p))/np.sum(x)
    else:
        return np.sum(x*np.log(p))


def nnls_likelihood_backward(x, W, thresh=0.001, per_trial=True):
    """Likelihood NNLS with backward stepwise trimming

    An initial NNLS is first done. Then we trim the exposure vector in a
    backward stepwise manner. At each step, we calculate the multinomial
    log likelihood ratio (logLR) between the full model with all currently
    included signatures and the model where one signature is excluded,
    and then remove the signature with the smallest logLR. We repeat this
    process until all logLRs are above a certain threshold.
    ----------
    Notes:
    1. Thresh is in the scale of log likelihood ratio between the full model
    and the full model minus one signature. Greater thresh means more sparsity.
    Thresh should be nonnegative.
    """
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ### Initial NNLS
    h, _ = sp.optimize.nnls(W, x)
    indices_retained = indices_all[h > 0]
    ### Backward trimming loop
    while True:
        if len(indices_retained) == 1:
            break
        else:
            # Log likelihood of current full model
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            p_current = W[:, indices_retained] @ h
            p_current = p_current/np.sum(p_current)
            loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
            loglikelihoods = []
            # Log likelihoods of model that removes 1 signature
            for index in indices_retained:
                _indices = np.array([i for i in indices_retained if i != index])
                p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
                p = p/np.sum(p)
                loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
            loglikelihoods = np.array(loglikelihoods)
            # Log likelihood ratios
            loglikelihoods = loglikelihood_current - loglikelihoods
            # Test
            if np.min(loglikelihoods) >= thresh:
                break
            else:
                index_remove = indices_retained[np.argmin(loglikelihoods)]
                indices_retained = np.array([i for i in indices_retained if i != index_remove])
    ### Final NNLS
    h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    h = _fill_vector(h, indices_retained, n_sigs)
    return h


def nnls_likelihood_bidirectional(x, W, thresh_backward=0.001, thresh_forward=0.002, max_iter=1000, per_trial=True):
    """Likelihood NNLS with both backward and forward stepwise rountines.

    Notes:
    1. thresh_forward should be greater than thresh_backward. Otherwise the
    loop may run into a dead loop where the same signature is being removed
    and added back within one iteration, although this is caught gracefully in
    the code.
    2. Both thresh_backward and thresh_forward should be nonnegative.
    """
    if thresh_backward >= thresh_forward:
        warnings.warn('thresh_backward is not smaller than thresh_forward. This might lead to indefinite loops.', UserWarning)
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ### Initial NNLS
    h, _ = sp.optimize.nnls(W, x)
    indices_retained = indices_all[h > 0]
    indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
    ### Didirectional loop
    i_iter = 0
    while i_iter < max_iter:
        i_iter += 1
        ########################## Backward ##########################
        if len(indices_retained) == 1:
            backward_stop = True
        else:
            # Log likelihood of current full model
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            p_current = W[:, indices_retained] @ h
            p_current = p_current/np.sum(p_current)
            loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
            loglikelihoods = []
            # Log likelihoods of model that removes 1 signature
            for index in indices_retained:
                _indices = np.array([i for i in indices_retained if i != index])
                p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
                p = p/np.sum(p)
                loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
            loglikelihoods = np.array(loglikelihoods)
            # Log likelihood ratios
            loglikelihoods = loglikelihood_current - loglikelihoods
            # Test
            if np.min(loglikelihoods) >= thresh_backward:
                backward_stop = True
                #print(i_iter, 'Remove', backward_stop, None, indices_retained)
            else:
                backward_stop = False
                index_remove = indices_retained[np.argmin(loglikelihoods)]
                indices_retained = np.array([i for i in indices_retained if i != index_remove])
                #print(i_iter, 'Remove', backward_stop, index_remove, indices_retained)
        ########################## Forward ##########################
        indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
        if len(indices_others) == 0:
            forward_stop = True
        else:
            # Log likelihood of current full model
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            p_current = W[:, indices_retained] @ h
            p_current = p_current/np.sum(p_current)
            loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
            loglikelihoods = []
            # Log likelihoods of model that adds 1 signature
            for index in indices_others:
                _indices = np.sort(np.append(indices_retained, index))
                p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
                p = p/np.sum(p)
                loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
            loglikelihoods = np.array(loglikelihoods)
            # Log likelihood ratios
            loglikelihoods = loglikelihoods - loglikelihood_current
            # Test
            if np.max(loglikelihoods) <= thresh_forward:
                forward_stop = True
                #print(i_iter, 'Add', forward_stop, None, indices_retained)
            else:
                forward_stop = False
                index_add = indices_others[np.argmax(loglikelihoods)]
                indices_retained = np.sort(np.append(indices_retained, index_add))
                #print(i_iter, 'Add', forward_stop, index_add, indices_retained)
        ######################## Stopping criterion ########################
        if backward_stop and forward_stop:
            break
        if not backward_stop and not forward_stop and index_remove == index_add:
            warnings.warn('The same signature is being removed and added back within one iteration, suggesting ill convergence.',
                          UserWarning)
            break
    if i_iter >= max_iter:
        warnings.warn('Max_iter reached, suggesting that the problem may not converge. Or try increasing max_iter.',
                      UserWarning)
    ### Final NNLS
    h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    h = _fill_vector(h, indices_retained, n_sigs)
    return h


def nnls_likelihood_backward_relaxed(x, W, thresh=0.001, per_trial=True):
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ### Initial NNLS
    h, _ = sp.optimize.nnls(W, x)
    indices_retained = indices_all[h > 0]
    ### Backward trimming loop
    while True:
        if len(indices_retained) == 1:
            break
        else:
            # Log likelihood of current full model
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            p_current = W[:, indices_retained] @ h
            p_current = p_current/np.sum(p_current)
            loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
            loglikelihoods = []
            # Log likelihoods of model that removes 1 signature
            for index in indices_retained:
                _indices = np.array([i for i in indices_all if i != index]) # !!!!!! Difference in here.
                p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
                p = p/np.sum(p)
                loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
            loglikelihoods = np.array(loglikelihoods)
            # Log likelihood ratios
            loglikelihoods = loglikelihood_current - loglikelihoods
            # Test
            if np.min(loglikelihoods) >= thresh:
                break
            else:
                index_remove = indices_retained[np.argmin(loglikelihoods)]
                indices_retained = np.array([i for i in indices_retained if i != index_remove])
    ### Final NNLS
    h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    h = _fill_vector(h, indices_retained, n_sigs)
    return h


def nnls_likelihood_bidirectional_relaxed(x, W, thresh_backward=0.001, thresh_forward=0.002, max_iter=1000, per_trial=True):
    if thresh_backward >= thresh_forward:
        warnings.warn('thresh_backward is not smaller than thresh_forward. This might lead to indefinite loops.', UserWarning)
    n_sigs = W.shape[1]
    indices_all = np.arange(0, n_sigs)
    ### Initial NNLS
    h, _ = sp.optimize.nnls(W, x)
    indices_retained = indices_all[h > 0]
    indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
    ### Didirectional loop
    i_iter = 0
    while i_iter < max_iter:
        i_iter += 1
        ########################## Backward ##########################
        if len(indices_retained) == 1:
            backward_stop = True
        else:
            # Log likelihood of current full model
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            p_current = W[:, indices_retained] @ h
            p_current = p_current/np.sum(p_current)
            loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
            loglikelihoods = []
            # Log likelihoods of model that removes 1 signature
            for index in indices_retained:
                _indices = np.array([i for i in indices_all if i != index]) # !!!!!! Difference in here.
                p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
                p = p/np.sum(p)
                loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
            loglikelihoods = np.array(loglikelihoods)
            # Log likelihood ratios
            loglikelihoods = loglikelihood_current - loglikelihoods
            # Test
            if np.min(loglikelihoods) >= thresh_backward:
                backward_stop = True
                #print(i_iter, 'Remove', backward_stop, None, indices_retained)
            else:
                backward_stop = False
                index_remove = indices_retained[np.argmin(loglikelihoods)]
                indices_retained = np.array([i for i in indices_retained if i != index_remove])
                #print(i_iter, 'Remove', backward_stop, index_remove, indices_retained)
        ########################## Forward ##########################
        indices_others = np.array(sorted(list(set(indices_all) - set(indices_retained))))
        if len(indices_others) == 0:
            forward_stop = True
        else:
            # Log likelihood of current full model
            h, _ = sp.optimize.nnls(W[:, indices_retained], x)
            p_current = W[:, indices_retained] @ h
            p_current = p_current/np.sum(p_current)
            loglikelihood_current = _multinomial_loglikelihood(x, p_current, per_trial=per_trial)
            loglikelihoods = []
            # Log likelihoods of model that adds 1 signature
            for index in indices_others:
                _indices = np.sort(np.append(indices_retained, index))
                p = W[:, _indices] @ sp.optimize.nnls(W[:, _indices], x)[0]
                p = p/np.sum(p)
                loglikelihoods.append(_multinomial_loglikelihood(x, p, per_trial=per_trial))
            loglikelihoods = np.array(loglikelihoods)
            # Log likelihood ratios
            loglikelihoods = loglikelihoods - loglikelihood_current
            # Test
            if np.max(loglikelihoods) <= thresh_forward:
                forward_stop = True
                #print(i_iter, 'Add', forward_stop, None, indices_retained)
            else:
                forward_stop = False
                index_add = indices_others[np.argmax(loglikelihoods)]
                indices_retained = np.sort(np.append(indices_retained, index_add))
                #print(i_iter, 'Add', forward_stop, index_add, indices_retained)
        ######################## Stopping criterion ########################
        if backward_stop and forward_stop:
            break
        if not backward_stop and not forward_stop and index_remove == index_add:
            warnings.warn('The same signature is being removed and added back within one iteration, suggesting ill convergence.',
                          UserWarning)
            break
    if i_iter >= max_iter:
        warnings.warn('Max_iter reached, suggesting that the problem may not converge. Or try increasing max_iter.',
                      UserWarning)
    ### Final NNLS
    h, _ = sp.optimize.nnls(W[:, indices_retained], x)
    h = _fill_vector(h, indices_retained, n_sigs)
    return h


class SparseNNLS:
    def __init__(self,
                 method='likelihood_bidirectional',
                 thresh1=None,
                 thresh2=None,
                 max_iter=None,
                 per_trial=None,
                 N=None
                 ):
        self.method = method
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        self.max_iter = max_iter
        self.per_trial = per_trial
        self.N = N

    def fit(self, X, W):
        # Save original input data
        self.X_original = X
        self.W_original = W
        ##########
        # Convert input to appropriate data types.
        if isinstance(W, pd.DataFrame):
            self.W = W
            self.sigs_all = self.W.columns.values
        else:
            self.W = pd.DataFrame(W, columns=['Sig' + str(i) for i in range(0, W.shape[1])])
            self.sigs_all = self.W.columns.values
        if isinstance(X, pd.DataFrame):
            self.X = X
            self.samples = self.X.columns.values
        else:
            if len(X.shape) == 1:
                self.X = pd.DataFrame(np.reshape(X, (-1, 1)), columns=['Sample' + str(i) for i in range(0, 1)], index=self.W.index)
            else:
                self.X = pd.DataFrame(X, columns=['Sample' + str(i) for i in range(0, X.shape[1])], index=self.W.index)
            self.samples = self.X.columns.values
        ##########
        # If a rescale factor is used, rescale data.
        if self.N is not None:
            if type(self.N) is not int:
                raise ValueError('N must be an integer.')
            self._X_in = self.X * self.N
            self._X_in = self._X_in.round(0).astype(int)
        else:
            self._X_in = self.X
        ##########
        # NNLS
        if self.method == 'thresh_naive':
            if self.thresh1 is None:
                self.thresh1 = 0.05
            self.thresh = self.thresh1
            if self.thresh2 is None:
                self.thresh2 = 0.0
            self.thresh_agnostic = self.thresh2
            self.H = [
                nnls_thresh_naive(x, self.W.values, thresh=self.thresh, thresh_agnostic=self.thresh_agnostic) for x in self._X_in.T.values
            ]
        elif self.method == 'thresh':
            if self.thresh1 is None:
                self.thresh1 = 0.05
            self.thresh = self.thresh1
            if self.thresh2 is None:
                self.thresh2 = 0.0
            self.thresh_agnostic = self.thresh2
            self.H = [
                nnls_thresh(x, self.W.values, thresh=self.thresh, thresh_agnostic=self.thresh_agnostic) for x in self._X_in.T.values
            ]
        elif self.method == 'likelihood_backward':
            if self.thresh1 is None:
                self.thresh1 = 0.001
            self.thresh = self.thresh1
            if self.per_trial is None:
                self.per_trial = True
            if self.thresh2 is not None:
                warnings.warn('Method is chosen as likelihood_backward. The supplied thresh2 will not be used and thus is set to None.', UserWarning)
                self.thresh2 = None
            self.H = [
                nnls_likelihood_backward(x, self.W.values, thresh=self.thresh, per_trial=self.per_trial) for x in self._X_in.T.values
            ]
        elif self.method == 'likelihood_backward_relaxed':
            if self.thresh1 is None:
                self.thresh1 = 0.001
            self.thresh = self.thresh1
            if self.per_trial is None:
                self.per_trial = True
            if self.thresh2 is not None:
                warnings.warn('Method is chosen as likelihood_backward_relaxed. The supplied thresh2 will not be used and thus is set to None.', UserWarning)
                self.thresh2 = None
            self.H = [
                nnls_likelihood_backward_relaxed(x, self.W.values, thresh=self.thresh, per_trial=self.per_trial) for x in self._X_in.T.values
            ]
        elif self.method == 'likelihood_bidirectional':
            if self.thresh1 is None:
                self.thresh1 = 0.001
            self.thresh_backward = self.thresh1
            if self.thresh2 is None:
                self.thresh2 = 0.002
            self.thresh_forward = self.thresh2
            if self.max_iter is None:
                self.max_iter = 1000
            if self.per_trial is None:
                self.per_trial = True
            self.H = [
                nnls_likelihood_bidirectional(x, self.W.values, thresh_backward=self.thresh_backward, thresh_forward=self.thresh_forward, max_iter=self.max_iter, per_trial=self.per_trial) for x in self._X_in.T.values
            ]
        elif self.method == 'likelihood_bidirectional_relaxed':
            if self.thresh1 is None:
                self.thresh1 = 0.001
            self.thresh_backward = self.thresh1
            if self.thresh2 is None:
                self.thresh2 = 0.002
            self.thresh_forward = self.thresh2
            if self.max_iter is None:
                self.max_iter = 1000
            if self.per_trial is None:
                self.per_trial = True
            self.H = [
                nnls_likelihood_bidirectional_relaxed(x, self.W.values, thresh_backward=self.thresh_backward, thresh_forward=self.thresh_forward, max_iter=self.max_iter, per_trial=self.per_trial) for x in self._X_in.T.values
            ]
        else:
            raise ValueError('Invalid method for SparseNNLS.')
        ##########
        # If a rescale factor is used, convert back to the original scale.
        # A final NNLS is needed constraining to only active signatures.
        if self.N is not None:
            tmp = []
            n_sigs = len(self.sigs_all)
            indices_all = np.arange(0, n_sigs)
            for x, h in zip(self.X.T.values, self.H):
                indices_retained = indices_all[h > 0]
                h_new, _ = sp.optimize.nnls(self.W.iloc[:, indices_retained], x)
                h_new = _fill_vector(h_new, indices_retained, n_sigs)
                tmp.append(h_new)
            self.H = tmp
        ##########
        # Collect final data
        self.H = pd.DataFrame(np.array(self.H).T, index=self.W.columns, columns=self.X.columns)
        self.X_reconstructed = pd.DataFrame(np.array([self.W.values @ h for h in self.H.T.values]).T,
                                            columns=self.X.columns, index=self.X.index)
        self.cos_similarities = []
        self.matched_sigs = []
        self.coefficients = []
        for x, h, x_reconstructed in zip(self.X.T.values, self.H.T.values, self.X_reconstructed.T.values):
            self.matched_sigs.append(self.sigs_all[h > 0])
            self.coefficients.append(h[h > 0])
            self.cos_similarities.append(1 - sp.spatial.distance.cosine(x, x_reconstructed))

        self.sigs_reduced = self.H.index[self.H.sum(1) > 0].values
        self.W_reduced = pd.DataFrame.copy(self.W[self.sigs_reduced])
        self.H_reduced = pd.DataFrame.copy(self.H.loc[self.sigs_reduced])
        self.H_reduced_normalized = pd.DataFrame(normalize(self.H_reduced.values, norm='l1', axis=0), index=self.H_reduced.index, columns=self.H_reduced.columns)

        return self


class SparseNNLSGrid:
    def __init__(self,
                 method='likelihood_bidirectional',
                 thresh1_grid=None,
                 thresh2_grid=None,
                 max_iter=None,
                 per_trial=None,
                 N=None
                 ):
        self.method = method
        self.thresh1_grid = thresh1_grid
        self.thresh2_grid = thresh2_grid
        self.max_iter = max_iter
        self.per_trial = per_trial
        self.N = N

    def fit(self, X, W):
        self.X = X
        self.W = W
        ##########
        if self.method == 'thresh_naive':
            if self.thresh1_grid is None:
                self.thresh1_grid = np.arange(0.0, 0.201, 0.01)
            if self.thresh2_grid is None:
                self.thresh2_grid = np.array([0.0])
        elif self.method == 'thresh':
            if self.thresh1_grid is None:
                self.thresh1_grid = np.arange(0.0, 0.201, 0.01)
            if self.thresh2_grid is None:
                self.thresh2_grid = np.array([0.0])
        elif self.method == 'likelihood_backward':
            if self.thresh1_grid is None:
                self.thresh1_grid = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
            if self.thresh2_grid is None:
                self.thresh2_grid = np.array([None])
        elif self.method == 'likelihood_backward_relaxed':
            if self.thresh1_grid is None:
                self.thresh1_grid = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
            if self.thresh2_grid is None:
                self.thresh2_grid = np.array([None])
        elif self.method == 'likelihood_bidirectional':
            if self.thresh1_grid is None:
                self.thresh1_grid = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
            if self.thresh2_grid is None:
                self.thresh2_grid = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        elif self.method == 'likelihood_bidirectional_relaxed':
            if self.thresh1_grid is None:
                self.thresh1_grid = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
            if self.thresh2_grid is None:
                self.thresh2_grid = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        else:
            raise ValueError('Invalid method for SparseNNLSGrid.')

        self.models_grid = {}
        for thresh1 in self.thresh1_grid:
            for thresh2 in self.thresh2_grid:
                model = SparseNNLS(method=self.method, thresh1=thresh1, thresh2=thresh2,
                                   max_iter=self.max_iter, per_trial=self.per_trial, N=self.N)
                model.fit(self.X, self.W)
                self.models_grid[(thresh1, thresh2)] = model

        return self
