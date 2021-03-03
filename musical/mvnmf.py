"""Min-volume non-negative matrix factorization

TODO
----------
1. Parallelize wrappedMVNMF. The problem is that, DenovoSig already parallelizes
    multiple runs of wrappedMVNMF. If inside wrappedMVNMF there is also parallelization,
    then there will be problems. I'm not sure if there is a workaround. 
"""

import numpy as np
from sklearn.preprocessing import normalize
import scipy.stats as stats
import warnings

from .utils import beta_divergence, normalize_WH, _samplewise_error
from .initialization import initialize_nmf


EPSILON = np.finfo(np.float32).eps

def _solve_mvnmf_matlab(X, W, H, eng=None, lambda_tilde=0.001, delta=0.1, max_iter=100, gamma=1.0):
    """Mvnmf solver

    Solve mvnmf using the matlab code betaNMF_logdetKL.
    """
    ### Matlab engine
    import matlab.engine
    import importlib.resources
    with importlib.resources.path('musical', 'matlab') as filepath:
        MATLABPATH = str(filepath)
    if type(eng) is not matlab.engine.matlabengine.MatlabEngine:
        eng = matlab.engine.start_matlab()
    eng.addpath(MATLABPATH, nargout=0)

    ### Solver
    n_features, n_samples = X.shape
    n_components = W.shape[1]
    delta = float(delta) # Necessary for matlab code
    lambda_tilde = float(lambda_tilde)
    gamma = float(gamma)

    W_ML = matlab.double(W.tolist())
    H_ML = matlab.double(H.tolist())
    X_ML = matlab.double(X.tolist())
    alpha = 0.0
    W_ML, H_ML, losses_ML, t, reconstruction_errors_ML, volumes_ML, line_search_steps_ML, gammas_ML, Lambda = eng.betaNMF_logdetKL(X_ML, W_ML, H_ML, n_components, max_iter, lambda_tilde, delta, gamma, alpha, nargout=9)
    W = np.asarray(W_ML._data).reshape(W_ML.size, order='F')
    H = np.asarray(H_ML._data).reshape(H_ML.size, order='F')
    losses = np.asarray(losses_ML._data)
    reconstruction_errors = np.asarray(reconstruction_errors_ML._data)
    volumes = np.asarray(volumes_ML._data)
    line_search_steps = np.asarray(line_search_steps_ML._data)
    gammas = np.asarray(gammas_ML._data)

    return W, H, Lambda, losses, reconstruction_errors, volumes, line_search_steps, gammas


def _volume_logdet(W, delta):
    K = W.shape[1]
    volume = np.log10(np.linalg.det(W.T @ W + delta * np.eye(K)))
    return volume


def _loss_mvnmf(X, W, H, Lambda, delta):
    reconstruction_error = beta_divergence(X, W @ H, beta=1, square_root=False)
    volume = _volume_logdet(W, delta)
    loss = reconstruction_error + Lambda * volume
    return loss, reconstruction_error, volume


def _solve_mvnmf(X, W, H, lambda_tilde=1e-5, delta=1.0, gamma=1.0,
                 max_iter=200, min_iter=100, tol=1e-4,
                 conv_test_freq=10, conv_test_baseline=None, verbose=0):
    """Mvnmf solver

    Python version of _solve_mvnmf_matlab().

    There are several differences between our implementation and the matlab code.
    1. After updating W according to equation (3.8) in 2019 Leplat, we do a simple renormalization
        of W and H, such that W lies on the simplex. Accordingly, in the backtracking line search part,
        we also do this simple renormalization. By comparison, in the matlab code, a projection onto the
        simplex is performed for W, which does not really make sense to me. The projection changes the W
        vectors and thus breaks the update rule. In my test runs, our code performs better than the matlab
        code. The matlab code still ensures that the objective is non-increasing. However, that seems to be
        achieved mostly by the backtracking line search part instead of the MU update part, which can be seen
        from the fact that the matlab code needs significantly more line search steps than our python code.
    2. In the update according to equation (3.8), the matlab code introduced EPSILON in the denominators. I
        don't think that is necessary, since we are already clipping out small values in W and H at each iteration.
    3. We do an initial renormalization of the input W and H, such that W satisfies the normalization criterion.
        Note that this makes the lambda_tilde values un-comparable between the matlab code and our python code.

    Parameters
    ----------
    X : array-like of shape (n_features, n_samples)
        Constant input matrix.

    W : array-like of shape (n_features, n_components)
        Initial guess.

    H : array-like of shape (n_components, n_samples)
        Initial guess.

    lambda_tilde : float
        Hyperparameter.

    delta : float
        Hyperparameter.

    gamma : float
        Initial step size for backtracking line search. Should be between 0 and 1. If -1,
        then backtracking line search is skipped.

    max_iter : int, default=200
        Maximum number of iterations.

    min_iter : int, default=100
        Minimum number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    conv_test_freq : int, default=10
        Convergence test frequency. Convergence test is performed every conv_test_freq iterations.

    conv_test_baseline : float, default=None
        Baseline for convergence test. If None, the initial loss is taken as the baseline.

    verbose : int, default=0
        Verbosity level.

    Notes
    ----------
    1. The algorithm should work better when the initial guesses are better. One reason lies in Lambda and lambda_tilde.
        Lambda is calculated in a way such that the two terms in the objective function is comparable. Ideally, Lambda
        should be set to beta_divergence(X, W_true @ H_true)/abs(volume(W_true)) * lambda_tilde. In our code, the true W
        and H are replaced by the initial guesses. So if the initial guesses are good, then indeed the two terms will be
        comparable. If the initial guesses are far off, then the beta_divergence part will be far over-estimated. As a result,
        the two terms are not comparable anymore. One potential improvement is to first run a small number of NMF iterations,
        and then use the NMF results as hot starts for the mvNMF algorithm.

    TODO
    ----------
    1. Remove codes for tracking the losses. Those are only for test purposes.
    2. Do better when backtracking line search is stuck. E.g., increase the lower limit of gamma. When it's reached, stop iteration.
        E.g., set a max number of steps for line search. When it's reached, stop iteration.
    """
    # Convert to float if they are not
    # Convert to np array in case they are not, e.g., when they are pd DataFrames.
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    if (type(W) != np.ndarray) or (not np.issubdtype(W.dtype, np.floating)):
        W = np.array(W).astype(float)
    if (type(H) != np.ndarray) or (not np.issubdtype(H.dtype, np.floating)):
        H = np.array(H).astype(float)
    n_features, n_samples = X.shape
    n_components = W.shape[1]
    # Redefine dimensions to be consistent with Algorithm 1 in 2019 Laplat.
    M = n_features
    T = n_samples
    K = n_components
    ##############################################
    #### Algorithm
    ##############################################
    # First normalize W
    W, H = normalize_WH(W, H)
    # Clip small values: important.
    W = W.clip(EPSILON)
    H = H.clip(EPSILON)
    # Calculate Lambda from labmda_tilde
    reconstruction_error = beta_divergence(X, W @ H, beta=1, square_root=False)
    volume = _volume_logdet(W, delta)
    Lambda = lambda_tilde * reconstruction_error / abs(volume)
    loss = reconstruction_error + Lambda * volume
    # Useful constants
    ones = np.ones((M, T))
    # Baseline of convergence test
    if conv_test_baseline is None:
        conv_test_baseline = loss
    elif type(conv_test_baseline) is str and conv_test_baseline == 'min-iter':
        pass
    else:
        conv_test_baseline = float(conv_test_baseline)
    # Loop
    losses = [loss]
    reconstruction_errors = [reconstruction_error]
    volumes = [volume]
    line_search_steps = []
    gammas = [gamma]
    loss_previous = loss # Loss in the last iteration
    loss_previous_conv_test = loss # Loss in the last convergence test
    converged = False
    for n_iter in range(1, max_iter + 1):
        # Update H according to 2001 Lee
        H = H * ( ( W.T @ (X/(W @ H)) ) / (W.T @ ones) )
        H = H.clip(EPSILON)
        # Update W
        Y = np.linalg.inv(W.T @ W + delta * np.eye(K))
        Y_plus = np.maximum(Y, 0)
        Y_minus = np.maximum(-Y, 0)
        JHT = ones @ H.T
        LWYm = Lambda * (W @ Y_minus)
        LWY = Lambda * (W @ (Y_plus + Y_minus))
        numerator = ( (JHT - 4 * LWYm)**2 + 8 * LWY * ((X/(W @ H)) @ H.T) )**0.5 - JHT + 4 * LWYm
        denominator = 4 * LWY
        Wup = W * (numerator / denominator)
        Wup = Wup.clip(EPSILON)
        # Backtracking line search for W
        if gamma != -1:
            W_new = (1 - gamma) * W + gamma * Wup
            W_new, H_new = normalize_WH(W_new, H)
            W_new = W_new.clip(EPSILON)
            H_new = H_new.clip(EPSILON)
            loss, reconstruction_error, volume = _loss_mvnmf(X, W_new, H_new, Lambda, delta)
            line_search_step = 0
            while (loss > loss_previous) and (gamma > 1e-16):
                gamma = gamma * 0.8
                W_new = (1 - gamma) * W + gamma * Wup
                W_new, H_new = normalize_WH(W_new, H)
                W_new = W_new.clip(EPSILON)
                H_new = H_new.clip(EPSILON)
                loss, reconstruction_error, volume = _loss_mvnmf(X, W_new, H_new, Lambda, delta)
                line_search_step += 1
            W = W_new
            H = H_new
        else:
            line_search_step = 0
            W = Wup
            W, H = normalize_WH(W, H)
            W = W.clip(EPSILON)
            H = H.clip(EPSILON)
        line_search_steps.append(line_search_step)
        # Update gamma
        if gamma != -1:
            gamma = min(gamma*2.0, 1.0)
        gammas.append(gamma)
        # Losses
        loss, reconstruction_error, volume = _loss_mvnmf(X, W, H, Lambda, delta)
        losses.append(loss)
        reconstruction_errors.append(reconstruction_error)
        volumes.append(volume)
        loss_previous = loss
        # Convergence test
        if n_iter == min_iter and conv_test_baseline == 'min-iter':
            conv_test_baseline = loss
        if n_iter >= min_iter and tol > 0 and n_iter % conv_test_freq == 0:
            relative_loss_change = (loss_previous_conv_test - loss) / conv_test_baseline
            if (loss <= loss_previous_conv_test) and (relative_loss_change <= tol):
                converged = True
            else:
                converged = False
            if verbose:
                print('Epoch %02d reached. Loss: %.3g. Loss in the previous convergence test: %.3g. '
                      'Baseline: %.3g. Relative loss change: %.3g' %
                      (n_iter, loss, loss_previous_conv_test, conv_test_baseline, relative_loss_change))
            loss_previous_conv_test = loss
        # If converged, stop
        if converged and n_iter >= min_iter:
            break

    losses = np.array(losses)
    reconstruction_errors = np.array(reconstruction_errors)
    volumes = np.array(volumes)
    line_search_steps = np.array(line_search_steps)
    gammas = np.array(gammas)

    return W, H, n_iter, converged, Lambda, losses, reconstruction_errors, volumes, line_search_steps, gammas


class MVNMF:
    """A single run of mvNMF

    Notes
    ----------
    1. I removed eng from __init__ and did not set eng as an attribute. Otherwise pickle will have
    a problem when saving the class instance, because pickle does not deal with matlab well.
    """
    def __init__(self,
                 X,
                 n_components,
                 init='random',
                 init_W_custom=None,
                 init_H_custom=None,
                 lambda_tilde=1e-5,
                 delta=1.0,
                 gamma=1.0,
                 max_iter=200,
                 min_iter=100,
                 tol=1e-4,
                 conv_test_freq=10,
                 conv_test_baseline=None,
                 verbose=0#,
                 #eng=None
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = n_components
        self.init = init
        if init_W_custom is not None:
            if (type(init_W_custom) != np.ndarray) or (not np.issubdtype(init_W_custom.dtype, np.floating)):
                init_W_custom = np.array(init_W_custom).astype(float)
        if init_H_custom is not None:
            if (type(init_H_custom) != np.ndarray) or (not np.issubdtype(init_H_custom.dtype, np.floating)):
                init_H_custom = np.array(init_H_custom).astype(float)
        self.init_W_custom = init_W_custom
        self.init_H_custom = init_H_custom
        self.lambda_tilde = lambda_tilde
        self.delta = delta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.verbose = verbose
        #self.eng = eng

    def fit(self, eng=None):
        W_init, H_init = initialize_nmf(self.X, self.n_components,
                                        init=self.init,
                                        init_W_custom=self.init_W_custom,
                                        init_H_custom=self.init_H_custom,
                                        eng=eng)
        self.W_init = W_init
        self.H_init = H_init

        (W, H, n_iter, converged, Lambda, losses, reconstruction_errors,
            volumes, line_search_steps, gammas) = _solve_mvnmf(
            X=self.X, W=self.W_init, H=self.H_init, lambda_tilde=self.lambda_tilde,
            delta=self.delta, gamma=self.gamma, max_iter=self.max_iter,
            min_iter=self.min_iter, tol=self.tol,
            conv_test_freq=self.conv_test_freq,
            conv_test_baseline=self.conv_test_baseline,
            verbose=self.verbose)
        self.W = W
        self.H = H
        self.n_iter = n_iter
        self.converged = converged
        self.Lambda = Lambda
        #self.loss_track = losses
        #self.reconstruction_error_track = reconstruction_errors
        #self.volume_track = volumes
        #self.line_search_step_track = line_search_steps
        #self.gamma_track = gammas
        self.loss = losses[-1]
        self.reconstruction_error = reconstruction_errors[-1]
        self.volume = volumes[-1]

        return self


class wrappedMVNMF:
    """mvNMF with automatic selection of lambda_tilde.

    Notes
    ----------
    1. I removed eng from __init__ and did not set eng as an attribute. Otherwise pickle will have
    a problem when saving the class instance, because pickle does not deal with matlab well.
    """
    def __init__(self,
                 X,
                 n_components,
                 lambda_tilde_grid=None,
                 pthresh=0.05,
                 init='random',
                 init_W_custom=None,
                 init_H_custom=None,
                 delta=1.0,
                 gamma=1.0,
                 max_iter=200,
                 min_iter=100,
                 tol=1e-4,
                 conv_test_freq=10,
                 conv_test_baseline=None,
                 verbose=0#,
                 #eng=None
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = n_components
        if lambda_tilde_grid is None:
            lambda_tilde_grid = np.array([1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])
        else:
            lambda_tilde_grid = np.array(lambda_tilde_grid)
        self.lambda_tilde_grid = lambda_tilde_grid
        self.pthresh = pthresh
        self.init = init
        if init_W_custom is not None:
            if (type(init_W_custom) != np.ndarray) or (not np.issubdtype(init_W_custom.dtype, np.floating)):
                init_W_custom = np.array(init_W_custom).astype(float)
        if init_H_custom is not None:
            if (type(init_H_custom) != np.ndarray) or (not np.issubdtype(init_H_custom.dtype, np.floating)):
                init_H_custom = np.array(init_H_custom).astype(float)
        self.init_W_custom = init_W_custom
        self.init_H_custom = init_H_custom
        self.delta = delta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.verbose = verbose
        #self.eng = eng

    def fit(self, eng=None):
        ##################################################
        ################# Initialization #################
        ##################################################
        W_init, H_init = initialize_nmf(self.X, self.n_components,
                                        init=self.init,
                                        init_W_custom=self.init_W_custom,
                                        init_H_custom=self.init_H_custom,
                                        eng=eng)
        self.W_init = W_init
        self.H_init = H_init

        ##################################################
        ################### Run mvNMF ####################
        ##################################################
        # Need to parallelize this part.
        models = []
        for lambda_tilde in self.lambda_tilde_grid:
            if self.verbose:
                print('==============================================')
                print('Running mvNMF with lambda_tilde = %.5g......' % lambda_tilde)
            model = MVNMF(self.X, self.n_components, init='custom',
                          init_W_custom=self.W_init, init_H_custom=self.H_init,
                          lambda_tilde=lambda_tilde, delta=self.delta, gamma=self.gamma,
                          max_iter=self.max_iter, min_iter=self.min_iter, tol=self.tol,
                          conv_test_freq=self.conv_test_freq, conv_test_baseline=self.conv_test_baseline,
                          verbose=self.verbose)
            model.fit(eng=eng)
            models.append(model)
        self.Lambda_grid = np.array([model.Lambda for model in models])
        self.loss_grid = np.array([model.loss for model in models])
        self.reconstruction_error_grid = np.array([model.reconstruction_error for model in models])
        self.volume_grid = np.array([model.volume for model in models])
        self.model_grid = models

        ###################################################
        ############# Select the best model ###############
        ###################################################
        # First calculate sample-wise errors
        self.samplewise_reconstruction_errors_grid = np.array([
            _samplewise_error(self.X, model.W @ model.H) for model in models
        ])
        # Then perform statistical tests
        # Alternative tests we can use: ks_2samp, ttest_ind (perhaps on log errors)
        #self.pvalue_grid = np.array([
        #    stats.mannwhitneyu(self.samplewise_reconstruction_errors_grid[i, :],
        #                       self.samplewise_reconstruction_errors_grid[i+1, :],
        #                       alternative='less')[1] for i in range(0, len(self.lambda_tilde_grid) - 1)
        #])
        self.pvalue_grid = np.array([
            stats.ttest_ind(self.samplewise_reconstruction_errors_grid[i, :],
                            self.samplewise_reconstruction_errors_grid[i+1, :],
                            alternative='less')[1] for i in range(0, len(self.lambda_tilde_grid) - 1)
        ])
        # Select the best model
        if np.min(self.pvalue_grid) > self.pthresh:
            warnings.warn('No p-value is smaller than or equal to %.3g. The largest lambda_tilde is selected. Enlarge the search grid of lambda_tilde.' % self.pthresh,
                          UserWarning)
            index_selected = len(self.pvalue_grid) - 1
        else:
            index_selected = np.argmax(self.pvalue_grid <= self.pthresh)
        self.lambda_tilde = self.lambda_tilde_grid[index_selected]
        self.model = models[index_selected]
        self.W = self.model.W
        self.H = self.model.H
        self.Lambda = self.model.Lambda
        self.loss = self.model.loss
        self.reconstruction_error = self.model.reconstruction_error
        self.volume = self.model.volume

        return self
