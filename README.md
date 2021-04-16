# MuSiCal - Mutational Signature Calculator

MuSiCal provides functionalities for both de-novo extraction of mutational signatures and refitting to known signature catalogs.

## Installation

Create a conda environment:
```
conda create -n python37_musical python=3.7
conda activate python37_musical
```

Then install some necessary packages:
```
conda install numpy scipy seaborn statsmodels scikit-learn cython matplotlib numba pandas h5py
conda install -c phlya adjusttext
conda install -c anaconda xlrd
```

Finally install MuSiCal in the development mode
```
cd  /Users/hujin/GitHub
sudo -H pip install -e ./MuSiCal
```

If Matlab codes are to be used, then the python interface for Matlab needs to be installed:
```
cd //Applications/MATLAB_R2019b.app/extern/engines/python
python setup.py build --build-base="//anaconda3/envs/python37_musical/matlab2019b" install
```
Note that matlab2019b or above is required for python3.7.

Note that matlab is not required. If the python-matlab interface is not installed, MuSiCal should run just fine (not tested yet).

Currently, there are two places where matlab codes could be used: SPA initialization and mvNMF calculation. We wrote python codes for both algorithms.

## Usage

The input matrix X can be a numpy array or a pandas DataFrame. Note that columns should be samples and rows should be features (e.g., 96 trinucleotide features).

### Running a single NMF
For running NMF, one can use
```
import musical
model = musical.NMF(X, n_components, init='random')
model.fit()
```
Then the result can be accessed through `model.W` (signatures) and `model.H` (exposures).


### Running a single mvNMF with pre-specified `lambda_tilde`
For running mvNMF, one can use
```
model = musical.mvnmf.MVNMF(X, n_components, init='random', lambda_tilde=1e-5)
model.fit()
```
The result can be accessed similarly through `model.W` and `model.H`.

### Running a single mvNMF with automatic `lambda_tilde` selection
For automatic hyperparameter selection in mvNMF, one can use
```
wrappedModel = musical.mvnmf.wrappedMVNMF(X, n_components, lambda_tilde_grid, init='random')
wrappedModel.fit()
```
where `lambda_tilde_grid` is a list of lambda_tide values that will be tested. The final selected MVNMF model can be accessed though `wrappedModel.model`. And the selected solution can be accessed through `wrappedModel.W` and `wrappedModel.H`.

### De novo signature discovery
To run de novo signature discovery with automatic rank selection, one can do
```
model = DenovoSig(X, min_n_components=2, max_n_components=20, init='random', method='nmf', n_replicates=100, ncpu=10)
model.fit()
```
The selected rank is saved at `model.n_components`, with corresponding solutions at `model.W` and `model.H`, silhouette scores for each signature at `model.sil_score`. Solutions of all tested ranks are saved at `model.W_all` and `model.H_all`, with corresponding silhouette scores at `model.sil_score_all`.

### Saving solved models
To save the solved model, one can use pickle:
```
import pickle
with open('saved_model.pkl', 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
```
To load saved models:
```
with open('saved_model.pkl', 'rb') as f:
    model = pickle.load(f)
```
