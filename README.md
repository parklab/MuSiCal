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

Currently, MuSiCal has two Matlab dependencies: SPA initialization and mvNMF calculation. The mvNMF dependency has already been removed -- a python version of the mvNMF code is available. I will write the python code for SPA later, such that MuSiCal is completely Matlab independent.

## Usage
For running NMF, one can use
```
import musical
model = musical.NMF(X, n_components, init='random')
model.fit()
```
Then the result can be accessed through `model.W` and `model.H`.

For running mvNMF, one can use
```
model = musical.mvnmf.MVNMF(X, n_components, init='random', lambda_tilde=1e-5)
model.fit()
```
The result can be accessed similarly through `model.W` and `model.H`.

For automatic hyperparameter selection in mvNMF, one can use
```
wrappedModel = musical.mvnmf.wrappedMVNMF(X, n_components, lambda_tilde_grid, init='random')
wrappedModel.fit()
```
where `lambda_tilde_grid` is a list of lambda_tide values that will be tested. The final selected MVNMF model can be accessed though `wrappedModel.model`. And the selected solution can be accessed through `wrappedModel.W` and `wrappedModel.H`.

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
