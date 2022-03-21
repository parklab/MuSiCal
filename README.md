
![MuSiCal workflow](./images/musical_logo.png)

MuSiCal (<ins>Mu</ins>tational <ins>Si</ins>gnature <ins>Cal</ins>culator) is a comprehensive toolkit for mutational signature analysis. It leverages novel algorithmic developments to enable accurate signature assignment as well as robust and sensitive signature discovery.

## Install

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

### General introduction

![MuSiCal workflow](./images/musical_workflow.png)
