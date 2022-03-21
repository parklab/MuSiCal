
![MuSiCal workflow](./images/musical_logo.png)

**MuSiCal** (<ins>Mu</ins>tational <ins>Si</ins>gnature <ins>Cal</ins>culator) is a comprehensive toolkit for mutational signature analysis. It leverages novel algorithmic developments to enable accurate signature assignment as well as robust and sensitive signature discovery.

## Installation

### Third-party distributions

MuSiCal will be made available via *anaconda* soon. 


### Installing from source

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

## Usage

### General introduction

![MuSiCal workflow](./images/musical_workflow.png)
