
![MuSiCal logo](./images/musical_logo.png)

**MuSiCal** (<ins>Mu</ins>tational <ins>Si</ins>gnature <ins>Cal</ins>culator) is a comprehensive toolkit for mutational signature analysis. It leverages novel algorithmic developments to enable accurate signature assignment as well as robust and sensitive signature discovery.

## Installation

MuSiCal requires Python 3.7 or above. We recommend conda for managing packages and environments. If you do not have conda on your system yet, you can install conda through [Anaconda](https://docs.anaconda.com/anaconda/install/index.html "Installation guide for Anaconda") or [Miniconda](https://docs.conda.io/en/latest/miniconda.html "Installation guide for Miniconda").  

### Third-party distributions

MuSiCal will be made available via conda soon.

### Installing from source

First, download the latest repository (e.g., via `git clone`, by downloading the zip file directly, etc.). Then, create a conda environment:
```
conda create -n python37_musical python=3.7
```

Activate the environment with `conda activate python37_musical` or `source activate python37_musical`, depending on the version of conda you have on your system.

Install some dependencies:
```
conda install numpy scipy scikit-learn matplotlib pandas seaborn
```

Finally install MuSiCal:
```
cd  /Path/To/MuSiCal
pip install ./MuSiCal
```

If you want to install MuSiCal in the development mode:
```
pip install -e ./MuSiCal
```

If `pip install` fails, try adding `sudo -H`.

### Installing Jupyter Notebook

After installing MuSiCal (either from third-party distributions or from source), you need to set up Jupyter Notebook to try out the [example scripts](./examples).


## Usage

### General introduction

![MuSiCal workflow](./images/musical_workflow.png)
