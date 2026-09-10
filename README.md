<p align="center">
  <img src="./images/musical_logo.png" width="260" alt="MuSiCal logo">
</p>

**MuSiCal** (<ins>Mu</ins>tational <ins>Si</ins>gnature <ins>Cal</ins>culator) is a comprehensive toolkit for mutational signature analysis. It leverages novel algorithmic developments to enable accurate signature assignment as well as robust and sensitive signature discovery.

## Contents

- [Installation](#installation)
  - [Installing from source](#installing-from-source)
  - [Installing Sonata for Cornet](#installing-sonata-for-cornet)
  - [Setting up Jupyter Notebook](#setting-up-jupyter-notebook)
- [Usage](#usage)
  - [*De novo* signature discovery](#de-novo-signature-discovery)
  - [Refitting](#refitting)
  - [Preprocessing](#preprocessing)
- [Citation](#citation)

## Installation

MuSiCal requires Python 3.10 or above. We recommend conda for managing packages and environments. If you do not have conda on your system yet, you can install conda through [Anaconda](https://docs.anaconda.com/anaconda/install/index.html "Installation guide for Anaconda") or [Miniconda](https://docs.conda.io/en/latest/miniconda.html "Installation guide for Miniconda").

You will also need Jupyter Notebook to try out the [example scripts](./examples). If you have installed Anaconda, Jupyter Notebook will be installed already. Otherwise, follow this [guide](https://docs.jupyter.org/en/latest/install/notebook-classic.html "Installation guide for Jupyter Notebook") to install Jupyter Notebook separately. Note that it is better to install Jupyter Notebook in the `base` environment.

### Installing from source

First, download the latest repository (e.g., via `git clone`, by downloading the zip file directly, etc.).

Then, create a conda environment:
```
conda create -n python310_musical python=3.10
```

Activate the environment with `conda activate python310_musical` or `source activate python310_musical`, depending on the version of conda you have on your system.

Install some dependencies:
```
conda install numpy scipy scikit-learn matplotlib pandas seaborn
```

Install MuSiCal:
```
cd  /Path/To/MuSiCal
pip install ./MuSiCal
```

If you want to install MuSiCal in the development mode, use:
```
pip install -e ./MuSiCal
```

If `pip install` fails, try adding `sudo -H`.

### Installing Sonata for Cornet

Cornet is implemented in [Sonata](https://github.com/parklab/Sonata), which is an optional dependency of MuSiCal. It is required if you use the Cornet algorithm for *de novo* signature discovery.

With the `python310_musical` environment activated, install Sonata with:
```
pip install sonata-tools
```

### Setting up Jupyter Notebook

After installing MuSiCal (either from third-party distributions or from source), you need to set up Jupyter Notebook to try out the [example scripts](./examples).

Assuming that the `python310_musical` environment is activated, do:
```
conda install ipykernel
python -m ipykernel install --user --name python310_musical --display-name "python310_musical"
```

Since Jupyter Notebook is installed in the `base` environment, you need to deactivate the `python310_musical` environment with `conda deactivate` or `source deactivate` (depending on your conda version) to access Jupyter Notebook. You can launch Jupyter Notebook with
```
jupyter notebook
```  
If you have installed Anaconda, you can also launch Jupyter Notebook from the graphical interface of Anaconda-Navigator.

Now you are ready to try out the [example scripts](./examples). Remember to set the kernel of the notebook to `python310_musical`.

## Usage

MuSiCal can be used after `import musical` within python.

The overall goal of mutational signature analysis is to decompose a mutation count matrix *X* into a signature matrix *W* and an exposure matrix *H*. Note that *X* is mutation type by sample (i.e., each column is a sample), *W* is mutation type by signature, and *H* is signature by sample.

### *De novo* signature discovery

*De novo* signature discovery is performed to derive *de novo* signatures, through `musical.DenovoSig`. The input matrix is factorized repeatedly, over a range of numbers of signatures and over multiple replicates, optionally on bootstrapped count matrices. The resulting solutions are then filtered and aggregated, and the best number of signatures is selected.

Three algorithms are available, selected with the `method` parameter of `musical.DenovoSig`:

| `method` | Algorithm | Notes |
| --- | --- | --- |
| `'nmf'` | Standard NMF | Nonnegative matrix factorization with the Kullback-Leibler divergence. Simple and fast. |
| `'mvnmf'` | Minimum-volume NMF | Resolves the non-uniqueness problem of standard NMF by penalizing the volume spanned by the signatures. Addresses the "weight-stealing" problem in signature discovery, which is particularly relevant for flat signatures. |
| `'cornet'` | Cornet | Explicitly models the correlation structure between signatures. Excels at resolving correlated and composite signatures and mitigating cross-contaminations, especially in homogeneous datasets, smaller cohorts, or specific biological contexts such as healthy tissues. Implemented in [Sonata](https://github.com/parklab/Sonata), which must be installed separately. |

For example:
```
import musical

model = musical.DenovoSig(X, min_n_components=1, max_n_components=15,
                          method='cornet', n_replicates=20, ncpu=10)
model.fit()
```

#### Rescaling the total mutation burden

The `tmb_factor` parameter rescales the input matrix so that the mean total mutation burden per sample becomes `tmb_factor` times the number of mutation types. For example, for SBS96 signatures there are 96 mutation types, so a `tmb_factor` of 10 corresponds to a mean total mutation burden of 960. Relative differences in mutation burden between samples are preserved. Rescaling is only relevant for Cornet. In general, a larger `tmb_factor` leads to sparser exposures and flatter signatures in Cornet solutions, and a smaller `tmb_factor` leads to flatter exposures and sparser signatures. We recommend a `tmb_factor` between 10 and 40 for SBS96 signatures. 

### Refitting

In **refitting**, the observed mutational spectrum is decomposed into a combination of existing signatures with nonnegative coefficients. MuSiCal utilizes a novel algorithm called likelihood-based sparse nonnegative least squares (NNLS) for refitting. See [example scripts](./examples/example_refitting.ipynb).

### Preprocessing

MuSiCal provides **preprocessing** functionalities for automatic cohort stratification and outlier removal, to further improve the sensitivity of *de novo* signature discovery. See [example scripts](./examples/example_preprocessing.ipynb).

## Citation

If you use MuSiCal, please cite:

> Jin, H., Gulhan, D. C., Geiger, B., Ben-Isvy, D., Geng, D., Ljungström, V. & Park, P. J. Accurate and sensitive mutational signature analysis with MuSiCal. *Nature Genetics* **56**, 541–552 (2024). https://doi.org/10.1038/s41588-024-01659-0

If you use the Cornet algorithm, please additionally cite the Cornet paper. A manuscript describing Cornet is currently in preparation, and this section will be updated with the citation once it becomes available.

