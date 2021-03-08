"""Class for signature catalog

TODO
----------
1. Decide if it is necessary to have a Catalog class. If not, remove it.
2. Decide whether to have static attributes like COSMIC_v3_SBS_WGS. See end of the file.
    Currently, we do not write static attributes, but use load_catalog() function instead.
"""

import numpy as np
import pandas as pd
import warnings
import importlib.resources

from . import data

### Available catalogs
# For each catalog, a renormalization was done such that each signature is l1 normalized.
CATALOG_NAMES = [
    'COSMIC_v2_SBS_WGS', # https://cancer.sanger.ac.uk/cancergenome/assets/signatures_probabilities.txt
    'COSMIC_v3_SBS_WGS', # https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Signatures/SP_Signatures/SigProfiler_reference_signatures/SigProfiler_reference_whole-genome_signatures/sigProfiler_SBS_signatures_2019_05_22.csv
    'COSMIC_v3_SBS_WES', # https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Signatures/SP_Signatures/SigProfiler_reference_signatures/Sigprofiler_Exome_Signatures/sigProfiler_exome_SBS_signatures.csv
    'COSMIC_v3p1_SBS_WGS' # https://cancer.sanger.ac.uk/sigs-assets-20/COSMIC_Mutational_Signatures_v3.1.xlsx
]

def load_catalog(name='COSMIC_v3p1_SBS_WGS', sep=',', index_col=0):
    """Load saved or custom signature catalog.

    Parameters
    ----------
    name : str
        File path for a custom catalog or name of a saved catalog. For custom catalog,
        the file must contain a header line.

    sep : str
        Delimiter for the custom catalog file.

    index_col : See pd.read_csv()


    Returns
    ----------
    catalog : Catalog
        Signature catalog of the Catalog class.
    """
    if name in CATALOG_NAMES:
        catalog = pd.read_csv(importlib.resources.open_text(data, name + '.csv'), sep=',', index_col=0)
        catalog = Catalog(catalog)
        return catalog
    else:
        catalog = pd.read_csv(name, sep=sep, index_col=index_col)
        catalog = Catalog(catalog)
        return catalog


class Catalog:
    """Class for signature catalog"""
    def __init__(self, W=None, signatures=None, features=None):
        if W is None:
            self._W = pd.DataFrame(W)
        elif type(W) is pd.DataFrame:
            if signatures is not None:
                warnings.warn('Columns of W are used as signatures. The provided signatures attribute is ignored.',
                              UserWarning)
            if features is not None:
                warnings.warn('Index of W is used as features. The provided features attribute is ignored.',
                              UserWarning)
            self._W = W
        elif type(W) is np.ndarray:
            if signatures is None:
                signatures = ['Signature_' + str(i) for i in range(1, W.shape[1] + 1)]
            if features is None:
                features = ['Feature_' + str(i) for i in range(1, W.shape[0] + 1)]
            self._W = pd.DataFrame(W, columns=signatures, index=features)
        else:
            raise ValueError('W must be pd.DataFrame, np.ndarray, or None.')
        self._signatures = self._W.columns.values.tolist()
        self._features = self._W.index.values.tolist()

    @property
    def W(self):
        return self._W

    @property
    def signatures(self):
        return self._signatures

    @property
    def sigs(self):
        return self._signatures

    @property
    def features(self):
        return self._features

    @property
    def n_sigs(self):
        return len(self.signatures)

    @property
    def n_signatures(self):
        return len(self.signatures)

    @property
    def n_features(self):
        return len(self.features)

###############################################################################
######################## COSMIC reference signatures ##########################
###############################################################################
# For each signature catalog, a renormalization was done such that each signature is l1 normalized.

# https://cancer.sanger.ac.uk/cancergenome/assets/signatures_probabilities.txt
#COSMIC_v2_SBS_WGS = read_catalog(importlib.resources.open_text(data, 'COSMIC_v2_SBS_WGS.csv'))

# https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Signatures/SP_Signatures/SigProfiler_reference_signatures/SigProfiler_reference_whole-genome_signatures/sigProfiler_SBS_signatures_2019_05_22.csv
#COSMIC_v3_SBS_WGS = read_catalog(importlib.resources.open_text(data, 'COSMIC_v3_SBS_WGS.csv'))

# https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Signatures/SP_Signatures/SigProfiler_reference_signatures/Sigprofiler_Exome_Signatures/sigProfiler_exome_SBS_signatures.csv
#COSMIC_v3_SBS_WES = read_catalog(importlib.resources.open_text(data, 'COSMIC_v3_SBS_WES.csv'))

# https://cancer.sanger.ac.uk/sigs-assets-20/COSMIC_Mutational_Signatures_v3.1.xlsx
#COSMIC_v3p1_SBS_WGS = read_catalog(importlib.resources.open_text(data, 'COSMIC_v3p1_SBS_WGS.csv'))
