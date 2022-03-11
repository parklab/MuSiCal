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
    'COSMIC_v3p1_SBS_WGS', # https://cancer.sanger.ac.uk/sigs-assets-20/COSMIC_Mutational_Signatures_v3.1.xlsx
    'COSMIC_v3p2_SBS_WGS', #https://cancer.sanger.ac.uk/signatures/downloads/
    'COSMIC-MuSiCal_v3p2_SBS_WGS',
    'COSMIC_v3p1_Indel', # https://cancer.sanger.ac.uk/signatures/documents/440/COSMIC_v3.1_ID_GRCh37.txt
    'MuSiCal_v4_Indel_WGS',
]

def load_catalog(name='COSMIC-MuSiCal_v3p2_SBS_WGS', sep=',', index_col=0):
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
        catalog = Catalog(catalog, name = name)
        return catalog
    else:
        catalog = pd.read_csv(name, sep=sep, index_col=index_col)
        catalog = Catalog(catalog, name = name)
        return catalog


        

class Catalog:
    """Class for signature catalog"""
    def __init__(self, W=None, signatures=None, features=None, name = None):
        if W is None:
            self._W = pd.DataFrame(W)
            self._name = name
        elif type(W) is pd.DataFrame:
            if signatures is not None:
                warnings.warn('Columns of W are used as signatures. The provided signatures attribute is ignored.',
                              UserWarning)
            if features is not None:
                warnings.warn('Index of W is used as features. The provided features attribute is ignored.',
                              UserWarning)
            self._W = W
            self._name = name
        elif type(W) is np.ndarray:
            if signatures is None:
                signatures = ['Signature_' + str(i) for i in range(1, W.shape[1] + 1)]
            if features is None:
                features = ['Feature_' + str(i) for i in range(1, W.shape[0] + 1)]
            self._W = pd.DataFrame(W, columns=signatures, index=features)
            self._name = name
        else:
            raise ValueError('W must be pd.DataFrame, np.ndarray, or None.')
        self._signatures = self._W.columns.values.tolist()
        self._features = self._W.index.values.tolist()
        self._W_norm = self._W
        self._sig_type = ''
        if 'SBS' in self._name:
            self._sig_type = 'SBS'
        elif 'Indel' in self._name:
            self._sig_type = 'Indel'

    def restrict_catalog(self, tumor_type = None, is_MMRD = False, is_PPD = False):
        if tumor_type != None:
            if self._sig_type == '':
                raise ValueError('Supported for SBS and Indel catalogs')
            tts_sigs = pd.read_csv(importlib.resources.open_text(data, 'TumorType_' + self._sig_type + '_Signatures.csv'), sep =',')        
            tts_sigs_selected = tts_sigs.loc[tts_sigs['tumor_type'] == tumor_type]
            signatures_tt = np.array(tts_sigs_selected['signatures'])
            # adding MMRD PPD signatures irrespective of tumor type 
            tts_sigs_any = tts_sigs.loc[tts_sigs['tumor_type'] == 'any']
            signatures_any = np.array(tts_sigs_any['signatures'])
            signatures_tt = np.unique(np.append(signatures_tt, signatures_any))
            self._signatures = [item for index,item in enumerate(self._signatures) if item in signatures_tt]
            self._W = self._W[self._signatures]
        # removing MMRD and PPD signatures if is_MMRD and is_PPD is set to False
        if not is_MMRD or not is_PPD:
            MMRD_PPD_sigs = pd.read_csv(importlib.resources.open_text(data, 'MMRD_PPD_' + self._sig_type + '_Signatures.csv'), sep =',')
            signatures_MMRD = np.array(MMRD_PPD_sigs.loc[MMRD_PPD_sigs['MMRD_PPD_category'] == "MMRD"])
            signatures_MMRD_PPD = np.array(MMRD_PPD_sigs.loc[MMRD_PPD_sigs['MMRD_PPD_category'] == "MMRD_PPD"])
            signatures_PPD = np.array(MMRD_PPD_sigs.loc[MMRD_PPD_sigs['MMRD_PPD_category'] == "PPD"])
            if not is_MMRD:
                self._signatures = [item for index,item in enumerate(self._signatures) if item not in signatures_MMRD]
                self._signatures = [item for index,item in enumerate(self._signatures) if item not in signatures_MMRD_PPD]
                self._W = self._W[self._signatures]
            if not is_PPD:
                self._signatures = [item for index,item in enumerate(self._signatures) if item not in signatures_PPD]
                self._W = self._W[self._signatures]

        
    def normalize_W_catalog(self, sequencing = 'WES'):
        W = self._W
        if self._sig_type == 'SBS':
            weights = pd.read_csv(importlib.resources.open_text(data, 'TriNucFreq_Weights.csv'), sep =',', index_col=0)
        else:
            raise ValueError('No weight provided with the specified sig_type')
    
        sequencing_type = weights.columns
        weights = np.array(weights)
        weight = weights[:,np.where(np.array(sequencing_type) == sequencing)[0]]
        weight = np.array(weight)
        weight = np.ravel(weight)
        W = np.array(W)
        W_norm = []
        for w in W.T:
            w = np.array(w)
            w = np.multiply(w, weight)
            w = w/np.sum(w)
            W_norm.append(w)
        
        W_norm = np.array(W_norm)
        W_norm = W_norm.T
        self._W_norm = pd.DataFrame(W_norm, columns = self._signatures)

    @property
    def W(self):
        return self._W

    @property
    def W_norm(self):
        return self._W_norm

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
