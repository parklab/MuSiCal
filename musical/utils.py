"""Utilities"""

import numpy as np
import sklearn.decomposition._nmf as sknmf

##################
# Useful globals #
##################

trinucleotides_C = ["ACA",
                    "ACC",
                    "ACG",
                    "ACT",
                    "CCA",
                    "CCC",
                    "CCG",
                    "CCT",
                    "GCA",
                    "GCC",
                    "GCG",
                    "GCT",
                    "TCA",
                    "TCC",
                    "TCG",
                    "TCT"]

trinucleotides_T = ["ATA",
                    "ATC",
                    "ATG",
                    "ATT",
                    "CTA",
                    "CTC",
                    "CTG",
                    "CTT",
                    "GTA",
                    "GTC",
                    "GTG",
                    "GTT",
                    "TTA",
                    "TTC",
                    "TTG",
                    "TTT"]

snv_types_6_str = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]

snv_types_6_set = [{"C", "A"}, {"C", "G"}, {"C", "T"},
                   {"T", "A"}, {"T", "C"}, {"T", "G"}]

snv_types_96_str = (["C>A" + ":" + item for item in trinucleotides_C] +
                    ["C>G" + ":" + item for item in trinucleotides_C] +
                    ["C>T" + ":" + item for item in trinucleotides_C] +
                    ["T>A" + ":" + item for item in trinucleotides_T] +
                    ["T>C" + ":" + item for item in trinucleotides_T] +
                    ["T>G" + ":" + item for item in trinucleotides_T])

snv_types_96_list = ([["C>A", item] for item in trinucleotides_C] +
                     [["C>G", item] for item in trinucleotides_C] +
                     [["C>T", item] for item in trinucleotides_C] +
                     [["T>A", item] for item in trinucleotides_T] +
                     [["T>C", item] for item in trinucleotides_T] +
                     [["T>G", item] for item in trinucleotides_T])


#####################
# Utility functions #
#####################

def beta_divergence_deprecated(A, B, beta=1, square_root=False):
    """Beta_divergence

    A and B must be float arrays.

    DEPRECATED
    ----------
    sknmf._beta_divergence has problems when beta = 1. The problem seems to be in the following line:
    sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1)),
    together with the fact that it takes 3 separate matrix sums and then do the additions. That results
    in reduced accuracy in small differences, which is really what matters for the final result. Basically,
    KL divergence is \sum(Alog(A/B) - A + B). If you do \sum(Alog(A/B)) - \sum(A) + \sum(B), it'll be less
    accurate.
    Note that using the aux matrix as in this function seems better than using sknmf._beta_divergence alone.
    But we should not use this function anyway.
    """
    aux = np.diag(np.ones(B.shape[1])).astype(float)
    return sknmf._beta_divergence(A, B, aux, beta=beta, square_root=square_root)


def beta_divergence(A, B, beta=1, square_root=False):
    """Beta_divergence

    A and B must be float arrays.

    Notes
    ----------
    For KL divergence, we do not deal with the case where B_ij = 0 but A_ij != 0,
    i.e., we'll output inf if that happens.
    For the purpose of NMF, B_ij will never be 0, because within the NMF algorithm,
    we always clip small values. So 0's will be replaced by EPSILON in W and H. As
    a result, W @ H will never have 0 entries.
    """
    if beta == 1 or beta == "kullback-leibler":
        # When B_ij = 0, KL divergence is not defined unless A_ij = 0.
        # Whenever A_ij = 0, then the contribution of the term A_ij log(A_ij/B_ij)
        # is considered as 0.
        A_data = A.ravel()
        B_data = B.ravel()
        indices = A_data > 0
        A_data = A_data[indices]
        B_data_remaining = B_data[~indices]
        B_data = B_data[indices]
        # Here we must take matrix additions first and then take sum.
        # Otherwise, the separate matrix sums will be too big and the small
        # differences will be lost, and we'll get 0.0 results.
        res = np.sum(A_data*np.log(A_data/B_data) - A_data + B_data)
        res = res + np.sum(B_data_remaining)
    else:
        raise ValueError('Only beta = 1 is implemented.')
    if square_root:
        res = np.sqrt(2*res)

    return res


def normalize_WH(W, H):
    normalization_factor = np.sum(W, 0)
    return W/normalization_factor, H*normalization_factor[:, None]
