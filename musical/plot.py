"""Plot tools for mutational signature analysis."""

import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from .utils import snv_types_96_str

##################
# Color palettes #
##################

# 10 colors
colorPaletteMathematica97 = [(0.368417, 0.506779, 0.709798),
                             (0.880722, 0.611041, 0.142051),
                             (0.560181, 0.691569, 0.194885),
                             (0.922526, 0.385626, 0.209179),
                             (0.528288, 0.470624, 0.701351),
                             (0.772079, 0.431554, 0.102387),
                             (0.363898, 0.618501, 0.782349),
                             (1, 0.75, 0),
                             (0.280264, 0.715, 0.429209),
                             (0, 0, 0)]

# 30 colors
colorPaletteBrown = ['#a0e3b7', '#20d8fd', '#003c70', '#f6adff', '#163719',
                     '#a2f968', '#5b0891', '#d1c1d9', '#71114b', '#09f54c',
                     '#e81659', '#36a620', '#ec4dd8', '#6c8e45', '#a05abc',
                     '#24ffcd', '#52351d', '#f8ba7c', '#842411', '#c9ce23',
                     '#3d84e3', '#0f767a', '#f7794f', '#8b716f', '#ff2a0d',
                     '#3f16f9', '#ab7b05', '#91B493', '#FFB11B', 'white']

# 12 colors, paired
colorPalettePaired = sns.color_palette('Paired', 12)

# 10 colors, gradient
colorPaletteHusl = sns.color_palette('husl', 10)

# 10 colors
colorPaletteSet1 = sns.color_palette('Set1', 10)
colorPaletteSet1[-1] = (0, 0, 0)

# Trinucleotide colors for the 96 dimensional mutation spectrum
colorPaletteTrinucleotides = [(0.33, 0.75, 0.98),
                              (0, 0, 0),
                              (0.85, 0.25, 0.22),
                              (0.78, 0.78, 0.78),
                              (0.51, 0.79, 0.24),
                              (0.89, 0.67, 0.72)]

##################
# Plot functions #
##################


# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
def _set_size(w, h, ax=None):
    """w, h: width, height in inches."""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def sigplot_bar(sig, norm=True, figsize=None, title=None,
                xlabel="", ylabel="", tick_fontsize=12, label_fontsize=14,
                colors=None, ylim=None, xticklabels=False, rotation=90, ha="center",
                outfile=None):
    """Bar plot for signatures.

    sig can be a single n_features dimensional vector, in which case a single plot will
    be generated for this signature. If n_features = 96, the vector should follow the
    order as in snv_types_96_str. sig can also be a catalogue (W), i.e., an n_features by
    n_components matrix, in which case the entire catalogue is plotted.

    Parameters
    ----------
    sig : array-like of shape (n_features,) or (n_features, n_components)
        Signature vector or signature matrix.

    figsize : tuple
        Size of a single signature plot.
    """
    sig = np.array(sig).astype(float)
    # Input is a single signature vector, convert to a matrix
    if len(sig.shape) == 1:
        sig = np.reshape(sig, (-1, 1))
    n_features, n_components = sig.shape
    if norm:
        sig = normalize(sig, axis=0, norm='l1')

    # Colors
    if colors is None:
        if n_features == 96:
            colors = []
            for i in range(0, 6):
                colors.extend([colorPaletteTrinucleotides[i]]*16)
        else:
            colors = ['gray']*n_features
    else:
        if type(colors) is str or type(colors) is tuple:
            colors = [colors]*n_features
        elif type(colors) is list:
            if len(colors) != n_features:
                raise ValueError('If colors is a list, its length must be the same as n_features.')
        else:
            raise TypeError('Colors can be a single color, i.e., a str or tuple, or a list of colors.')

    # x tick labels
    if xticklabels is None:
        if n_features == 96:
            xticklabels = snv_types_96_str
        else:
            xticklabels = list(map(str, range(1, n_features + 1)))
    else:
        if type(xticklabels) is bool:
            if xticklabels:
                if n_features == 96:
                    xticklabels = snv_types_96_str
                else:
                    xticklabels = list(map(str, range(1, n_features + 1)))
            else:
                xticklabels = ['']*n_features
        elif type(xticklabels) is list:
            if len(xticklabels) != n_features:
                raise ValueError('If xticklabels is a list, its length must be the same as n_features.')
        else:
            raise TypeError('xticklabels must be bool, list, or None.')

    # Title
    if title is None:
        title = ['Signature ' + str(i) for i in range(1, n_components + 1)]
    else:
        if type(title) is bool:
            if title:
                title = ['Signature ' + str(i) for i in range(1, n_components + 1)]
            else:
                title = ['']*n_components
        elif type(title) is str:
            title = [title]*n_components
        elif type(title) is list:
            if len(title) != n_components:
                raise ValueError('If title is a list, its length must be the same as n_components.')
        else:
            raise TypeError('Title must be str, bool, list, or None.')

    # Figure size
    if figsize is None:
        figsize = (8, 2*n_components)
    else:
        figsize = (figsize[0], figsize[1]*n_components)

    # ylim
    if ylim is not None:
        if type(ylim) is tuple:
            ylim = [ylim]*n_components
        elif type(ylim) is list:
            if type(ylim[0]) is not tuple:
                raise ValueError('When ylim is a list, it must be a list of tuples.')
            if len(ylim) != n_components:
                raise ValueError('When ylim is a list, its length must be the same as n_components.')
        else:
            raise TypeError('ylim must be tuple, list of tuples, or None.')

    # Plot
    mpl.rcParams['pdf.fonttype'] = 42
    fig = plt.figure()
    fig.set_size_inches(figsize[0], figsize[1])
    sns.set_context('notebook')
    sns.set_style('ticks')
    plt.rc('xtick', labelsize=tick_fontsize)
    plt.rc('ytick', labelsize=tick_fontsize)
    for sig_index in range(0, n_components):
        y = sig[:, sig_index]
        x = np.arange(0, n_features)
        subfig = fig.add_subplot(n_components, 1, sig_index + 1)
        subfig.set_title(title[sig_index], fontsize=label_fontsize)
        subfig.spines['right'].set_visible(False)
        subfig.spines['top'].set_visible(False)
        subfig.spines['bottom'].set_color('k')
        subfig.spines['left'].set_color('k')
        for tick in subfig.get_xticklabels():
            tick.set_fontname('monospace')
        for tick in subfig.get_yticklabels():
            tick.set_fontname('Arial')
        subfig.set_xlabel(xlabel, fontsize=label_fontsize)
        subfig.set_ylabel(ylabel, fontsize=label_fontsize)
        plt.bar(x, y, color=colors)

        plt.xticks(x, xticklabels, fontsize=tick_fontsize, rotation=rotation, ha=ha)
        plt.xlim(-1, n_features)
        if ylim is not None:
            plt.ylim(ylim[sig_index])
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
