"""Plot tools for mutational signature analysis."""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
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


def sigplot_bar(sig, norm=True, figsize=None, title=None, width=0.8,
                xlabel="", ylabel="", tick_fontsize=12, label_fontsize=14,
                colors=None, ylim=None, xticklabels=False, xticks=True, yticks=None, rotation=90, ha="center",
                outfile=None, fix_size=False):
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
    if fix_size:
        mpl.rcParams['pdf.fonttype'] = 42
        fig = plt.figure()
        #fig.set_size_inches(figsize[0], figsize[1])
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
            _set_size(figsize[0], figsize[1], ax=subfig)
            for tick in subfig.get_xticklabels():
                tick.set_fontname('monospace')
            for tick in subfig.get_yticklabels():
                tick.set_fontname('Arial')
            subfig.set_xlabel(xlabel, fontsize=label_fontsize)
            subfig.set_ylabel(ylabel, fontsize=label_fontsize)
            plt.bar(x, y, width=width, linewidth=0, color=colors)

            if xticks:
                plt.xticks(x, xticklabels, fontsize=tick_fontsize, rotation=rotation, ha=ha)
            else:
                subfig.set_xticks([])
            if yticks is not None:
                if type(yticks) is bool:
                    if not yticks:
                        subfig.set_yticks([])
                else:
                    subfig.set_yticks(yticks)
            plt.xlim(-1, n_features)
            if ylim is not None:
                plt.ylim(ylim[sig_index])
    else:
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
            plt.bar(x, y, width=width, linewidth=0, color=colors)

            if xticks:
                plt.xticks(x, xticklabels, fontsize=tick_fontsize, rotation=rotation, ha=ha)
            else:
                subfig.set_xticks([])
            if yticks is not None:
                if type(yticks) is bool:
                    if not yticks:
                        subfig.set_yticks([])
                else:
                    subfig.set_yticks(yticks)
            plt.xlim(-1, n_features)
            if ylim is not None:
                plt.ylim(ylim[sig_index])
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')


def plot_silhouettes(model,  title_tag=None, plotpvalues=True,
                tick_fontsize=12, label_fontsize=14,
                outfile=None):

    """Plotting function for silhouette scores.

    model is a MuSiCal object that has been processed with model.fit().

    Parameters
    ----------
    model : MuSiCal object after fitting using NMF or mnNMF

    title_tag : string
        String to add to plot titles, e.g. disease type
    """

    #Convert dictionaries of all mean silhouette scores and all reconstruction errors to arrays
    sil_score_mean_array = np.array(list(model.sil_score_mean_all.values()))
    reconstruction_error_array = np.array(list(model.reconstruction_error_all.values()))

    #Create DF from silhouette scores for heatmap and rename columns
    sil_score_all_df = pd.DataFrame.from_dict(model.sil_score_all,orient = 'index')
    sil_score_all_df.columns=[i for i in range(1, sil_score_all_df.shape[1] + 1)]

    #Plot

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(16, 4))
    grid = plt.GridSpec(1, 5, hspace=0.5, wspace=4)
    host = fig.add_subplot(grid[0, 0:3])
    plt2 = host.twinx()

    if plotpvalues:
        plt3 = host.twinx()

    heat_map = fig.add_subplot(grid[0, 3:])

    #Generate line plot
    host.set_xlabel("n components")
    host.set_ylabel("Mean silhouette score")
    plt2.set_ylabel("Reconstruction error")
    if plotpvalues:
        plt3.set_ylabel("p-value")

    color1 = '#E94E1B'
    color2 = '#1D71B8'
    if plotpvalues:
        color3 = '#2FAC66'

    p1, = host.plot(model.n_components_all, sil_score_mean_array, color=color1, label="Mean silhouette score", linestyle='--', marker='o',)
    p2, = plt2.plot(model.n_components_all, reconstruction_error_array,    color=color2, label="Reconstruction error", linestyle=':', marker='D')
    if plotpvalues:
        p3, = plt3.plot(model.n_components_all[1:], model.pvalue_all,    color=color3, label="p-value", linestyle='-.', marker='.', alpha=0.5)
    if plotpvalues:
        lns = [p1, p2, p3]
    else:
        lns = [p1, p2]

    host.legend(handles=lns, loc="lower center", bbox_to_anchor=(0.5, -0.5))

    host.yaxis.label.set_color(p1.get_color())
    plt2.yaxis.label.set_color(p2.get_color())

    if plotpvalues:
        plt3.yaxis.label.set_color(p3.get_color())

    #Adjust p-value spine position
    if plotpvalues:
        plt3.spines['right'].set_position(('outward', 65))

    #Set ticks interval to 1
    host.xaxis.set_major_locator(ticker.MultipleLocator(1))

    #Higlight suggested signature
    host.axvspan(model.n_components-0.25, model.n_components+0.25, color='grey', alpha=0.3)

    #Set title
    if title_tag is not None:
        host.set_title('Silhouette scores and reconstruction errors for '+title_tag)
    else:
        host.set_title('Silhouette scores and reconstruction errors')

    #Generate heatmap
    heat_map = sns.heatmap(sil_score_all_df,vmin=0, vmax=1, cmap="YlGnBu")
    heat_map.set_xlabel("Signatures")
    heat_map.set_ylabel("n components")

    if title_tag is not None:
        heat_map.set_title('Silhouette scores for '+title_tag)
    else:
        heat_map.set_title('Silhouette scores')

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
