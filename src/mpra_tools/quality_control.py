"""Functions to perform quality control checks during library processing.

"""
import itertools
import warnings

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

from . import plot_utils


def report_coverage(counts_df):
    """Report the average barcode coverage for each sample.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw barcode counts for each sample.

    Returns
    -------
    coverage : list of lists
        Values of inner list are (1) sample name, (2) "mean" to indicate coverage is calculated with the mean, (3) the coverage, (4) the n
    """
    n_bc = counts_df.shape[0]
    # Need to remove the labels column of the df. All remaining columns are samples
    coverage = counts_df.drop(columns="label").sum() / n_bc
    coverage = coverage.to_dict()
    coverage = [[k, "mean", v, n_bc] for k, v in coverage.items()]
    return coverage


def reproducibility(df, title, log=None, pseudocount=0, figname=None):
    """Make reproducibility plots for each pair of replicates, in linear or log space.

    Parameters
    ----------
    df : pd.DataFrame
        Columns are samples, rows represent library members (barcodes or elements).
    title : str
        String for figure suptitles indicating what measurement is being shown (raw counts, RNA/DNA counts, etc.)
    log : int or None
        If not specified, plot the data in input linear space. Otherwise, plot the data in log2 or log10 space,
        depending on what is specified.
    pseudocount : int
        Pseudocount to add to the data before plotting in log space.
    figname : str or None
        If specified, save the figure with the provided name.

    Returns
    -------
    reproducibilities : list of lists
        Values of the inner list are, in order: (1) the two samples being compared (2) the correlation metric being used, (3) the value of that metric, (4) the n.
    fig : Figure handle.
    """
    reproducibilities = []
    # Setup the figure -- there are n choose 2 pairs.
    n_samples = df.shape[1]
    fig, ax_list = plot_utils.setup_multiplot(n_samples * (n_samples - 1) / 2, big_dimensions=False)
    ax_list = ax_list.flatten()

    # log transform, if necessary
    if log:
        if log == 2:
            transform = np.log2
        else:
            if log != 10:
                warnings.warn(f"Did not recognize type of log transform. Using log10. Saw {log}.")
            transform = np.log10
        df = transform(df + pseudocount)

    # For all pairs of replicates...
    for ax, (rep1, rep2) in zip(ax_list, itertools.combinations(df, 2)):
        # Exclude things that have a NaN in either
        mask = df[[rep1, rep2]].notna().all(axis=1)
        _, _, (pcc, scc) = plot_utils.scatter_with_corr(
            df.loc[mask, rep1],
            df.loc[mask, rep2],
            rep1,
            rep2,
            colors="k",
            alpha=0.1,
            rasterize=True,
            figax=(fig, ax),
            reproducibility=True,
            loc="upper left",
        )
        k = f"{rep1}_vs_{rep2}"
        n = mask.sum()
        reproducibilities.append([k, "pearson", pcc, n])
        reproducibilities.append([k, "spearman", scc, n])

    fig.suptitle(title)
    if figname:
        plot_utils.save_fig(fig, figname)

    return reproducibilities, fig


def rna_vs_dna(rna_counts, dna_counts, y_label, pseudocount=0, title=None, figname=None):
    """Plot the log2 RNA counts (or RNA/DNA counts) vs the log2 DNA counts and make sure the two are not too
    correlated. Data is provided in linear space and will be log-transformed internally.

    Parameters
    ----------
    rna_counts : pd.DataFrame
        RNA or RNA/DNA counts for each sample in linear space. Rows are barcodes, columns are RNA samples.
    dna_counts : pd.Series
        The barcode counts in linear space in the DNA pool.
    y_label : str
        Label for the y-axis.
    pseudocount : float
        Optional pseudocount before taking log.
    title : str or None
        If specified, the suptitle for the plot.
    figname : str or None
        If specified, save the figure under this name.

    Returns
    -------
    correlations : list of lists
        Values of the inner list are, in order: (1) The two samples being compared, (2) the correlation metric being used, (3) the value of that metric, (4) the n.
    fig : figure handle
    """
    correlations = []
    rna_counts = np.log2(rna_counts + pseudocount)
    dna_counts = np.log2(dna_counts + pseudocount)

    fig, ax_list = plot_utils.setup_multiplot(rna_counts.shape[1], big_dimensions=False)
    ax_list = ax_list.flatten()
    for ax, sample in zip(ax_list, rna_counts):
        # Exclude things that have a NaN in either
        mask = dna_counts.notna() & rna_counts[sample].notna()
        _, _, (pcc, scc) = plot_utils.scatter_with_corr(
            dna_counts[mask],
            rna_counts.loc[mask, sample],
            "log2 DNA Counts",
            y_label,
            colors="k",
            alpha=0.1,
            rasterize=True,
            figax=(fig, ax),
            reproducibility=True,
            loc="upper right",
        )
        k = f"{sample}_vs_DNA"
        n = mask.sum()
        correlations.append([k, "pearson", pcc, n])
        correlations.append([k, "spearman", scc, n])

    if title:
        fig.suptitle(title)
    if figname:
        plot_utils.save_fig(fig, figname)

    return correlations, fig
