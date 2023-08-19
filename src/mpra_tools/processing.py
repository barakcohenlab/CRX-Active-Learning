"""Functions to process raw barcode counts into activity scores for each element of an MPRA library
and report QC failures.
"""
import os

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

from . import quality_control
from . import fasta_utils
from . import plot_utils
from . import stat_utils
from . import loaders
from . import log

logger = log.get_logger()


def _load_count_file(filename):
    """Helper function for load_counts_files to read in one file with barcode counts."""
    return loaders.load_data(filename, usecols=["barcode", "count"]).squeeze()


def load_counts_files(configs):
    """Read in barcode count files and store the result in a DataFrame, where rows are barcodes and
    columns are samples and the name of the library member associated with the barcode.

    Parameters
    ----------
    configs : dict
        Nested dictionary containing file name of RNA and DNA count files, and the desired name to
        use for each sample. Also contains the mapping between barcodes and library member names.

    Returns
    -------
    counts_df : pd.DataFrame
        Index is the barcode, columns are the samples and the label of the element that the barcode
        tags.
    rna_labels : np.array[str]
        Names of the RNA samples, in the order they appear in the DataFrame.
    dna_label : str
        Name of the DNA sample, for now assume there is only one. (FIXME allow for multiple?)
    """
    logger.info("Loading in data.")
    # Start by storing things with built in data structures
    rna_labels = []
    counts_df = dict()
    # Read RNA samples
    for name, file in configs["rna_files"].items():
        rna_labels.append(name)
        counts_df[name] = _load_count_file(file)

    # Read DNA sample
    dna_label, dna_file = configs["dna_file"]
    counts_df[dna_label] = _load_count_file(dna_file)

    # Finally add in the name of the assocaited library members
    counts_df["label"] = loaders.load_data(configs["barcode_mapping"], header=None).squeeze()

    rna_labels = np.array(rna_labels)
    # Keys of the dict become columns of the df
    counts_df = pd.DataFrame(counts_df)
    return counts_df, rna_labels, dna_label


def plot_barcode_distributions(counts_df, dna_label, min_dna_counts, title=None, figname=None):
    """For each sample, make a histogram of the log10 barcode counts.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw barcode counts. Each row is a barcode and each column is a sample. Element names are
        not included.
    dna_label : str
        Indicates the name of the column with the DNA sample.
    min_dna_counts : int
        Threshold that will be used to remove low-abundance DNA barcodes. This will be shown on
        the DNA plot.
    title : str or None
        The suptitle of the Figure, if specified.
    figname : str or None
        If specified, save the figure with this name.

    Returns
    -------
    fig : Handle to the figure

    """
    log_df = np.log10(counts_df + 1)
    min_dna_counts = np.log10(min_dna_counts + 1)

    fig, ax_list = plot_utils.setup_multiplot(counts_df.shape[1], big_dimensions=False)
    ax_list = ax_list.flatten()

    # For each sample...
    for ax, sample in zip(ax_list, log_df):
        ax.hist(log_df[sample], bins="auto")
        ax.set_title(sample)
        # If this is the DNA sample also show the cutoff
        if sample == dna_label:
            ax.axvline(min_dna_counts, color="k")

    # x axis
    fig.text(0.5, 0.025, "log10 Raw Barcode Counts + 1", ha="center", va="top")
    # y axis
    fig.text(0.025, 0.5, "Number of Barcodes", rotation=90, ha="right", va="center")

    if title:
        fig.suptitle(title)
    if figname:
        plot_utils.save_fig(fig, figname)

    return fig


def filter_low_counts(counts_df, min_dna_counts, dna_label, rna_labels, max_rna_cv,
                      output_prefix=None):
    """Remove any barcodes that are either below the minimum DNA counts cutoff, or above the maximum
    RNA sample cofficient of variation cutoff.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Rows are barcodes, columns are samples or name of element corresponding to the barcode.
        One sample is the DNA.
    min_dna_counts : int
        Minimum number of reads for a barcode to be present in the DNA pool. Anything that is
        strictly less than this value gets set to NaN.
    dna_label : str
        Name of the column for the DNA.
    rna_labels : np.array[str]
        Names of the columns for the RNA samples.
    max_rna_cv : float
        Maximum sample coefficient of variation allowed for a barcode.
    output_prefix : str or None
        If specified, save the CV vs mean plot to file.

    Returns
    -------
    counts_df : pd.DataFrame
        The input, but with low-abundance barcodes set to NaN.
    qc_info : list of lists
        Each inner list is [sample, "count", number of BCs removed, total number of BCs]
    """
    logger.info("Filtering low-abundance and noisy measurements.")
    qc_info = []
    mask = counts_df[dna_label] < min_dna_counts
    qc_info.append([dna_label, "count", mask.sum(), len(mask)])
    sample_labels = np.append(rna_labels, [dna_label])
    counts_df.loc[mask, sample_labels] = np.nan
    # Now do RNA samples
    cvs = counts_df[rna_labels].apply(stat_utils.variation, axis=1, pseudocount=1e-2)
    cv = cvs["cv"]
    means = cvs["mean"]
    # Show CV vs Mean
    mask = means.notna()
    fig, ax, corrs = plot_utils.scatter_with_corr(
        means[mask],
        cv[mask],
        "Raw Barcode Mean",
        "CV",
        colors="density",
        rasterize=True,
        figax=plt.subplots(figsize=plot_utils.get_figsize(0.3, 1))
    )
    ax.axhline(max_rna_cv, color="k", linestyle="--")
    ax.set_xscale("log")
    plot_utils.save_fig(fig, f"{output_prefix}BarcodeCV")
    plt.close(fig)

    cv_fail = cv > max_rna_cv
    qc_info.append(["RNA", "count", cv_fail.sum(), len(cv_fail)])
    counts_df.loc[cv_fail, rna_labels] = np.nan
    logger.info(qc_info)

    return counts_df, qc_info


def counts_per_million(counts_df, sample_labels):
    """Normalize each sample to counts per million.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Rows are barcodes, columns are samples or name of element corresponding to the barcode.
    sample_labels : np.array[str]
        Names of the columns for the RNA or DNA samples.

    Returns
    -------
    counts_df : pd.DataFrame
        The input, but now each sample is normalized to the sequencing depth.
    """
    logger.info("CPM normalizing.")
    for sample in sample_labels:
        counts = counts_df[sample]
        millions = counts.sum() / 1e6
        counts_df[sample] = counts / millions

    return counts_df


def dna_normalize(counts_df, rna_labels, dna_label, title=None, output_prefix=None):
    """Normalize all RNA samples to the DNA barcode counts. Then plot the log reproducibility of
    RNA/DNA ratios between replicates. Then check the RNA/DNA vs DNA to make sure there is no
    residual correlation.

    Parameters
    ----------
    counts_df : pd.DataFrame
        The CPM-normalized barcode counts for each sample. Rows are barcodes, columns are samples.
        One column contains information on the library element associated with the barcode.
    rna_labels : np.array[str]
        Column names for the RNA samples.
    dna_label : str
        Column name for the DNA sample.
    title : str or None
        If specified, the title to use in the reproducibility and RNA/DNA vs DNA plots.
    output_prefix : str or None
        If specified, save the figures using this as the prefix.

    Returns
    -------
    counts_df : pd.DataFrame
        Rows are the same, but columns are now RNA/DNA ratios for each sample. The column with
        information on library element associations is preserved, but the DNA column is removed.
    reproducibility : (dict {str: float}, fig)
        Keys are pairs of samples, values are the r2 of the log2 RNA/DNA counts between samples.
        Second value of tuple is Figure handle.
    residual_dna_corr : (dict {str: float}, fig)
        Keys are RNA samples, values are the r2 between the log2 RNA/DNA counts and the DNA counts.
        Second value of tuple is Figure handle.
    """
    logger.info("Normalizing RNA to DNA.")
    dna_counts = counts_df[dna_label]
    counts_df = counts_df.drop(columns=dna_label)
    for rna in rna_labels:
        counts_df[rna] /= dna_counts

    reproducibility_title = "log2 RNA/DNA Counts"
    if title:
        reproducibility_title = f"{title}\n{reproducibility_title}"
    if output_prefix:
        reproducibility_figname = f"{output_prefix}RnaDnaLogReproducibility"
        rna_vs_dna_figname = f"{output_prefix}RnaDnaVsDna"
    else:
        reproducibility_figname = None
        rna_vs_dna_figname = None

    reproducibility = quality_control.reproducibility(
        counts_df[rna_labels],
        reproducibility_title,
        log=2,
        pseudocount=1e-3,
        figname=reproducibility_figname
    )

    residual_dna_corr = quality_control.rna_vs_dna(
        counts_df[rna_labels],
        dna_counts,
        "log2 RNA/DNA Counts",
        pseudocount=1e-3,
        title=title,
        figname=rna_vs_dna_figname,
    )

    return counts_df, reproducibility, residual_dna_corr


def setup_basal_norm(counts_df, basal_key, min_basal_median, pseudobasal_keys, output_prefix=None):
    """Prepare data for basal normalization, if necessary, based on the provided configurations.
    Options are no normalization, basal normalization, or pseudo-basal normalization using scrambled
    sequences. Returns a mask for the df indicating which rows are to be used for normalization.
    Additionally, plot the distribution of the log RNA/DNA barcode counts.

    Parameters
    ----------
    counts_df : pd.DataFrame
        RNA/DNA ratios for library members, rows are barcodes, columns are samples and names of
        library members.
    basal_key : str
        The library label for all barcodes corresponding to the basal sequence.
    min_basal_median : float
        If the median RNA/DNA ratio for basal barcodes is below this value in any replicate,
        consider the basal promoter poorly measured. All barcodes corresponding to the pseudobasal
        keys will be used instead to estimate basal activity.
    pseudobasal_keys : list[str]
        List of sequence labels to use as an alternative to basal activity. Ideally these should be
        scrambled sequences or other sequences expected to be similar to basal.
    output_prefix : str
        If specified, prefix for outputting the basal distribution.

    Returns
    -------
    basal_replicate_avg : pd.Series
        The mean (pseudo)basal RNA/DNA activity for each replicate.
    summary_stats : list of lists
        Each inner list is [sample name, "basal" or "pseudobasal", the basal average, and the number
        of barcodes used to get this value]
    """
    logger.info("Setting up basal normalization.")
    basal_bc_counts = counts_df[
        counts_df["label"].str.match(basal_key)
    ].drop(columns="label")
    # Plot the distribution of observed basal BCs, if desired
    if output_prefix:
        fig, ax_list = plot_utils.setup_multiplot(basal_bc_counts.shape[1], big_dimensions=False)
        ax_list = ax_list.flatten()
        for ax, sample in zip(ax_list, basal_bc_counts):
            ax.hist(np.log2(basal_bc_counts[sample]), bins="auto")
            ax.set_xlabel("log2 RNA/DNA")
            ax.set_ylabel("Num. Basal Barcodes")

        plot_utils.save_fig(fig, f"{output_prefix}BasalDistr")
        plt.close()

    medians = basal_bc_counts.median()
    logger.info(f"Median basal activity:\n{medians}")
    # If basal is poorly measured in any replicate, pseudobasal in all replicates, for consistency
    if (medians < min_basal_median).any():
        logger.info("Basal is too low! Falling back to pseudobasal.")
        method = "pseudobasal"
        basal_bc_counts = counts_df[
            counts_df["label"].isin(pseudobasal_keys)
        ].dropna()
    else:
        method = "basal"

    avg_basal = basal_bc_counts.dropna().mean()
    summary_stats = []
    for sample, metric in avg_basal.iteritems():
        summary_stats.append([sample, method, metric, len(basal_bc_counts)])

    logger.info(summary_stats)
    return avg_basal, summary_stats


def average_barcodes(counts_df, library_name, output_prefix=None):
    """Average across barcodes in each replicate and compute the reproducibility between replicates.
    If there is only one barcode per element, this is the same as the reproducibility of the RNA/DNA
    ratios.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Index are barcodes, columns are library member name and RNA/DNA ratios for each sample.
    library_name : str
        Name of the library, for the reproducibility plot.
    output_prefix : str or None
        Output prefix to use for the figure that is generated, if specified.

    Returns
    -------
    counts_df : pd.DataFrame
        Same as before, but now the index is the name of the library member and values are activity
        scores in each replicate.
    reproducibility : (dict {str: float}, fig)
        For each pair of replicates (keys), reports the R2 (values) of the activity scores once
        averaged across barcodes. Second value of tuple is Figure handle.
    """
    logger.info("Averaging across barcodes.")
    counts_df = counts_df.groupby("label").mean()
    if output_prefix:
        figname = f"{output_prefix}ActivityLogReproducibility"
    else:
        figname = None

    reproducibility = quality_control.reproducibility(
        counts_df,
        f"{library_name} log2 Activity",
        log=2,
        pseudocount=1e-3,
        figname=figname
    )
    return counts_df, reproducibility


def _aggregate_replicates(counts_df):
    """
    Helper function for replicate_average that takes a set of measurements taken across conditions
    and returns the mean, std, count, and lognormal distribution parameters.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Activity scores for each library member in each sample. Rows are library members, columns
        are samples.

    Returns
    -------
    agg_df : pd.DataFrame
        Summary statistics for each sequence.
    """
    function_renamer = {
        "mean": "expression",
        "std": "expression_std",
        "count": "expression_reps",
    }
    aggfuncs = ["mean", "std", "count"]
    agg_df = counts_df.agg(aggfuncs, axis=1).rename(function_renamer, axis=1)
    # Get lognormal parameters
    params = stat_utils.lognormal_params(agg_df[["expression", "expression_std"]])
    agg_df = agg_df.join(params)
    return agg_df


def replicate_average(counts_df, basal_replicate_avg=None):
    """
    Average across replicates and report summary statistics, including the parameters of the
    underlying log-normal distribution. Additionally, Welch's t-test is performed to determine if
    each sequence has the same activity as basal and each replicate is normalized to the basal mean,
    if specified. These steps are performed before averaging across replicates.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Activity scores for each library member in each sample. Rows are library members, columns
        are samples.
    basal_replicate_avg : pd.Series or None
        If specified, a Series with the average basal levels in each sample. Statistics are
        performed to determine if sequences have the same activity as basal, and each replicate is
        normalized to basal before averaging across replicates. Otherwise, no statistics or
        normalization are calculated.

    Returns
    -------
    activity_df : pd.DataFrame
        Summary statistics of each sequence activity averaged across replicates. Rows are library
        members, columns are the (possibly basal-normalized) mean, the standard deviation, the
        number of replicates, the log2 mean activity, the mu and sigma for the lognormal
        distribution (for downstream statistics), and if testing for the same activity as basal, the
        p-value and FDR-corrected q-value for the statistical test.
    """
    logger.info("Averaging across barcodes within each replicate.")
    activity_df = _aggregate_replicates(counts_df)
    # Do statistics vs basal and normalize to basal, if necessary
    if type(basal_replicate_avg) is pd.Series:
        logger.info("Comparing activities to basal.")
        basal_mean, basal_std, basal_n = basal_replicate_avg.agg(["mean", "std", "count"])
        # Get basal lognormal
        basal_mu, basal_sigma = stat_utils.lognormal_params((basal_mean, basal_std))
        pvals = activity_df[["mu", "sigma", "expression_reps"]].apply(
            lambda x: stats.ttest_ind_from_stats(
                x["mu"],
                x["sigma"],
                x["expression_reps"],
                basal_mu,
                basal_sigma,
                basal_n,
                equal_var=False
            )[1],
            axis=1
        )
        # Correct for multiple hypotheses
        qvals = stat_utils.fdr(pvals)
        # FIXME think about if error propogation is necessary
        # Basal normalize within each replicate, then recompute mean, std, and lognormal parameters
        counts_df /= basal_replicate_avg
        activity_df = _aggregate_replicates(counts_df)
        activity_df["pval"] = pvals
        activity_df["qval"] = qvals

    activity_df["expression_log2"] = np.log2(activity_df["expression"])

    return activity_df


def process_library(configs, library_name, min_dna_counts, max_rna_cv, basal_key, min_basal_median,
                    pseudobasal_keys, output_prefix=None):
    """
    Outer function to process barcode counts into activity scores.

    Parameters
    ----------
    configs : dict
        The dictionary of filenames to process the library.
    library_name : str
        The name of the library, to use in figures and filenames.
    min_dna_counts : int
        Minimum number of DNA barcode counts for a barcode to be considered present.
    max_rna_cv : int
        Threshold for removing barcodes from RNA. If a barcode's sample coefficient of variation is
        above this threshold, the barcode is removed from analysis.
    basal_key : str
        The library label for all barcodes corresponding to the basal sequence.
    min_basal_median : float
        If the median RNA/DNA ratio for basal barcodes is below this value in any replicate,
        consider the basal promoter poorly measured. All barcodes corresponding to the pseudobasal
        keys will be used instead to estimate basal activity.
    pseudobasal_keys : list[str]
        List of sequence labels to use as an alternative to basal activity. Ideally these should be
        scrambled sequences or other sequences expected to be similar to basal.
    output_prefix : str
        If specified, the prefix (including a potential path) to give to all outputs. Otherwise, use
        the library name.

    Returns
    -------
    activity_df : pd.DataFrame
        Summary statistics of the activity for each sequence, including the mean, std, n, lognormal
        parameters, and possibly statistical tests for if the activity is the same as basal. Index
        is the sequence identifier.
    qc_metrics : pd.DataFrame
        QC metrics obtained throughout the data processing.

    """
    if output_prefix is None:
        output_prefix = library_name

    # Initialize dictionary to hold performance/QC metrics
    qc_metrics = dict()
    # For now assume there is only one DNA sample
    counts_df, rna_labels, dna_label = load_counts_files(configs)
    sample_labels = np.append(rna_labels, [dna_label])
    logger.info("Checking coverage and initial reproducibility.")
    qc_metrics["coverage"] = quality_control.report_coverage(counts_df)

    # Plot histograms of barcode counts for each sample
    fig = plot_barcode_distributions(
        counts_df[sample_labels],
        dna_label,
        min_dna_counts,
        title=library_name,
        figname=f"{output_prefix}CountsHistogram",
    )
    plt.close(fig)

    # Plot linear and log reproducibility, report R2
    qc_metrics["raw_linear_reprod"], fig = quality_control.reproducibility(
        counts_df[rna_labels],
        f"{library_name} Raw Barcode Counts",
        log=None,
        figname=f"{output_prefix}RawReproducibility",
    )
    plt.close(fig)
    qc_metrics["raw_log_reprod"], fig = quality_control.reproducibility(
        counts_df[rna_labels],
        f"{library_name} log10 Raw Barcode Counts",
        log=10,
        pseudocount=1,
        figname=f"{output_prefix}RawLogReproducibility",
    )
    plt.close(fig)

    counts_df, qc_metrics["bc_removed"] = filter_low_counts(
        counts_df, min_dna_counts, dna_label, rna_labels, max_rna_cv, output_prefix=output_prefix,
    )

    # Add a pseudocount to the RNA samples before CPM normalization
    counts_df[rna_labels] += 1
    counts_df = counts_per_million(counts_df, sample_labels)

    # Plot linear and log reproducibility again
    logger.info("Checking reproducibility again.")
    qc_metrics["cpm_linear_reprod"], fig = quality_control.reproducibility(
        counts_df[rna_labels],
        f"{library_name} Barcode CPM",
        log=None,
        figname=f"{output_prefix}CpmReproducibility",
    )
    plt.close(fig)
    qc_metrics["cpm_log_reprod"], fig = quality_control.reproducibility(
        counts_df[rna_labels],
        f"{library_name} log2 Barcode CPM",
        log=2,
        figname=f"{output_prefix}CpmLogReproducibility",
    )
    plt.close(fig)

    # Check log RNA vs log DNA
    qc_metrics["rna_vs_dna_corr"], fig = quality_control.rna_vs_dna(
        counts_df[rna_labels],
        counts_df[dna_label],
        "log2 RNA counts",
        title=library_name,
        figname=f"{output_prefix}RnaVsDna",
    )
    plt.close(fig)

    # Normalize to DNA, check reproducibility, check RNA/DNA vs DNA ratio, and drop DNA.
    counts_df, rna_vs_dna, ratio_vs_dna = dna_normalize(
        counts_df,
        rna_labels,
        dna_label,
        title=library_name,
        output_prefix=output_prefix
    )
    qc_metrics["rna/dna_log_corr"], fig = rna_vs_dna
    plt.close(fig)
    qc_metrics["rna_dna_vs_dna_corr"], fig = ratio_vs_dna
    plt.close(fig)

    # Setup basal normalization, if necessary, and check the distribution.
    basal_replicate_avg, qc_metrics["basal_distribution"] = setup_basal_norm(
        counts_df, basal_key, min_basal_median, pseudobasal_keys, output_prefix=output_prefix)

    # Average across barcodes, if necessary
    counts_df, activity_r2 = average_barcodes(counts_df, library_name, output_prefix)
    qc_metrics["activity_log_reprod"], fig = activity_r2
    plt.close(fig)

    # If necessary, compute statistics for sequences having the sample activity vs basal and basal
    # normalize. Then, regardless, average across replicates.
    activity_df = replicate_average(counts_df,
                                    basal_replicate_avg=basal_replicate_avg,
                                    )
    activity_df = activity_df.rename_axis(index="label")
    # Make each value of the dictionary a df
    qc_metrics = {
        k: pd.DataFrame(v, columns=["comparison", "method", "value", "n"])
        for k, v in qc_metrics.items()
    }
    # Now concat them. The keys of the dict (i.e. the QC metric) will be the first of two levels on
    # the index. We want to put that onto a column
    qc_metrics = pd.concat(qc_metrics, names=["qc_metric", None]).reset_index("qc_metric")

    # Write to file
    if configs["write_output"]:
        loaders.write_data(activity_df, f"{output_prefix}Activity.txt")
        loaders.write_data(qc_metrics, f"{output_prefix}Quality.txt", index=False)

    return activity_df, qc_metrics


def process_each_library(configs):
    """
    Process multiple libraries, each one separately. After performing the processing for a library,
    add a column to the activity_df returned by process_library indicating the library name, and
    then join everything together into one large DataFrame. In addition, join together the FASTA
    files of all libraries into one big FASTA file, removing any duplicate sequences.

    Parameters
    ----------
    configs : dict
        Nested dictionary representation of the YAML file of configs. All libraries are under the
        `library` key, and each value is the name of a library and the dictionary needed to process
        the data.

    Returns
    -------
    joined_activity_df : pd.DataFrame
        The result of process_library for each library, joined together into one large DataFame.
    all_seqs : pd.Series
        FASTA file representation with all unique sequences tested across all libraries.
    """
    joined_activity_df = []
    all_seqs = []
    joined_qc = []

    # Global parameters to use in each library
    kwargs = {
        k: configs[k] for k in ["min_dna_counts", "max_rna_cv", "basal_key", "min_basal_median"]
    }
    output_dir = configs["output_dir"]
    scrambled_seqs = fasta_utils.read_fasta(configs["scrambled_fasta"])
    scrambled_labels = scrambled_seqs.index.values

    for library_name, library_configs in configs["libraries"].items():
        logger.info(f"Processing {library_name}")
        library_dir = os.path.join(output_dir, library_name.capitalize())
        if not os.path.exists(library_dir):
            os.mkdir(library_dir)
        activity_df, qc_metrics = process_library(
            library_configs,
            library_name,
            output_prefix=os.path.join(library_dir, library_name),
            pseudobasal_keys=scrambled_labels,
            **kwargs,
        )
        activity_df["library"] = library_name
        sequences = fasta_utils.read_fasta(library_configs["fasta"])

        joined_activity_df.append(activity_df)
        all_seqs.append(sequences)
        qc_metrics["library"] = library_name
        joined_qc.append(qc_metrics)

    joined_activity_df = pd.concat(joined_activity_df)
    all_seqs = pd.concat(all_seqs).drop_duplicates()
    joined_qc = pd.concat(joined_qc)
    # Remove sequences only measured in one replicate
    joined_activity_df = joined_activity_df[joined_activity_df["expression_reps"] > 1]

    return joined_activity_df, all_seqs, joined_qc


def discretize_activity(x):
    """Convert a quantitative log2 basal-normalized activity score into a discretized class."""
    # Silencer = (-inf, -1)
    if x < -1:
        res = "Silencer"
    # Inactive = [-1, 1]
    elif x <= 1:
        res = "Inactive"
    # Weak enhancer = (1, 2.838...). Upper cutoff based on the number used in eLife paper, see
    # notebook 4 on the Github
    elif x < loaders.get_strong_cutoff():
        res = "WeakEnhancer"
    # Strong enhancer = [2.838..., inf)
    else:
        res = "StrongEnhancer"
    return res
