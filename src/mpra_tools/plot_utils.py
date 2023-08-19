"""Various wrapper functions to make custom styles of plots.

"""

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logomaker

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc

from . import predicted_occupancy, deeplift_utils


def set_presentation_params():
    """Set the matplotlib rcParams to values for presentation-size figures.
    
    """
    mpl.rcParams["axes.titlesize"] = 25
    mpl.rcParams["axes.labelsize"] = 20
    mpl.rcParams["xtick.labelsize"] = 15
    mpl.rcParams["ytick.labelsize"] = 15
    mpl.rcParams["legend.fontsize"] = 15
    mpl.rcParams["figure.figsize"] = (8, 8)
    mpl.rcParams["image.cmap"] = "viridis"
    mpl.rcParams["lines.markersize"] = 3
    mpl.rcParams["lines.linewidth"] = 3
    mpl.rcParams["font.size"] = 15
    mpl.rcParams["savefig.pad_inches"] = 0
    mpl.rcParams["savefig.facecolor"] = "white"


def set_print_params():
    """Set the matplotlib rcParams to values for print-size figures.

    """
    mpl.rcParams["figure.figsize"] = (4, 4)
    mpl.rcParams["axes.titlesize"] = 15
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["image.cmap"] = "viridis"
    mpl.rcParams["lines.markersize"] = 1.25
    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["savefig.dpi"] = 200
    mpl.rcParams["savefig.pad_inches"] = 0
    mpl.rcParams["savefig.facecolor"] = "white"
    
    
def set_manuscript_params():
    """Set the matplotlib rcParams to values for manuscript-quality figures. Adapted from the BPNet manuscript https://github.com/kundajelab/bpnet-manuscript/blob/master/basepair/plot/config.py#L31"""
    plt.rcdefaults()
    plt.style.use("seaborn-paper")
    
    paper_style = {
        "font.family": "sans-serif",
         "font.sans-serif": "Arial",
        # Use the same font for mathtext
        "axes.formatter.use_mathtext": True,
        "mathtext.default": "regular",
        
        "axes.grid": False,
        "axes.axisbelow": True,   # Sets axis gridlines and ticks below
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "font.size": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        
        "lines.linewidth": 0.8,
        "lines.markersize": 2,
        # Boxplot markers need to be smaller too!
        "boxplot.flierprops.markersize": 3,
        "boxplot.flierprops.markerfacecolor": "black",
        "boxplot.flierprops.markeredgecolor": "none",
        "boxplot.medianprops.color": "black",
        "boxplot.showcaps": False,
        "boxplot.showfliers": False,
        
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        "lines.antialiased": True,
        "patch.antialiased": True,
        
        "image.cmap": "viridis",
        
        # Figure export
        "pdf.fonttype": 42,   # Save text as text
        "ps.fonttype": 42,
        # Don't cross the borders
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,    # Want very small padding to the plot
        "savefig.facecolor": "white",    # Life is easier when there is a white background. Can always take this out myself.
    }
    
    plt.style.use(paper_style)
    

def hex_to_rgb(hexcode):
    """Convert a hex code into (R, G, B) values as a fraction of 255."""
    return tuple(int(hexcode[i:i+2], 16) / 255 for i in (1, 3, 5))


def rotate_ticks(ticks, rotation=90, ha="center"):
    """Rotate tick labels from an Axes object after the ticks were already generated.

    Parameters
    ----------
    ticks : list[Text]
        The tick labels to rotate
    rotation : int or float
        The angle to set for the tick labels
    ha : str
        Horizontal alignment

    Returns
    -------
    None
    """
    for tick in ticks:
        tick.set_rotation(rotation)
        tick.set_ha(ha)


def set_color(values):
    """A wrapper for converting numbers into colors. Given a number between 0 and 1, convert it to the corresponding color in the color scheme.
    
    """
    my_cmap = mpl.cm.get_cmap()
    return my_cmap(values)


def save_fig(fig, prefix, tight_layout=False, tight_pad=1.08, file_type="svg"):
    """Save a figure as the specified file type, defaulting to svg.
    
    """
    if tight_layout:
        fig.tight_layout(pad=tight_pad)
    if file_type.lower() not in ["png", "svg", "jpg", "jpeg", "tif", "tiff", "pdf", "eps"]:
        print(f"Warning, did not recognize file type, defaulting to svg.")
        file_type = "svg"
    fig.savefig(f"{prefix}.{file_type}")

    
def get_figsize(frac=1, aspect=0.618, width=174):
    """Set aesthetic figure dimensions. Adapted from Kundaje lab https://github.com/kundajelab/bpnet-manuscript/blob/master/basepair/plot/config.py#L6
    Appears to be commonly used for the figsize argument of plt.subplots, particularly when generating locus plots.
    
    Parameters
    ----------
    width : int
        Total page width in mm
    frac : float
        What fraction of the page width should the figure take
    ratio : float
        What aspect ratio (width/height) the figure should be.
    
    Returns
    -------
        Dimensions of the figure in inches.
    """
    width = width * 0.03937 * frac   # convert to inches + scale
    return (width, width * aspect)


def get_class_colors():
    """Wrapper function to return a Series that maps the 4 activity classes to hexcode colors."""
    return pd.Series({
        "Silencer": "#e31a1c",
        "Inactive": "#33a02c",
        "WeakEnhancer": "#a6cee3",
        "StrongEnhancer": "#1f78b4",
    })
    
    
def setup_multiplot(n_plots, n_cols=2, sharex=True, sharey=True, big_dimensions=True):
    """Setup a multiplot and hide any superfluous axes that may result.

    Parameters
    ----------
    n_plots : int
        Number of subplots to make
    n_cols : int
        Number of columns in the multiplot. Number of rows is inferred.
    sharex : bool
        Indicate if the x-axis should be shared.
    sharey : bool
        Indicate if the y-axis should be shared.
    big_dimensions : bool
        If True, then the size of the multiplot is the default figure size multiplied by the number of rows/columns.
        If False, then the entire figure is the default figure size.

    Returns
    -------
    fig : figure handle
    ax_list : list-like
        The list returned by plt.subplots(), but any superfluous axes are removed and replaced by None
    """
    n_rows = int(np.ceil(n_plots / n_cols))
    row_size, col_size = mpl.rcParams["figure.figsize"]

    if big_dimensions:
        # A bit counter-intuitive...the SIZE of the row is the width, which depends on the number of columns
        row_size *= n_cols
        col_size *= n_rows

    fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(row_size, col_size), sharex=sharex, sharey=sharey)

    # The index corresponding to n_plots is the first subplot to be hidden
    for i in range(ax_list.size):
        coords = np.unravel_index(i, ax_list.shape)
        ax = ax_list[coords]
        if i >= n_plots:
            ax.remove()
            ax_list[coords] = None

    return fig, ax_list


def volcano_plot(df, x_col, y_col, colors, alpha=1, xaxis_label=None, yaxis_label=None, title=None, figname=None,
                 xline=None, yline=None, xticks=None, vmin=None, vmax=None, cmap=None, colorbar=False, figax=None):
    """Make a volcano plot, without transforming the x-axis but taking -log10 of the y-axis. Assign different points
    different colors to highlight different classes.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str
        Column of the df to plot on x
    y_col : str
        Column of the df to plot on y. Take -log10 of this column before plotting
    colors : list-like
        Indicates color to use for each row of df.
    alpha : float
        Opacity of the points.
    xaxis_label : str
        If specified, the label for the x-axis. Otherwise use x_col.
    yaxis_label : str
        If specified, the label for the y-axis. Otherwise use y_col.
    title : str
        If specified, make a title for the plot.
    figname : str
        If specified, save the figure with this name.
    xline : int or float or list
        If specified, plot a dashed vertical line at x = xline
    yline : int or float or list
        If specified, plot a dashed horizontal line at y = yline
    xticks : list
        If specified, set the x ticks to these values.
    vmin : int or float
        If specified, minimum value for the colormap.
    vmax : int or float
        If specified, maximum value for the colormap.
    cmap : str
        If specified, use this colormap. Otherwise, use the default.
    colorbar : bool
        If True, display a colorbar to the right.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.

    Returns
    -------
    fig : Figure handle
    """
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    # Prepare the data
    x = df[x_col]
    y = -np.log10(df[y_col])
    scatter_kwargs = {"c": colors, "alpha": alpha}
    if vmin:
        scatter_kwargs["vmin"] = vmin
    if vmax:
        scatter_kwargs["vmax"] = vmax
    if cmap:
        scatter_kwargs["cmap"] = cmap

    scatterplot = ax.scatter(x, y, **scatter_kwargs)

    # Default axis labels if none specified
    if not xaxis_label:
        xaxis_label = x_col
    if not yaxis_label:
        yaxis_label = f"-log10 {y_col}"

    # Add dotted lines if specified
    line_kwargs = {"linestyle": "--", "color": "black"}
    if xline is not None:
        if type(xline) is list:
            for xl in xline:
                ax.axhline(xl, **line_kwargs)
        else:
            ax.axhline(xline, **line_kwargs)
    if yline is not None:
        if type(yline) is list:
            for yl in yline:
                ax.axvline(yl, **line_kwargs)
        else:
            ax.axvline(yline, **line_kwargs)

    # Axis labels, ticks, colorbar, title if specified
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)

    if xticks is not None:
        ax.set_xticks(xticks)

    if colorbar:
        fig.colorbar(scatterplot, orientation="vertical")

    if title:
        ax.set_title(title)

    if figname:
        save_fig(fig, figname)

    return fig


def scatter_with_corr(x, y, xlabel, ylabel, colors="black", xticks=None, yticks=None, loc=None, figname=None,
                      alpha=1.0, figax=None, reproducibility=False, rasterize=False, **kwargs):
    """Make a scatter plot and display the correlation coefficients in a specified location.

    Parameters
    ----------
    x : list-like
        Data to plot on the x axis.
    y : list-like
        Data to plot on the y axis.
    xlabel : str
        Label for the x axis.
    ylabel : str
        Label for the y axis.
    colors : "density", str or list-like
        If "density", color points based on point density in 2D space. If another str, make every point the same
        color. If list-like, specifies the color for each point.
    xticks : list-like
        If specified, set the x axis ticks to these values.
    yticks: list-like
        If specified, set the y axis ticks to these values.
    loc : str or None
        The location of the plot to display the correlations, must be one of "upper left", "upper right",
        "lower left", or "lower right". If some other string, assume "lower right".
    figname : str
        If specified, save the figure with this name.
    alpha : float
        Alpha (opacity) of the points.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    reproducibility : bool
        If True, the plot is a reproducibility plot and the text should show the r2. If False, instead show the
        Pearson and Spearman correlation, and the n.
    rasterize : bool
        If True, rasterize the dots.
    kwargs : dict
        Arguments for saving the figure.

    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    correlations : (float, float)
        The Pearson and Spearman correlation coefficients.
    """
    pcc, _ = stats.pearsonr(x, y)
    scc, _ = stats.spearmanr(x, y)
    n = len(x)
    correlations = (pcc, scc)
    if reproducibility:
        text = fr"$r^2$={pcc**2:.3f}"
    else:
        text = f"PCC = {pcc:.3f}\nSCC = {scc:.3f}\nn = {n}"

    # Calculate the density to display on the scatter plot, if specified
    if type(colors) is str and colors == "density":
        xy = np.vstack([x, y])
        colors = stats.gaussian_kde(xy)(xy)
        order = colors.argsort()
        x, y, colors = x[order], y[order], colors[order]
        colors = set_color(colors / colors.max())

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    ax.scatter(x, y, color=colors, alpha=alpha, rasterized=rasterize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # Parse info on location
    if loc:
        yloc, xloc = loc.split()
        if yloc == "upper":
            yloc = 0.98
            va = "top"
        else:
            if yloc != "lower":
                print("Warning, did not recognize yloc, assuming lower")
            yloc = 0.02
            va = "bottom"

        if xloc == "left":
            xloc = 0.02
            ha = "left"
        else:
            if xloc != "right":
                print("Warning, did not recognize xloc, assuming right")
            xloc = 0.98
            ha = "right"

        ax.text(xloc, yloc, text, ha=ha, va=va, transform=ax.transAxes)

    if figname:
        save_fig(fig, figname, **kwargs)

    return fig, ax, correlations


def make_heatmap(data, xticklabels=None, yticklabels=None, xtick_angle=None, title=None, annotate_thresh=None,
                 cbar_label=None, figax=None, **kwargs):
    """Make a heatmap with the provided data. Optionally add x and/or y tick labels, a title to the heatmap,
    numeric annotations of the heatmap, and a colorbar to the right.

    Parameters
    ----------
    data : pd.DataFrame, ndarray, or list of lists
        2D matrix containing the values for the heatmap with M rows and N columns. Can also be 3D with shape (M, N,
        3) with RGB values for the heatmap.
    xticklabels : list[str] or None
        If specified, the values to put on the x ticks. Otherwise, show no ticks.
    yticklabels : list[str] or None
        If specified, the values to put on the y ticks. Otherwise, show no ticks.
    xtick_angle : int or None
        If specified, rotate the x ticks by the specified amount. Only used if xticklabels is also specified.
    title : str or None
        If specified, the title for the heatmap.
    annotate_thresh : float
        If specified, annotate the heatmap with the numeric values for each cell. This value represents the threshold
        for the text to switch from a dark color to a light color. Usually the midpoint between vmin and vmax.
    cbar_label : str or None
        If specified, create a colorbar to the right of the heatmap and use this value as the label for the bar.
    figax : (Figure, Axes)
        If provided, place the heatmap in a pre-existing Axes object with an associated Figure. Otherwise,
        create it in this function.
    kwargs : dict
        Other arguments for creating the heatmap, e.g. the cmap, vmin and vmax, etc.

    Returns
    -------
    fig : Figure
        Handle to the figure object
    ax : Axes
        Handle to the axes object with the heatmap
    """
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    heatmap = ax.imshow(data, interpolation="none", **kwargs)

    # Create ticks
    if xticklabels is None:
        ax.set_xticks([])
    else:
        xticks = np.arange(len(xticklabels))
        ax.set_xticks(xticks)
        tick_kwargs = {}
        if xtick_angle:
            tick_kwargs["rotation"] = xtick_angle

        ax.set_xticklabels(xticklabels, **tick_kwargs)
    if yticklabels is None:
        ax.set_yticks([])
    else:
        yticks = np.arange(len(yticklabels))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    if title:
        ax.set_title(title)

    # Annotate
    if annotate_thresh:
        annotate_heatmap(ax, data.T, annotate_thresh)

    # Add colorbar
    if cbar_label:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
        fig.colorbar(heatmap, cax=cax, label=cbar_label)

    return fig, ax


def plot_deeplift(scores, smoothing=False, seq_name=None, figax=None, window=5):
    """Visualize saliency maps or DeepLIFT importance scores as a sequence logo.

    Parameters
    ----------
    scores : array-like, shape = (seq_len, 4)
        DeepLIFT importance scores (hypothetical or actual) for each base at each position. Can handle arrays that
        are transposed, but expects the characters to be on the 1st axis.
    smoothing : bool
        If True, show the smoothed actual importance as specified in deeplift_utils.smooth_importance_scores. Note this should only be done using actual importance scores!
    seq_name : str or None
        If specified, name of the sequence to display on the figure.
    figax : (Figure, Axes)
        Optional, pre-created handles for the figure and axes to plot the profile on. Otherwise use get_figsize with aspect ratio of 0.2
    window : int
        If smoothing, the window size to smooth over.

    Returns
    -------
    fig, ax : Handles to the Figure and Axes objects associated with the logo.
    """
    # If the columns are not the bases, see if we can transpose it
    if scores.shape[1] != 4:
        if scores.shape[0] == 4:
            scores = scores.T
        else:
            raise ValueError(f"Expected one of the axes to have shape 4, shape of scores is {scores.shape}")

    scores = pd.DataFrame(scores, columns=["A", "C", "G", "T"])
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=get_figsize(0.5, aspect=0.2))
        ax.set_ylabel("Contrib. score")

    logomaker.Logo(scores, color_scheme="colorblind_safe", ax=ax)
    ax.axhline(0, color="k", lw=1)
    # Display the sequence name
    ax.set_title(seq_name, fontsize="small")   
    if smoothing:
        smoothed = deeplift_utils.smooth_importance_scores(scores, window=window)
        ax.plot(smoothed, color="k")
        ax.set_ylim(top=smoothed.max())

    return fig, ax


def show_confusion_matrix(confusion, class_names=None, vmax=None, cmap="Reds", title=None, figax=None):
    """
    Visualize a confusion matrix as a heatmap.

    Parameters
    ----------
    confusion : matrix-like
        A square confusion matrix with the true labels on the rows and predicted label on the columns.
    class_names : list[str] or None
        If specified, the names of the classes to show on the ticks. Otherwise just show int values.
    vmax : int or None
        If specified, maximum value for the cmap. Otherwise just use the largest value in the matrix.
    cmap : str
        Colormap to use for the heatmap. Reds by default.
    title : str or None
        If specified, title for the confusion matrix. Otherwise don't show anything.
    figax : (Figure, Axes)
        Optional, pre-created handles for the figure and axes to plot the profile on.

    Returns
    -------
    fig, ax : Handles to the Figure and Axes objects associated with the heatmap.
    """
    confusion = pd.DataFrame(confusion)

    if figax is None:
        figax = plt.subplots()

    if vmax is None:
        vmax = confusion.max().max()

    if class_names is None:
        class_names = np.arange(len(confusion))

    fig, ax = make_heatmap(
        confusion,
        xticklabels=class_names,
        yticklabels=class_names,
        figax=figax,
        cmap=cmap,
        title=title,
        vmax=vmax,
        vmin=0,
        annotate_thresh=vmax/2,
        cbar_label="Number of sequences"
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    return fig, ax


def visualize_sequence(sequence, basecolor, ewms, mu, motif_colors, occ_cutoff=0.5, above_text=None, below_text=None,
                       figax=None):
    """Visualize where in a sequence motifs are located. First, do predicted occupancy on the sequence to identify
    where the motifs are located. Then, make a graphic that shows the location and name of those motifs, potentially
    coloring motifs differently.

    Parameters
    ----------
    sequence : str
        The DNA sequence to visualize.
    basecolor : tuple, (float, float, float)
        RGB values between 0 and 1 to set as the main color of the sequence.
    ewms : pd.Series[dict[dict]]
        EWMs of TFs for predicted occupancy scans.
    mu : int or float
        Chemical potential of TFs for predicted occupancy.
    motif_colors : str, tuple, dict or pd.Series, {str: tuple}
        The color to use for different motifs. If a string or tuple, use the same color for all motifs. If a dict or
        Series, provides a mapping from each TF to the color to use.
    occ_cutoff : float
        Threshold for calling a position occupied by a TF.
    above_text : str
        If specified, text to put as the "title" above the sequence.
    below_text : str
        If specified, text to put as the "x label" below the sequence.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes using get_figsize with an aspect ratio of 0.1.

    Returns
    -------
    fig, ax : The handles to the Figure and Axes.
    """
    occupancy_df = predicted_occupancy.total_landscape(sequence, ewms, mu)
    site_map = predicted_occupancy.get_occupied_sites_and_tfs(occupancy_df, cutoff=occ_cutoff)
    
    if figax is None:
        fig, ax = plt.subplots(figsize=get_figsize(0.5, aspect=0.1))
        ax.set_yticks([])
    else:
        fig, ax = figax
    ax.set_xlim((0, len(sequence)))
    ax.set_facecolor(basecolor)
    
    for motif_start, tf in site_map.items():
        tf = tf.split("_")[0]
        motif_end = motif_start + len(ewms[tf])
        if type(motif_colors) is str or type(motif_colors) is tuple:
            color = motif_colors
        else:
            color = motif_colors[tf]
        ax.axvspan(motif_start, motif_end, facecolor=color, edgecolor="black", zorder=3)
        ax.text(np.mean((motif_start, motif_end)), np.mean(ax.get_ylim()), tf, ha="center", va="center", color="white", rotation=90)

    if above_text:
        ax.set_title(above_text)
    if below_text:
        ax.set_xlabel(below_text)

    return fig, ax


def violin_plot_groupby(grouper, yname, class_names=None, class_colors=None, alpha=1.0, transformation_function=None,
                        pseudocount=0, figname=None, vert=True, yticks=None, figax=None, **kwargs):
    """Make a violin plot from a groupby object.

    Parameters
    ----------
    grouper : pd.DataFrameGroupBy or pd.SeriesGroupBy
        Group by object where each group is data for a different violin.
    yname : str
        Name for the y axis
    class_names : list
        Optional names for each group. If not specified, use the names from the grouper
    class_colors : list
        Optional colors for each group.
    alpha : float
        Opacity of the violins.
    transformation_function : function handle
        Optional transformation to apply to the data.
    pseudocount : int or float
        Optional pseudocount for the data.
    figname : str
        If specified, save the figure to a file with this name.
    vert : bool
        If True, violins are vertical. Otherwise, violins are horizontal.
    yticks : list
        If specified, indicates the ticks for the y axis.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    kwargs : dict
        Arguments for saving the figure

    Returns
    -------
    fig : figure handle
    """
    names, data = zip(*[(i, j) for i, j in grouper if len(j) > 0])
    if class_names:
        names = class_names

    fig = _make_violin_plot(data, names, yname, colors=class_colors, alpha=alpha,
                            transformation_function=transformation_function, pseudocount=pseudocount,
                            figname=figname, vert=vert, yticks=yticks, figax=figax, **kwargs)
    return fig


def _make_violin_plot(data_values, x_labels, y_label, colors=None, alpha=1.0, transformation_function=None,
                      pseudocount=0, figname=None, vert=True, yticks=None, whisker=1.5, figax=None, **kwargs):
    """Helper function to make violin plots"""
    # Transform the data (e.g. take the log10) if necessary
    if transformation_function:
        data_values = [transformation_function(i + pseudocount) for i in data_values]
    xaxis = np.arange(len(x_labels)) + 1

    # Set the color to grey for everything if colors aren't specified
    if colors is None:
        colors = ["grey"] * len(x_labels)

    # Separate outliers from the rest so we don't use them in the KDE
    class_quartiles = np.array([np.percentile(i, [25, 50, 75]) for i in data_values])
    class_iqrs = class_quartiles[:, 2] - class_quartiles[:, 0]
    class_whisker = class_iqrs * whisker
    outlier_masks = [(group_data > quartiles[2] + whisk) | (group_data < quartiles[0] - whisk)
            for group_data, quartiles, whisk in zip(data_values, class_quartiles, class_whisker)]
    outlier_data = [group_data[group_mask] for group_data, group_mask in zip(data_values, outlier_masks)]
    main_data = [group_data[~group_mask] for group_data, group_mask in zip(data_values, outlier_masks)]

    # Plot the data and color the violins accordingly.
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    parts = ax.violinplot(main_data, vert=vert)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(alpha)

    # Clean up the plot
    parts["cmins"].remove()
    parts["cmaxes"].remove()
    parts["cbars"].remove()

    # Boxplot the median, IQR, and 1.5IQR whiskers
    ax.boxplot(data_values, vert=vert)
    
    # Plot outliers
    for x, c, outliers in zip(xaxis, colors, outlier_data):
        jittered = np.random.normal(x, 0.04, len(outliers))
        # Figure out the alpha as follows:
        # Figure out how many outliers there are
        # Take the log10, and then the floor of that, to figure out the order of magnitude
        # The alpha is 1/(2^(magnitude-1)), or 0.1 as a maximum.
        a = np.floor(np.log10(len(outliers) + 1))
        a = 2**-a
        a = max(a, 0.01)
        jit_kwargs = dict(
            c=c,
            alpha=a,
            s=mpl.rcParams["boxplot.flierprops.markersize"]
        )
        if vert:
            ax.scatter(jittered, outliers, **jit_kwargs)
        else:
            ax.scatter(outliers, jittered, **jit_kwargs)        
    
    # Formatting
    if vert:
        ax.set_ylabel(y_label)
        ax.set_xticks(xaxis)
        ax.set_xticklabels(x_labels)
        if yticks is not None:
            ax.set_yticks(yticks)
    else:
        ax.set_xlabel(y_label)
        ax.set_yticks(xaxis)
        ax.set_yticklabels(x_labels)
        if yticks is not None:
            ax.set_xticks(yticks)

    fig.tight_layout()
    if figname:
        save_fig(fig, figname, **kwargs)

    return fig


def multi_hist(df, column_list, xlabel, ylabel, n_cols=2, transform=None, sharex=True, sharey=True, bins=10,
               pseudocount=0, figname=None, big_dimensions=True):
    """Make a figure with multiple subplots, each subplot containing a histogram for a different column of the
    dataframe. Optionally add a pseudocount and transform the data before plotting.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot
    column_list : list-like
        Column names to plot. Each column is plotted on a separate histogram.
    xlabel : str
        Label for the x-axis of the plots
    ylabel : str
        Label for the y-axis of the plots
    n_cols : int
        Number of columns in the multiplot
    transform : function handle
        If specified, add a pseudocount to the data and then apply the transformation function.
    sharex : bool
        Indicates if the x-axis should be shared across subplots.
    sharey : bool
        Same as sharex for y-axis.
    bins : int
        Number of bins for the histogram.
    pseudocount : int or float
        Add a pseudocount to the data if a transformation function is specified.
    figname : str
        If specified, save the figure with this name.
    big_dimensions : bool
        If True, then the size of the multiplot is the default figure size multiplied by the number of rows/columns.
        If False, then the entire figure is the default figure size.

    Returns
    -------
    fig : Figure handle
    """
    n_plots = len(column_list)
    fig, ax_list = setup_multiplot(n_plots, n_cols=n_cols, sharex=sharex, sharey=sharey, big_dimensions=big_dimensions)
    if len(ax_list.shape) == 1:
       ax_list = np.reshape(ax_list, (len(ax_list), 1)) 
    
    n_rows, _ = ax_list.shape # Used for the x axis display

    for i in range(n_plots):
        row, col = np.unravel_index(i, ax_list.shape)
        ax = ax_list[row, col]
        label = column_list[i]

        # Get rid of any NaN in the data since this is different from a zero
        data = df[label]
        data = data[data.notna()]

        if transform:
            data = transform(data + pseudocount)

        ax.hist(data, bins)
        ax.set_title(label)

        # Add axis labels if the axis is not shared or the axis is shared and on the appropriate axis.
        if not sharex or row == n_rows - 1:
            ax.set_xlabel(xlabel)
        if not sharey or col == 0:
            ax.set_ylabel(ylabel)

    if figname:
        save_fig(fig, figname, tight_layout=True)

    return fig


def stacked_bar_plots(df, ax_name, group_names, value_colors, legend_upper_left=None, legend_title=None,
                      legend_cols=1, vert=False, plot_title=None, figname=None, figax=None, **kwargs):
    """Make stacked bar plots, one bar per row of the provided DataFame, and optionally show a legend.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot, rows are bar groups, columns are different values/colors
    ax_name : str
        Name of the axis for the plot
    group_names : list[str]
        Names of each group to display as ticks
    value_colors : list-like, length = len(df.columns)
        Color for each value of the df
    legend_upper_left : tuple(float, float)
        If specified, make a legend, with the upper left corner of the bounding box at these axes coordinates.
    legend_title : str
        If specified, title for the legend.
    legend_cols : int
        If specified, number of columns for the legend. Default is 1.
    vert : bool
        If False (default), make a horizontal bar plot. If True, make a vertical bar plot.
    plot_title : str
        If specified, title for the plot.
    figname : str
        If specified, save the figure to this filename.
    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    kwargs : for save_fig

    Returns
    -------
    fig : Figure handle
    """
    tick_values = np.arange(len(group_names))
    margin_edge = np.zeros(len(tick_values))
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    for (label, values), color in zip(df.items(), value_colors):
        if vert:
            ax.bar(tick_values, values, color=color, label=label, bottom=margin_edge, tick_label=group_names)
        else:
            ax.barh(tick_values, values, color=color, label=label, left=margin_edge, tick_label=group_names)

        # Advance the margin
        margin_edge += values

    # Set the max of axis
    if vert:
        ax.set_ylim(top=margin_edge.max())
    else:
        ax.set_xlim(right=margin_edge.max())

    # Add axis label
    if vert:
        ax.set_ylabel(ax_name)
    else:
        ax.set_xlabel(ax_name)

    # Add legend if specified
    if legend_upper_left:
        legend_args = {"ncol": legend_cols, "bbox_to_anchor": legend_upper_left, "loc": "upper left"}
        if legend_title:
            legend_args["title"] = legend_title
        ax.legend(**legend_args)

    if plot_title:
        ax.set_title(plot_title)

    if figname:
        save_fig(fig, figname, **kwargs)

    return fig


def annotate_heatmap(ax, df, thresh, adjust_lower_triangle=False):
    """Display numbers on top of a heatmap to make it easier to view for a reader. If adjust_lower_triangle is True,
    then the lower triangle of the heatmap will display values in parentheses. This should only happen if the heatmap
    is symmetric. Assumes that low values are displayed as a light color and high values are a dark color.

    Parameters
    ----------
    ax : Axes object
        The plot containing the heatmap on which annotations should be made
    df : pd.DataFrame
        The data underlying the heatmap.
    thresh : float
        Cutoff for switching from dark to light colors. Values above the threshold will be displayed as white text,
        those below as black text.
    adjust_lower_triangle : bool
        If True, the lower triangle values will be shown in parentheses.

    Returns
    -------
    None
    """
    # Determine whether the values are int or float
    # All columns of the df will be the same dtype
    if df.dtypes.iloc[0] == int:
        fmt = "d"
    else:
        fmt = "f"

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            value = df.iloc[row, col]
            if value > thresh:
                color = "white"
            else:
                color = "black"

            # Format the value as text
            if fmt == "d":
                value = f"{value:d}"
            elif fmt == "f":
                value = f"{value:.2f}"
            else:
                print("Error in format string, stopping and returning.")
                return
            # Add parentheses if desired and in the lower triangle and the heatmap is square
            if adjust_lower_triangle and row < col and df.shape[0] == df.shape[1]:
                value = "(" + value + ")"

            ax.text(row, col, value, ha="center", va="center", color=color)


# Legacy code FIXME
def roc_pr_curves(xaxis, tpr_list, precision_list, model_names, model_colors=None, prc_chance=None,
                  prc_upper_ylim=None, figname=None, legend=True, figax=None, **kwargs):
    """Make a ROC and PR curve for each model, optionally with a SD. Compute an AUC score for each curve.

    Parameters
    ----------
    xaxis : list-like
        The FPR and Recall, i.e. the x-axis for both plots. All TPR and Precision lists should be
        interpolated/computed to reflect the values at each point on xaxis.
    tpr_list : list of lists, shape = [n_models, len(xaxis)]
        tpr_list[i] corresponds to the TPR values for model i along xaxis. If tpr_list[i] is a list, then do not plot a
        standard deviation of the TPR. If tpr_list[i] is a list of lists, then it represents the TPR of each fold
        from cross-validation, in which case it is used to compute the mean and std of the TPR.
    precision_list : list of lists, shape = [n_models, len(xaxis)]
        precision_list[i] corresponds to the precision values for model i along xaxis. If precision_list[i] is a list,
        then do not plot a standard deviation of the precision. If precision_list[i] is a list of lists,
        then it represents the precision of each fold from cross-validation, in which case it is used to compute the
        mean and std of the precision.
    model_names : list-like
        The name of each model.
    model_colors : list-like or None
        If not none, the color to use for each model.
    prc_chance : float or None
        If not none, plot a chance line for the PR curve at this value.
    prc_upper_ylim : float or None
        If specified, the upper ylim for the PR curve. Otherwise, use the uper ylim of the ROC curve.
    figname : str or None
        If specified, save the figure with prefix figname.
    legend : bool
        If specified, display a legend.
    figax : ([figure, figure], [axes, axes]) or None
        If specified, make the plot in the two provided axes. Otherwise, generate a new axes.
    kwargs : dict
        Additional parameters for saving a figure.

    Returns
    -------
    fig_list : The handle to both figures (one for the ROC and one for the PR).
    auroc_list : AUROC scores for each model
    auroc_std_list : 1SD of AUROC scores for each model, or None if not computed.
    aupr_list : AUPR scores for each model
    aupr_std_list : 1SD of AUPR scores for each model, or None if not computed.

    """
    if figax:
        fig_list, ax_list = figax
    else:
        fig_roc, ax_roc = plt.subplots()
        fig_pr, ax_pr = plt.subplots()
        fig_list = [fig_roc, fig_pr]
        ax_list = [ax_roc, ax_pr]

    # If no colors specified, evenly sample the colormap to color each model
    if model_colors is None:
        model_colors = np.linspace(0, 0.99, len(model_names))
        model_colors = set_color(model_colors)

    # ROC curves
    ax = ax_list[0]
    auroc_list, auroc_std_list = _plot_each_model(ax, xaxis, tpr_list, model_colors, model_names)

    # Chance line
    ax.plot(xaxis, xaxis, color="black", linestyle="--", zorder=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect("equal")
    if legend:
        ax.legend(loc="lower right", frameon=False)

    # ylim of ROC curve will help format PR curve
    lower_ylim, upper_ylim = ax.get_ylim()

    # PR curves
    ax = ax_list[1]
    aupr_list, aupr_std_list = _plot_each_model(ax, xaxis, precision_list, model_colors, model_names)

    # Optional chance line and formatting
    if prc_chance:
        ax.axhline(prc_chance, color="black", linestyle="--", zorder=1)
    if not prc_upper_ylim:
        prc_upper_ylim = upper_ylim
    ax.set_ylim(bottom=lower_ylim, top=prc_upper_ylim)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_aspect("equal")
    if legend:
        ax.legend(frameon=False)

    if figname:
        save_fig(fig_list[0], figname + "Roc", **kwargs)
        save_fig(fig_list[1], figname + "Pr", **kwargs)

    return fig_list, auroc_list, auroc_std_list, aupr_list, aupr_std_list


def _plot_each_model(ax, xaxis, y_list, model_colors, model_names):
    """Helper function for roc_pr_curves to plot each model on an Axes object.

    """
    area_list = []
    area_std_list = []
    for y, color, name in zip(y_list, model_colors, model_names):
        y = np.array(y)
        area_std = None

        # If y is a list of lists (i.e. a matrix), then compute the std of the curve and AUC
        if len(y.shape) == 2:
            y_std = np.std(y, axis=0)
            # Compute std of AUC and format as a string
            area_std = np.std([auc(xaxis, i) for i in y])

            # Now compute the mean curve
            y = y.mean(axis=0)

            # The std can't go above 1 or below 0
            y_std_upper = np.min([y + y_std, np.ones(y.size)], axis=0)
            y_std_lower = np.max([y - y_std, np.zeros(y.size)], axis=0)

            # Plot the std of the curve
            ax.fill_between(xaxis, y_std_lower, y_std_upper, alpha=0.2, zorder=2, color=color)

        # Plot the curve and compute AUC
        area = auc(xaxis, y)
        ax.plot(xaxis, y, label=name, zorder=3, color=color)
        area_list.append(area)
        area_std_list.append(area_std)

    return area_list, area_std_list
