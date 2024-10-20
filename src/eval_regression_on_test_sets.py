#!/usr/bin/env python3
"""
Take the provided regression CNN and make predictions on each test set. All data, including the checkpoint and .mat
files for the test set, should all be located within the directory provided as input.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import selene_sdk
import selene_sdk.samplers

from mpra_tools import loaders, modeling, plot_utils

figures_dir = "Figures"

# X aand Y limits of scatterplot
limits = {
    'x' : (-8,7),
    'y' : (-7,7)
}
ticks = {
    'x' : [-5, 0, 5],
    'y' : np.arange(-6, 7, 2)
}


###################
### MAIN SCRIPT ###
###################
# Lazy argument parsing
if len(sys.argv) != 2:
    print(f"Usage: python3 {__file__} base_dir/")
    print(__doc__)
    quit(1)

plot_utils.set_manuscript_params()
base_dir = sys.argv[1]
batch_size = 64
use_cuda = torch.cuda.is_available()
model = loaders.load_cnn(os.path.join(base_dir,"best_model", "best_model.pth.tar"))
if use_cuda:
    model.cuda()

model.eval()
metrics = {}
test_set_names = ["validate_genomic", "test_set", "test_retinopathy", "high_confidence"]
test_set_titles = ["Genomic Validation", "Mutagenic Test Set", "Test Set", "High Confidence Predictions"]
for test_set, title in zip(test_set_names, test_set_titles):
    sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        os.path.join(base_dir,"best_model", test_set + ".mat"),
        "sequence",
        targets_key="expression_log2",
        shuffle=False
    )
    data, truth = sampler.get_data_and_targets(batch_size)
    truth = truth.flatten()
    preds = modeling.cnn_predict_batches(model, data, use_cuda=use_cuda)
    preds = preds.flatten()
    # Get the seq IDs too so we can write this to file all in one place
    seq_ids = loaders.load_data(os.path.join(base_dir,"best_model", test_set + "_ids.txt"), index_col=None)
    loaders.write_data(
        pd.DataFrame({
            "predicted": preds,
            "observed": truth
        }, index=seq_ids),
        os.path.join(base_dir, test_set + "_preds.txt"),
        index_label="label"
    )
    # Plot pred vs obs for visualization purposes
    fig, ax = plt.subplots(figsize=plot_utils.get_figsize(0.33, 1))
    fig, ax, corrs = plot_utils.scatter_with_corr(
        truth,
        preds,
        "log2 Observed Activity",
        "log2 Predicted Activity",
        colors="density",
        loc="upper left",
        figax=(fig, ax),
        rasterize=True,
    )
    metrics[test_set] = corrs
    # Set axis 
    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    ax.set_xticks(ticks['x'])
    ax.set_yticks(ticks['y'])
    # Show y = x
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = max(xlim[0], ylim[0])
    vmax = min(xlim[1], ylim[1])
    ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", lw=1)
    ax.set_title(title)


    plot_utils.save_fig(fig, os.path.join(figures_dir, f"{test_set}_PredVsObs"))
    plt.close()
    
loaders.write_data(
    pd.DataFrame(metrics, index=["pearson", "spearman"]).T,
    os.path.join(base_dir, 'best_model', "test_set_metrics.txt")
)
