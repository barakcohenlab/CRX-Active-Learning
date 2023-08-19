#!/usr/bin/env python3
"""
Take the provided regression CNN and make predictions on each test set. All data, including the checkpoint and .mat
files for the test set, should all be located within the directory provided as input.
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import torch
import selene_sdk
import selene_sdk.samplers

from mpra_tools import loaders, modeling, plot_utils


###################
### MAIN SCRIPT ###
###################
# Lazy argument parsing
if len(sys.argv) != 2:
    print(f"Usage: python3 {__file__} data_dir/")
    print(__doc__)
    quit(1)

plot_utils.set_manuscript_params()
data_dir = sys.argv[1]
batch_size = 64
use_cuda = torch.cuda.is_available()
model = loaders.load_cnn(os.path.join(data_dir, "best_model.pth.tar"))
if use_cuda:
    model.cuda()

model.eval()
metrics = {}
test_set_names = ["validate_genomic", "test_set", "test_retinopathy"]
for test_set in test_set_names:
    sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        os.path.join(data_dir, test_set + ".mat"),
        "sequence",
        targets_key="expression_log2",
        shuffle=False
    )
    data, truth = sampler.get_data_and_targets(batch_size)
    truth = truth.flatten()
    preds = modeling.cnn_predict_batches(model, data, use_cuda=use_cuda)
    preds = preds.flatten()
    # Get the seq IDs too so we can write this to file all in one place
    seq_ids = loaders.load_data(os.path.join(data_dir, test_set + "_ids.txt"), index_col=None)
    loaders.write_data(
        pd.DataFrame({
            "predicted": preds,
            "observed": truth
        }, index=seq_ids),
        os.path.join(data_dir, test_set + "_preds.txt"),
        index_label="label"
    )
    # Plot pred vs obs for visualization purposes
    fig, ax = plt.subplots(figsize=plot_utils.get_figsize(0.33, 1))
    fig, ax, corrs = plot_utils.scatter_with_corr(
        truth,
        preds,
        "Observed",
        "Predicted",
        colors="density",
        loc="upper left",
        figax=(fig, ax),
        rasterize=True,
    )
    metrics[test_set] = corrs
    # Show y = x
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = max(xlim[0], ylim[0])
    vmax = min(xlim[1], ylim[1])
    ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", lw=1)
    plot_utils.save_fig(fig, os.path.join(data_dir, f"{test_set}_PredVsObs"))
    plt.close()
    
loaders.write_data(
    pd.DataFrame(metrics, index=["pearson", "spearman"]).T,
    os.path.join(data_dir, "test_set_metrics.txt")
)
