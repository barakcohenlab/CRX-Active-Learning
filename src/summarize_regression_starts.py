#!/usr/bin/env python3
"""
Summarize the performance of the regression model across multiple random starts.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpra_tools import loaders, log, plot_utils
plot_utils.set_manuscript_params()
mpl.rcParams["lines.markersize"] = mpl.rcParams["lines.markersize"] * 3

# Lazy argument parsing
if len(sys.argv) != 2:
    print(f"Usage: python3 {__file__} base_dir/")
    print(__doc__)
    print("base_dir is the directory where things should be written and where each random start subdirectory is "
          "located.")
    quit(1)

base_dir = sys.argv[1]
nstarts = 20
logger = log.get_logger()

# cd into the appropriate directory and get all subdirectories, which correspond to the random starts
os.chdir(base_dir)
# (best epoch number, train loss, val loss, val PCC, val SCC)
performance = []
subdirs = [str(i) for i in np.arange(1, nstarts + 1)]
for dirname in subdirs:
    train_loss = loaders.load_data(os.path.join(dirname, "selene_sdk.train_model.train.txt"), index_col=None).squeeze()
    valid_metrics = loaders.load_data(os.path.join(dirname, "selene_sdk.train_model.validation.txt"), index_col=None)
    # Determine the epoch with the lowest validation loss
    best_epoch = valid_metrics["loss"].idxmin()
    best_train_loss = train_loss[best_epoch]
    valid_loss, valid_pcc, valid_scc = valid_metrics.loc[best_epoch]
    # Add one since epoch is zero-based indexing
    best_epoch += 1
    logger.info(f"Random start {dirname} did best at epoch {best_epoch}. Training loss was {best_train_loss:.2f}, "
                f"validation loss was {valid_loss:.2f}. At this epoch, validation PCC was {valid_pcc:.3f} and SCC was "
                f"{valid_scc:.3f}.")
    subdir_metrics = dict(
        best_epoch=best_epoch,
        train_loss=best_train_loss,
        valid_loss=valid_loss,
        pcc=valid_pcc,
        scc=valid_scc,
    )
    performance.append(pd.Series(subdir_metrics, name=dirname))

performance = pd.DataFrame(performance)
loaders.write_data(performance, "random_start_performance.txt", index_label="seed")
summary = performance.agg(["mean", "std", "min", "max"])
logger.info(f"Average performance of this architecture:\n{summary}")

# Get the best model and make a copy in the main directory
best_model = performance["pcc"].idxmax()
logger.info(f"Best PCC was with model {best_model}")
os.symlink(best_model, "best_model")

# Plot training loss vs validation loss vs number of epochs
fig, ax = plt.subplots()
artists = ax.scatter(performance["train_loss"], performance["valid_loss"], c=performance["best_epoch"], vmin=0)
ax.set_xlabel("Training loss")
ax.set_ylabel("Validation loss")
cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
fig.colorbar(artists, cax=cax, label="Number of epochs")
plot_utils.save_fig(fig, "trainVsValidationLoss")
plt.close(fig)

# Plot training loss vs validation PCC
fig, ax = plt.subplots()
ax.scatter(performance["train_loss"], performance["pcc"], c="k")
ax.set_xlabel("Training loss")
ax.set_ylabel("Validation PCC")
plot_utils.save_fig(fig, "pccVsLoss")
plt.close(fig)

# Now read in the train/validation info for each model and join it together
metrics = {}
for dirname in subdirs:
    metrics[dirname] = loaders.load_data(os.path.join(
        dirname,
        "best_model_performance.txt"
    ))

metrics = pd.concat(
    metrics,
    names=["seed", "category"]
).rename_axis(columns="metric")

# Move the category from the index to the columns
metrics = metrics.reset_index(
    "category"
).pivot(
    columns="category"
).reorder_levels(
    ["category", "metric"],
    axis=1
)

loaders.write_data(metrics, "random_start_train_validate.txt")
