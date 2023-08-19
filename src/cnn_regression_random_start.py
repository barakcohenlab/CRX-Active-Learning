#!/usr/bin/env python3
"""
Fit a regression CNN using a specified random start. Then plot the training history and performance of the best model
on the validation and test sets.
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import selene_sdk
import selene_sdk.samplers

from mpra_tools import log, loaders, modeling, plot_utils
from selene_files import enhancer_resnet_regression, metrics

# Global parameters
SEQ_SIZE = 164
LR = 0.0003
BATCH_SIZE = 128
NEPOCHS = 50
DECAY_PATIENCE = 3
DECAY_FACTOR = 0.2
STOPPING_METRIC = "scc"
STOPPING_PATIENCE = 10
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "../")
DATA_DIR = os.path.join(PATH, "Data")

plot_utils.set_manuscript_params()


###################
### MAIN SCRIPT ###
###################
def main(seed, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    logger = log.get_logger()
    sequence_key = "sequence"
    activity_key = "expression_log2"
    logger.info("Reading in data.")
    activity_df = loaders.load_data(os.path.join(DATA_DIR, "activity_summary_stats_and_metadata.txt"))

    train_df, test_df, validate_df = loaders.train_test_split(activity_df, "cnn")
    batches_to_use = [
        "Genomic",
        "CrxMotifMutant",
        "Round2",
        "Round3a",
        "Round3b",
        "Round4b",
    ]
    train_df = train_df[train_df["data_batch_name"].isin(batches_to_use)]
    # Also pull out the genomic sequences in the validation set as another evaluation criteria
    validate_genomic_df = validate_df[validate_df["original_genomic"]]
    logger.info(f"Number of sequences available for training: {len(train_df)}")
    logger.info(f"Number of sequences in the validation set: {len(validate_df)}")
    logger.info(f"Number of sequences in the test set: {len(test_df)}")
    logger.info(f"Number of original genomic sequences in the validation set: {len(validate_genomic_df)}")

    logger.info("Reading in the retinopathy test set.")
    retinopathy_df = loaders.load_data(os.path.join(DATA_DIR, "retinopathy_reformatted.txt"))
    logger.info(f"Number of genomic sequences in the retinopathy test set: {len(retinopathy_df)}")

    # Prepare the datasets
    train_mat = os.path.join(output_dir, "train.mat")
    validate_mat = os.path.join(output_dir, "validate.mat")
    validate_genomic_mat = os.path.join(output_dir, "validate_genomic.mat")
    test_mat = os.path.join(output_dir, "test_set.mat")
    retinopathy_mat = os.path.join(output_dir, "test_retinopathy.mat")
    df_to_mat = [
        [train_df, train_mat],
        [validate_df, validate_mat],
        [validate_genomic_df, validate_genomic_mat],
        [test_df, test_mat],
        [retinopathy_df, retinopathy_mat],
    ]
    for df, mat in df_to_mat:
        modeling.prepare_data_for_selene(
            df[sequence_key],
            df[activity_key],
            activity_key,
            mat
        )
        # Also write the seq IDs to file
        basename = ".".join(mat.split(".")[:-1])
        loaders.write_data(df["label"], basename + "_ids.txt", index=False)

    # Metrics Selene will log during training
    training_metrics = dict(
        pcc=metrics.pearson,
        scc=metrics.spearman,
    )
    modeling.seed_selene(seed)

    # Initialize the model, optimizer, and loss function
    model = enhancer_resnet_regression.EnhancerResnet(SEQ_SIZE)
    criterion = enhancer_resnet_regression.criterion()
    optim_class, optim_kwargs = enhancer_resnet_regression.get_optimizer(LR)

    # Initialize the samplers
    train_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        train_mat,
        sequence_key,
        targets_key=activity_key,
        # No random seed since we are using the random seed passed on command line
        shuffle=True
    )
    validate_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        validate_mat,
        sequence_key,
        targets_key=activity_key,
        shuffle=False
    )
    sampler = selene_sdk.samplers.MultiSampler(
        train_sampler,
        validate_sampler,
        [activity_key]
    )

    # Figure out number of steps in an epoch
    steps_per_epoch = np.ceil(len(train_df) / BATCH_SIZE).astype(int)
    max_steps = steps_per_epoch * NEPOCHS

    # Create trainer
    trainer = selene_sdk.TrainModel(
        model,
        sampler,
        criterion,
        optim_class,
        optim_kwargs,
        batch_size=BATCH_SIZE,
        report_stats_every_n_steps=steps_per_epoch,
        max_steps=max_steps,
        output_dir=output_dir,
        use_cuda=True,
        data_parallel=False,
        logging_verbosity=0,
        metrics=training_metrics,
        scheduler_kwargs=dict(patience=DECAY_PATIENCE, factor=DECAY_FACTOR),
        stopping_criteria=[STOPPING_METRIC, STOPPING_PATIENCE]
    )
    # Fit model
    trainer.train_and_validate()

    # Plot training history
    train_loss = loaders.load_data(os.path.join(output_dir, "selene_sdk.train_model.train.txt"), index_col=None).squeeze()
    valid_metrics = loaders.load_data(os.path.join(output_dir, "selene_sdk.train_model.validation.txt"), index_col=None)
    # Plot loss history
    fig, ax = plt.subplots(figsize=plot_utils.get_figsize(0.4, 1))
    ax.plot(train_loss, color="blue", label="Training")
    ax.plot(valid_metrics["loss"], color="orange", label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plot_utils.save_fig(fig, os.path.join(output_dir, "lossHistory"))
    plt.close()

    # Plot performance history for each validation metric -- no longer need validation loss here
    valid_metrics = valid_metrics.drop(columns="loss")
    fig, ax_list = plot_utils.setup_multiplot(valid_metrics.shape[1], n_cols=int(valid_metrics.shape[1] / 2),
                                              sharex=False, sharey=False, big_dimensions=True)
    ax_list = np.ravel(ax_list)
    for ax, metric in zip(ax_list, valid_metrics):
        ax.plot(valid_metrics[metric], color="orange")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)

    plot_utils.save_fig(fig, os.path.join(output_dir, "validationHistory"))
    plt.close()

    # Make predictions on the train and validation set for display purposes
    trainer.model = loaders.load_cnn(os.path.join(output_dir, "best_model.pth.tar"), model=trainer.model)
    trainer.model.eval()
    
    best_model_metrics = {}
    for name in ["train", "validate"]:
        n_test_samples = None
        data, truth = sampler.get_data_and_targets(trainer.batch_size, n_test_samples, mode=name)
        truth = truth.flatten()
        preds = modeling.cnn_predict_batches(trainer.model, data, use_cuda=True)
        preds = preds.flatten()
        np.save(os.path.join(output_dir, f"{name}_labels.npy"), truth)
        np.save(os.path.join(output_dir, f"{name}_predictions.npy"), preds)

        fig, ax = plt.subplots(figsize=plot_utils.get_figsize(0.4, 1))
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
        # Show y = x
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        vmin = max(xlim[0], ylim[0])
        vmax = min(xlim[1], ylim[1])
        ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", lw=1)
        plot_utils.save_fig(fig, os.path.join(output_dir, f"{name}PredVsObs"))
        plt.close()
        
        # Keep track of the performance
        best_model_metrics[name] = corrs
        logger.info(best_model_metrics)
    
    best_model_metrics = pd.DataFrame(best_model_metrics, index=["pearson", "spearman"])
    loaders.write_data(best_model_metrics.T, os.path.join(output_dir, "best_model_performance.txt"))


if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("seed", type=int, help="Seed for random number generators.")
    parser.add_argument("output_dir", help="Directory where all output files will be written.")
    args = parser.parse_args()
    main(**vars(args))
