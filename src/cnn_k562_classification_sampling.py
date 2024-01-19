#!/usr/bin/env python3
"""
Fit a CNN classifier on a subsample of the K562 data and then perform either active learning or random sampling.
"""
import os
import argparse
import logging

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import selene_sdk
import selene_sdk.samplers

from mpra_tools import log, loaders, modeling, plot_utils
from selene_files import enhancer_model, enhancer_resnet_regression, metrics
plot_utils.set_manuscript_params()

PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "../")

sequence_key = "sequence"
activity_key = "activity_bin"


def load_data(activity_file, sequence_file, cv_cutoff=0.75, quantile_cutoffs=[0.5, 0.8]):
    """Load in the K562 data, remove noisy measurements, pull out putative enhancers in the F orientation, and define 2 bins."""
    activity_df = pd.read_excel(activity_file, sheet_name="K562_summary_data", index_col="name").squeeze()
    replicates = activity_df.loc[:, activity_df.columns.str.contains("replicate")]
    replicates = 2 ** replicates
    activity_sumstats_df = pd.DataFrame({
        "mean": replicates.mean(axis=1),
        "std": replicates.std(axis=1, ddof=1) # We want the sample std to calculate the CV
    })
    # Drop anything only obs in one replicate
    one_rep_mask = activity_sumstats_df["std"].isna()
    activity_sumstats_df = activity_sumstats_df[~one_rep_mask]
    activity_sumstats_df["cv"] = activity_sumstats_df["std"] / activity_sumstats_df["mean"]
    # Drop anything with high CV
    high_cv_mask = activity_sumstats_df["cv"] > cv_cutoff
    activity_sumstats_df = activity_sumstats_df[~high_cv_mask]
    activity = activity_df.loc[activity_sumstats_df.index, "mean"]

    sequences = pd.read_excel(sequence_file, sheet_name="K562 large-scale", skiprows=1, 
                              names=["name", "category", "chr", "start", "stop", "strand", "sequence"],
                              index_col="name")
    sequences = sequences.loc[activity.index]
    data = sequences.join(activity.rename("activity"))

    # Copy the df before modifying since this is a subset
    enhancers = data.groupby("category").get_group("putative enhancer").copy()
    enhancers = enhancers[enhancers["strand"] == "+"]
    enhancers[activity_key] = pd.qcut(enhancers["activity"],
                                        [0] + quantile_cutoffs + [1],
                                        labels = np.arange(len(quantile_cutoffs) + 1))
    
    # Get rid of the middle category
    enhancers[activity_key] = enhancers[activity_key].replace(
        enhancers[activity_key].cat.categories, [0, np.nan, 1]
    )
    enhancers = enhancers[enhancers[activity_key].notna()]
    return enhancers


def define_fold(chr, fold):
    chrom_to_fold = {
        "chr1": 1,
        "chr2": 2,
        "chr14": 2,
        "chr4": 3,
        "chr7": 3,
        "chr3": 4,
        "chr15": 4,
        "chr5": 5,
        "chr19": 5,
        "chr21": 5,
        "chr6": 6,
        "chrX": 6,
        "chrY": 6,
        "chr8": 7,
        "chr9": 7,
        "chr18": 7,
        "chr10": 8,
        "chr11": 8,
        "chr12": 9,
        "chr13": 9,
        "chr16": 9,
        "chr17": 10,
        "chr20": 10,
        "chr22": 10,
    }
    chrom_int_fold = chrom_to_fold[chr]
    if fold == chrom_int_fold:
        result = "test"
    elif (fold + 1 == chrom_int_fold) or (fold == 10 and chrom_int_fold == 1):
        result = "validation"
    else:
        result = "train"
    return result


def split_chromosomes(enhancers, fold, train_portion=0.2):
    """Split the chromosomes into train/validate/test, and randomly split the training into labeled and unlabeled"""
    enhancers["fold"] = enhancers["chr"].apply(define_fold, args=(fold,))
    train = enhancers[enhancers["fold"] == "train"]
    labeled, unlabeled = train_test_split(
        train.index,
        stratify=train[activity_key],
        random_state=0,
        train_size=train_portion,
    )
    enhancers.loc[unlabeled, "fold"] = "unlabeled"
    return enhancers


def close_extra_logs():
    for logname in ["selene", "selene_sdk.train_model.train", "selene_sdk.train_model.validation"]:
        sublog = logging.getLogger(logname)
        for handle in sublog.handlers:
            if type(handle) is logging.FileHandler:
                sublog.removeHandler(handle)


def train_and_validate(train_mat, val_mat, output_dir, batch_size, seq_size, epochs, seed, lr):
    """Perform a round of learning and return the trained model in evaluation mode."""
    modeling.seed_selene(seed)
    model = enhancer_resnet_regression.EnhancerResnet(seq_size)
    model.output = nn.Sequential(model.output, nn.Sigmoid())
    criterion = nn.BCELoss()
    sampler_label = [activity_key]
    training_metrics = dict(
        auroc=roc_auc_score,
        aupr=average_precision_score,
    )
    stopping_metric = "aupr"
    optim_class = torch.optim.Adam
    optim_kwargs = {"lr": lr, "weight_decay": 1e-6}

    train_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        train_mat,
        sequence_key,
        targets_key=activity_key,
        random_seed=seed,
        shuffle=True, # Need to mix newly sampled data in.
    )
    validate_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        val_mat,
        sequence_key,
        targets_key=activity_key,
        shuffle=False,
    )
    sampler = selene_sdk.samplers.MultiSampler(
        train_sampler,
        validate_sampler,
        sampler_label
    )

    # Figure out number of steps per epoch
    train_size = len(scipy.io.loadmat(train_mat)[sequence_key])
    steps_per_epoch = np.ceil(train_size / batch_size).astype(int)
    max_steps = steps_per_epoch * epochs

    trainer = selene_sdk.TrainModel(
        model,
        sampler,
        criterion,
        optim_class,
        optim_kwargs,
        batch_size=batch_size,
        report_stats_every_n_steps=steps_per_epoch,
        max_steps=max_steps,
        output_dir=output_dir,
        use_cuda=True,
        data_parallel=False,
        logging_verbosity=1,
        metrics=training_metrics,
        stopping_criteria=[stopping_metric, 15],
    )
    trainer.train_and_validate()
    close_extra_logs()

    # Plot loss history
    train_loss = loaders.load_data(
        os.path.join(output_dir, "selene_sdk.train_model.train.txt"),
        index_col=None
    ).squeeze()
    valid_metrics = loaders.load_data(
        os.path.join(output_dir, "selene_sdk.train_model.validation.txt"),
        index_col=None,
    )
    fig, ax = plt.subplots(figsize=plot_utils.get_figsize(0.4, 1))
    ax.plot(train_loss, color="blue", label="Training")
    ax.plot(valid_metrics["loss"], color="orange", label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plot_utils.save_fig(fig, os.path.join(output_dir, "lossHistory"))
    plt.close()

    valid_metrics = valid_metrics.drop(columns="loss")
    fig, ax_list = plot_utils.setup_multiplot(
        valid_metrics.shape[1],
        n_cols=int(valid_metrics.shape[1] / 2),
        sharex=False, sharey=False, big_dimensions=True
    )
    ax_list = np.ravel(ax_list)
    for ax, metric in zip(ax_list, valid_metrics):
        ax.plot(valid_metrics[metric], color="orange")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        
    plot_utils.save_fig(fig, os.path.join(output_dir, "validationHistory"))
    plt.close()

    # Load in the best model state and return it in evaluation mode
    model = loaders.load_cnn(os.path.join(output_dir, "best_model.pth.tar"), model=trainer.model)
    model.eval()
    return model


def cnn_predict(model, matfile, batch_size, output_dir, seq_size):
    """Load up a dataset, make predictions on it in batches, and save the predictions to file with the same path and basename as the matfile. Return the log probs and the true labels (always available here)."""
    assert matfile[-4:] == ".mat"
    basename = os.path.basename(matfile[:-4])
    sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
        matfile, sequence_key, targets_key=activity_key, shuffle=False,
    )
    data_batched, truth = sampler.get_data_and_targets(batch_size)
    log_probs = modeling.cnn_predict_batches(model, data_batched, use_cuda=True, seq_len=seq_size)
    np.save(os.path.join(output_dir, f"{basename}_log_probs.npy"), log_probs)
    return log_probs, truth.flatten()


def eval_on_test_set(model, matfile, logger, output_dir, batch_size, seq_size):
    """Make predictions on the provided test set and then calculate+return various performance metrics."""
    logger.info("Making predictions on the test set.")
    probs, truth = cnn_predict(model, matfile, batch_size, output_dir, seq_size)
    metrics = {}
    for method in [roc_auc_score, average_precision_score]:
        score = method(truth, probs)
        logger.info(f"{method.__name__} = {score}")
        metrics[method.__name__] = score
    score = f1_score(truth, probs > 0.5, average="weighted")
    logger.info(f"Weighted F1 = {score}")
    metrics["weighted_f1"] = score

    return pd.Series(metrics)


def train_and_eval(train_mat, validation_mat, test_set_mat, iteration_output_dir, logger, batch_size, seq_size, *const_modeling_params):
    """Train and validate a model, then evaluate on the test set. Return the trained model and performance metrics."""
    model = train_and_validate(
        train_mat,
        validation_mat,
        iteration_output_dir,
        batch_size,
        seq_size,
        *const_modeling_params
    )
    # I think I can do this but if not I need to get out the batch size and the sequence size from the const modeling parameters
    performance = eval_on_test_set(
        model, test_set_mat, logger, iteration_output_dir, batch_size, seq_size)
    return model, performance



def main(data_dir, output_dir, epochs, seed, lr, seq_size, batch_size, sampling_size, fold, activity_file, sequences_file, upper_bound, initial_data):
    logger = log.get_logger()
    enhancers = load_data(os.path.join(data_dir, activity_file),
                          os.path.join(data_dir, sequences_file))
    enhancers = split_chromosomes(enhancers, fold, train_portion=initial_data)
    logger.info(f"Fold {fold}, starting with {initial_data} of the initial data, number of sequences in each split:\n{enhancers['fold'].value_counts()}")

    output_dir = os.path.join(output_dir, str(initial_data))
    os.makedirs(output_dir, exist_ok=True)
    train_mat = os.path.join(output_dir, "train.mat")
    unlabeled_mat = os.path.join(output_dir, "unlabeled.mat")
    validation_mat = os.path.join(output_dir, "validation.mat")
    test_set_mat = os.path.join(output_dir, "test.mat")

    fold_to_mat = {
        "train": train_mat,
        "unlabeled": unlabeled_mat,
        "validation": validation_mat,
        "test": test_set_mat,
    }
    for fold, df in enhancers.groupby("fold"):
        modeling.prepare_data_for_selene(
            df["sequence"],
            df[activity_key].astype(int),
            activity_key,
            fold_to_mat[fold]
        )

    logger.info("Training initial model.")
    test_set_performance = {}
    iteration_output_dir = os.path.join(output_dir, "initial")

    const_modeling_params = [epochs, seed, lr]
    model, test_set_performance["initial"] = train_and_eval(
        train_mat, validation_mat, test_set_mat, iteration_output_dir, logger, batch_size, seq_size, *const_modeling_params)

    max_sample = max(sampling_size)
    train_data = scipy.io.loadmat(train_mat)

    for sampling in ["entropy", "random"]:
        logger.info(f"Performing {sampling} sampling for {max_sample} items.")
        unlabeled_data = scipy.io.loadmat(unlabeled_mat)
        unlabeled_idx = np.arange(len(unlabeled_data[sequence_key]))
        if sampling == "entropy":
            probs, truth = cnn_predict(model, unlabeled_mat, batch_size, iteration_output_dir, seq_size)
            probs = np.append(probs, 1 - probs, axis=1)
            entropies = scipy.stats.entropy(probs, axis=1, base=2)
            # Argsort is ascending
            ranks = np.argsort(entropies)
            full_sampled_idx = ranks[-max_sample:]
            logger.info(f"Highest entropies:\n{entropies[full_sampled_idx]}")

        elif sampling == "random":
            full_sampled_idx = np.random.choice(
                unlabeled_idx, size=max_sample, replace=False
            )

        else:
            # This should never happen
            raise ValueError(f"Sampling method should either be entropy or random, saw {sampling}.")
        
        for sample in sampling_size:
            sampled_idx = full_sampled_idx[-sample:]
            logger.info(f"Training a new model with {sample} sequences from the full sample of {max_sample}. If entropy sampling, these are the most uncertain. If random sampling, it is a subsample of the larger random sample.")
            train_seqs = np.append(
                train_data[sequence_key],
                unlabeled_data[sequence_key][sampled_idx],
                axis=0
            )
            train_labels = np.append(
                train_data[activity_key],
                unlabeled_data[activity_key][sampled_idx],
                axis=0
            )
            logger.info(f"Training data is now {len(train_seqs)} sequences.")
        
            round_str = f"{sampling}_{sample}"
            iteration_output_dir = os.path.join(output_dir, round_str)
            os.makedirs(iteration_output_dir, exist_ok=True)
            train_mat = os.path.join(iteration_output_dir, "train.mat")
        
            scipy.io.savemat(
                train_mat,
                {sequence_key: train_seqs, activity_key: train_labels}
            )

            logger.info("Training model.")
            _, test_set_performance[round_str] = train_and_eval(
            train_mat, validation_mat, test_set_mat, iteration_output_dir, logger, batch_size, seq_size, *const_modeling_params)

    if upper_bound:
        iteration_output_dir = os.path.join(output_dir, "full")
        os.makedirs(iteration_output_dir, exist_ok=True)
        train_mat = os.path.join(iteration_output_dir, "train.mat")

        train_data = scipy.io.loadmat(os.path.join(output_dir, "train.mat"))
        unlabeled_data = scipy.io.loadmat(os.path.join(output_dir, "unlabeled.mat"))
        train_seqs = np.append(
            train_data[sequence_key],
            unlabeled_data[sequence_key],
            axis=0
        )
        train_labels = np.append(
            train_data[activity_key],
            unlabeled_data[activity_key],
            axis=0
        )
        scipy.io.savemat(
            train_mat,
            {sequence_key: train_seqs, activity_key: train_labels}
        )
        logger.info("Training model on all training data.")
        _, test_set_performance["full"] = train_and_eval(
        train_mat, validation_mat, test_set_mat, iteration_output_dir, logger, batch_size, seq_size, *const_modeling_params)
    
    test_set_performance = pd.DataFrame.from_dict(test_set_performance, orient="index")
    loaders.write_data(test_set_performance, os.path.join(output_dir, "performance.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir")
    parser.add_argument("--data_dir", help="Directory where all input files live.", default=os.path.join(PATH, "Data", "Downloaded", "K562"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seq_size", type=int, default=230)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training, NOT the sampling batch size.")
    parser.add_argument("--sampling_size", type=int, nargs="+", default=[5000], help="Numbers of sequences to sample for one round of learning.")
    parser.add_argument("--fold", type=int, default=1, help="Which fold of cross-validation.")
    parser.add_argument("--activity_file", type=str, default="supTable4.xlsx")
    parser.add_argument("--sequences_file", type=str, default="supTable3.xlsx")
    parser.add_argument("--upper_bound", action="store_true", default=False, help="If specified, establish an upper bound by training a model on ALL training data. Otherwise this is not done.")
    parser.add_argument("--initial_data", default=0.2, type=float, help="The amount of the training data that should be used to train the initial model. The remaining training data is 'unlabeled' and used for sampling. Can either be a fraction or an integer value.")
    args = parser.parse_args()

    if args.initial_data <= 0:
        raise ValueError("Initial data must be a positive numeric value.")
    if args.initial_data >= 1:
        if int(args.initial_data) == args.initial_data:
            args.initial_data = int(args.initial_data)
        else:
            raise ValueError("Initial data must be an integer if 1 or larger. Fractional values can only be between 0 and 1.")

    main(**vars(args))
