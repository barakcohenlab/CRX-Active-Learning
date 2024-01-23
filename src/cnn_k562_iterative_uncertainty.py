#!/usr/bin/env python3
"""
Fit a CNN classifier on a subsample of the K562 data and then perform several rounds of either active learning or random sampling.
"""
import os
import argparse
import logging
import math
import numpy as np
import scipy
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import selene_sdk
import selene_sdk.samplers

from mpra_tools import log, loaders, modeling, plot_utils
from selene_files import enhancer_resnet_regression
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
        result = "unlabeled"
    return result


def close_extra_logs():
    for logname in ["selene", "selene_sdk.train_model.train", "selene_sdk.train_model.validation"]:
        sublog = logging.getLogger(logname)
        for handle in sublog.handlers:
            if type(handle) is logging.FileHandler:
                sublog.removeHandler(handle)


def create_model(train_mat, val_mat, output_dir, batch_size, seq_size, epochs, seed, lr):
    # Initialize classifier model
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
    
    # Optimizer type and values
    optim_class = torch.optim.Adam
    optim_kwargs = {"lr": lr, "weight_decay": 1e-6}
    
    # Initialize samplers for training and validation data
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
    
    # Train Model
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
    
    # Load in the best model state and return it in evaluation mode
    model = loaders.load_cnn(os.path.join(output_dir, "best_model.pth.tar"), model=trainer.model)
    model.eval()
    return model
    
def getEntropy(model, df, k, seq_size):
    
    unlabeled_df = df[df['fold'] == 'unlabeled']
    if unlabeled_df.empty:
        return
    # Get the model predictions
    hot_seqs = modeling.one_hot_encode(unlabeled_df['sequence'])
    batch_seqs = modeling.batch(hot_seqs)
    preds = modeling.cnn_predict_batches(model, batch_seqs, use_cuda=True, seq_len=seq_size)
    preds = np.concatenate(preds)
    preds = np.stack([1-preds,preds], axis=1)
    
    
    #Calculate the entropy for each unlabeled datapoint
    entropies = scipy.stats.entropy(preds,axis=1,base=2)
    entropy = pd.Series(entropies, index = unlabeled_df.index)
    if len(entropy) < k:
        k = len(entropy)
    return entropy.sort_values(ascending=False).index[:k]
    

def main(data_dir, output_dir, epochs, seed, lr, seq_size, batch_size, sampling_size, initial_size, fold, activity_file, sequences_file, iterations):
    logger = log.get_logger()
    enhancers = load_data(os.path.join(data_dir, activity_file),
                          os.path.join(data_dir, sequences_file))
    enhancers["fold"] = enhancers["chr"].apply(define_fold, args=(fold,))
    
    # Check to see if there is a cutoff for iterations. If not step up to the full dataset
    if iterations == 0:
        num_training_points = enhancers["fold"].value_counts()["unlabeled"] - initial_size
        iterations = math.ceil(float(num_training_points) / float(sampling_size)) + 1
    logger.info(f"{iterations} Iterations")
    
    for mode in ['Entropy', 'Random']:
        enhancers["fold"] = enhancers["chr"].apply(define_fold, args=(fold,))
        # Initialize the first set of training examples randomly
        new_train_data = enhancers[enhancers['fold'] == 'unlabeled'].sample(n=initial_size, random_state=1).index
        for i in range(1, iterations + 1):
            iteration_output_dir = os.path.join(output_dir, mode, "round"+str(i))
            os.makedirs(iteration_output_dir, exist_ok=True)
            
            # Include 'sampling_size' training points
            enhancers.loc[new_train_data, ['fold']] = 'train'
            logger.info(f"Fold: {fold}, Iteration: {i}, number of sequences in each split:\n{enhancers['fold'].value_counts()}")
            
            # Take the dataframe and split it into train, val, test       
            train_mat = os.path.join(iteration_output_dir, "train.mat")
            unlabeled_mat = os.path.join(iteration_output_dir, "unlabeled.mat")
            validation_mat = os.path.join(iteration_output_dir, "validation.mat")
            test_set_mat = os.path.join(iteration_output_dir, "test.mat")

            #Prepare data for use by pytorch
            fold_to_mat = {
                "train": train_mat,
                "unlabeled": unlabeled_mat,
                "validation": validation_mat,
                "test": test_set_mat,
            }
            for f, df in enhancers.groupby("fold"):
                modeling.prepare_data_for_selene(
                    df["sequence"],
                    df[activity_key].astype(int),
                    activity_key,
                    fold_to_mat[f]
                )
            # Use the new datasets to train the next model
            logger.info(f"Training the {mode} {i} model")
            const_modeling_params = [epochs, seed, lr]
            model = create_model(train_mat, validation_mat, iteration_output_dir, batch_size, seq_size, *const_modeling_params)

            # Get the indeces of the training data to add next round
            if mode == 'Entropy':
                new_train_data = getEntropy(model, enhancers, sampling_size, seq_size)
            else:
                k = sampling_size
                unlabeled_df = enhancers[enhancers['fold'] == 'unlabeled']
                if len(unlabeled_df) < k:
                    k = len(unlabeled_df)
                new_train_data = unlabeled_df.sample(n=k, random_state=1).index
    
    #Create the full model using all data
    iteration_output_dir = os.path.join(output_dir, "Full")
    os.makedirs(iteration_output_dir, exist_ok=True)
    enhancers["fold"] = enhancers["chr"].apply(define_fold, args=(fold,))
    enhancers['fold'].replace('unlabeled','train',inplace=True)
    logger.info(f"Training the final model with all {enhancers['fold'].value_counts()['train']} points of training data")

    # Take the dataframe and split it into train, val, test       
    train_mat = os.path.join(iteration_output_dir, "train.mat")
    validation_mat = os.path.join(iteration_output_dir, "validation.mat")
    test_set_mat = os.path.join(iteration_output_dir, "test.mat")

    #Prepare data for use by pytorch
    fold_to_mat = {
        "train": train_mat,
        "validation": validation_mat,
        "test": test_set_mat,
    }
    for f, df in enhancers.groupby("fold"):
        modeling.prepare_data_for_selene(
            df["sequence"],
            df[activity_key].astype(int),
            activity_key,
            fold_to_mat[f]
        )
    # Use the new datasets to train the next model
    logger.info(f"Training the Full model")
    const_modeling_params = [epochs, seed, lr]
    model = create_model(train_mat, validation_mat, iteration_output_dir, batch_size, seq_size, *const_modeling_params)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir")
    parser.add_argument("--data_dir", help="Directory where all input files live.", default=os.path.join(PATH, "Data", "Downloaded", "K562"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seq_size", type=int, default=230)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training, NOT the sampling batch size.")
    parser.add_argument("--sampling_size", type=int, default=10000, help="Numbers of sequences to sample for one round of learning.")
    parser.add_argument("--initial_size", type=int, default=5000, help="Initial amount of data before iterating steps")
    parser.add_argument("--fold", type=int, default=1, help="Which fold of cross-validation.")
    parser.add_argument("--activity_file", type=str, default="supTable4.xlsx")
    parser.add_argument("--sequences_file", type=str, default="supTable3.xlsx")
    parser.add_argument("--iterations", default=0, type=int, help="Number of times to select new data")
    args = parser.parse_args()
    
    main(**vars(args))
