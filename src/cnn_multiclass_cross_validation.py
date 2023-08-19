#!/usr/bin/env python3
"""
Fit a CNN on various portions of the data using cross-validation. For each portion of the data, only
do cross-validation on the new part of the data. The held-out part from the CV is ignored; we used
the predetermined validation set to track performance. Then assess model performance on the test
set.
"""
import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import selene_sdk
import selene_sdk.samplers

from mpra_tools import log, loaders, modeling, plot_utils
from selene_files import enhancer_model, metrics

# Global parameters
CV_SEED = 1260
# We want to be sure to use the same seed for Selene as before, but also need to seed the RNG for doing the CV!
SELENE_SEED = 400
SEQ_SIZE = 164
NBINS = 4
NFOLDS = 10
LR = 0.0001
BATCH_SIZE = 64
NEPOCHS = 500
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "../")
DATA_DIR = os.path.join(PATH, "Data")
class_name_shorthand = ["Si", "In", "WE", "SE"]
plot_utils.set_manuscript_params()


###################
### MAIN SCRIPT ###
###################
def main(output_dir, nfolds, activity_file, modeling_splits, epochs, report_train=False):
    logger = log.get_logger()
    sequence_key = "sequence"
    activity_key = "activity_bin"
    logger.info("Reading in data.")
    activity_df = loaders.load_data(activity_file)
    activity_df[activity_key] = activity_df[activity_key].apply(loaders.activity_to_int)
    training_sets = loaders.load_data(modeling_splits)
    logger.info(f"Planned data splits:\n{training_sets}")

    train_df, test_df, validate_df = loaders.train_test_split(activity_df, "cnn")
    logger.info(f"Number of sequences available for training: {len(train_df)}")
    logger.info(f"Number of sequences in the validation set: {len(validate_df)}")
    logger.info(f"Number of sequences in the test set: {len(test_df)}")

    logger.info("Reading in the retinopathy test set.")
    retinopathy_df = loaders.load_data(
        os.path.join(DATA_DIR, "retinopathy_reformatted.txt"))
    retinopathy_df[activity_key] = retinopathy_df[activity_key].apply(loaders.activity_to_int)
    logger.info(f"Number of genomic sequences in the retinopathy test set: {len(retinopathy_df)}")

    # Prepare the validation and test sets
    validate_mat = os.path.join(output_dir, "validate.mat")
    test_mat = os.path.join(output_dir, "test_set.mat")
    retinopathy_mat = os.path.join(output_dir, "test_retinopathy.mat")
    eval_datasets = [
        [validate_df, validate_mat],
        [test_df, test_mat],
        [retinopathy_df, retinopathy_mat]
    ]
    for df, mat in eval_datasets:
        logger.info(f"Frequency of each activity bin in {mat}:\n"
                    f"{df[activity_key].value_counts(normalize=True).sort_index()}")
        modeling.prepare_data_for_selene(
            df[sequence_key],
            df[activity_key],
            activity_key,
            mat
        )
        # Write the names to file as well in case we need these later. Because the names are
        # variable-length strings, we cannot put them into a .mat file, so let's just put it into a
        # separate text file.
        basename = ".".join(mat.split(".")[:-1])
        loaders.write_data(df["label"], basename + "_ids.txt", index=False)

    # Metrics that Selene will log during training
    training_metrics = dict(
        micro_auroc=metrics.micro_auroc,
        macro_auroc=metrics.macro_auroc,
        micro_aupr=metrics.micro_aupr,
        macro_aupr=metrics.macro_aupr,
        micro_f1=metrics.micro_f1,
        macro_f1=metrics.macro_f1,
    )

    # Test set info to log:
    # 1. Training data name
    # 2. CV fold
    # 3. Size of the training data (previous data + the part CV-ed)
    # 4. The test set used
    # 5. Micro, macro, and weighted F1 scores, and per-class scores
    performance_metrics = []
    logger.info(f"Beginning model fitting. Each model will be fit for {epochs} epochs.")
    for dataset_name, grouping_info in training_sets.iterrows():
        logger.info(f"Performing cross-validation with the {dataset_name} dataset.")
        dataset_df, prev_idx, cv_df = loaders.prepare_cross_validation(train_df, grouping_info)
        logger.info(f"{len(dataset_df)} sequences available for training.")
        logger.info(f"Of which, {len(cv_df)} will be cross-validated.")

        if len(cv_df) == 0:
            logger.warning("No new data to cross-validate! Skipping this dataset.")
            continue

        logger.info(f"Composition of classes in the full dataset:\n"
                    f"{dataset_df[activity_key].value_counts(normalize=True).sort_index()}")
        logger.info(f"Composition of classes in the new data:\n"
                    f"{cv_df[activity_key].value_counts(normalize=True).sort_index()}")

        # Grab the class labels for the new data, since that is what will stratify the CV
        cv_labels = cv_df[activity_key]
        cv = modeling.subset_cross_validation(cv_labels, prev_idx, nfolds, seed=CV_SEED)
        for fold, (full_subset_idx, fake_val_idx) in enumerate(cv, 1):
            logger.info(f"Fold {fold}.")
            fold_dir = os.path.join(output_dir, dataset_name, str(fold))
            os.makedirs(fold_dir, exist_ok=True)
            dataset_size = len(full_subset_idx)
            fold_mat = os.path.join(fold_dir, "train.mat")
            modeling.prepare_data_for_selene(
                dataset_df.loc[full_subset_idx, sequence_key],
                dataset_df.loc[full_subset_idx, activity_key],
                activity_key,
                fold_mat
            )
            modeling.seed_selene(SELENE_SEED)

            # Initialize the model, optimizer, and loss function
            model = enhancer_model.EnhancerModel(SEQ_SIZE, 4)
            criterion = enhancer_model.criterion()
            optim_class, optim_kwargs = enhancer_model.get_optimizer(LR)

            # Initialize the samplers
            train_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
                fold_mat,
                sequence_key,
                targets_key=activity_key,
                random_seed=4515,
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
                ["0", "1", "2", "3"]
            )

            # Figure out number of steps in an epoch
            steps_per_epoch = np.ceil(dataset_size / BATCH_SIZE).astype(int)
            max_steps = steps_per_epoch * epochs

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
                output_dir=fold_dir,
                use_cuda=True,
                data_parallel=False,
                logging_verbosity=1,
                metrics=training_metrics,
            )
            trainer.train_and_validate()
            # Close logging Handlers to files specific to this fold
            for logname in ["selene", "selene_sdk.train_model.train", "selene_sdk.train_model.validation"]:
                sublog = logging.getLogger(logname)
                for handle in sublog.handlers:
                    if type(handle) is logging.FileHandler:
                        sublog.removeHandler(handle)

            # Load in the best model state, then evaluate on the test sets
            model = loaders.load_cnn(os.path.join(fold_dir, "best_model.pth.tar"), model=trainer.model)
            model.eval()

            pred_datasets = [test_mat, retinopathy_mat]
            if report_train:
                pred_datasets.append(fold_mat)
                pred_datasets.append(validate_mat)
                
            for test_set in pred_datasets:
                test_set_name = os.path.basename(test_set).split(".")[0]
                logger.info(f"Evaluating performance of {dataset_name} fold {fold} on {test_set_name}")
                test_set_sampler = selene_sdk.samplers.file_samplers.MatFileSampler(
                    test_set,
                    sequence_key,
                    targets_key=activity_key,
                    shuffle=False
                )
                test_set_batched, test_set_truth = test_set_sampler.get_data_and_targets(BATCH_SIZE)
                test_set_truth = test_set_truth.flatten()
                log_probs = modeling.cnn_predict_batches(model, test_set_batched, use_cuda=True)
                np.save(os.path.join(fold_dir, f"{test_set_name}_log_probs.npy"), log_probs)
                predictions = log_probs.argmax(axis=1)
                fold_data = dict(
                    dataset=dataset_name,
                    fold=fold,
                    nseqs_train=len(full_subset_idx),
                    test_set=os.path.basename(test_set).split(".")[0],
                    nseqs_test=len(test_set_truth),
                )
                for averaging in ["micro", "macro", "weighted", None]:
                    performance = f1_score(test_set_truth, predictions, labels=[0, 1, 2, 3], average=averaging)
                    # Per-class metrics
                    if averaging is None:
                        for i, positive_class in enumerate(class_name_shorthand):
                            fold_data[positive_class] = performance[i]
                    # Global metrics
                    else:
                        fold_data[averaging] = performance
                    logger.info(f"{averaging} F1 score = {performance}")

                performance_metrics.append(pd.Series(fold_data))
                # Generate a confusion matrix and save heatmap to file
                confusion = confusion_matrix(test_set_truth, predictions)
                logger.info(f"Confusion matrix:\n{confusion}")
                fig, ax = plot_utils.show_confusion_matrix(
                    confusion,
                    class_names=class_name_shorthand,
                    title=f"{dataset_name}, Fold {fold}\n{test_set_name}",
                    figax=plt.subplots(figsize=plot_utils.get_figsize(0.4, 1)),
                )
                plot_utils.save_fig(fig, os.path.join(fold_dir, f"confusionMatrix_{test_set_name}"), tight_pad=0)
                plt.close(fig)

    # Write all results to file
    loaders.write_data(
        pd.DataFrame(performance_metrics),
        os.path.join(output_dir, "cnn_dataset_performance_metrics.txt"),
        index=False
    )


if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir")
    parser.add_argument("--nfolds", type=int, default=NFOLDS)
    parser.add_argument("--activity_file", default=os.path.join(DATA_DIR, "activity_summary_stats_and_metadata.txt"))
    parser.add_argument("--modeling_splits", default=os.path.join(DATA_DIR, "modeling_splits.txt"))
    parser.add_argument("--epochs", type=int, default=NEPOCHS)
    parser.add_argument("--report_train", action="store_true", help="If specified, generate a confusion matrix and performance metrics for the training and validation datasets.")
    args = parser.parse_args()
    main(**vars(args))
