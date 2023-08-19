#!/usr/bin/env python3
"""
Fit a SVM on various portions of the data using cross-validation. For each portion of the data,
only do cross-validation on the new part of the data. The held-out part from the CV is ignored. Then assess model
performance on the test set.
"""
import os
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC
from preimage.kernels.generic_string import GenericStringKernel

from mpra_tools import log, loaders, modeling, plot_utils

# Global parameters
SEED = 1260
NFOLDS = 10
KMER_SIZE = 6
SIGMA_POS = 10
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "../")
DATA_DIR = os.path.join(PATH, "Data")
class_name_shorthand = ["Si", "In", "WE", "SE"]
plot_utils.set_manuscript_params()


###################
### MAIN SCRIPT ###
###################
def main(output_dir, checkpoint, nfolds, activity_file, modeling_splits):
    logger = log.get_logger()
    sequence_key = "sequence"
    activity_key = "activity_bin"
    logger.info("Reading in data.")
    activity_df = loaders.load_data(activity_file)
    activity_df[activity_key] = activity_df[activity_key].apply(loaders.activity_to_int)
    training_sets = loaders.load_data(modeling_splits)
    logger.info(f"Planned data splits:\n{training_sets}")

    train_df, test_df, _ = loaders.train_test_split(activity_df, "svm")
    # Remove duplicate sequences in the training data
    train_df = train_df.drop_duplicates(subset=sequence_key, keep="first")
    train_df = train_df.reset_index()
    logger.info(f"Number of sequences available for training: {len(train_df)}")
    logger.info(f"Number of sequences in the test set: {len(test_df)}")

    kernel = GenericStringKernel("dna_kmer", n_min=KMER_SIZE, n_max=KMER_SIZE, sigma_position=SIGMA_POS, sigma_properties=0)
    train_seqs = train_df[sequence_key]
    test_seqs = test_df[sequence_key]
    test_labels = test_df[activity_key]

    if checkpoint:
        logger.info("Loading in the precomputed Gram matrices...")
        train_gram = kernel.set_gram_matrix_from_file(os.path.join(output_dir, "trainGram.txt"))
        test_gram = np.load(os.path.join(output_dir, "testGram.npy"))
    else:
        logger.info("Precomputing the training Gram matrix...")
        train_gram = kernel(train_seqs, train_seqs)
        kernel.save_gram_lower_triangle(train_gram, os.path.join(output_dir, "trainGram.txt"))
        logger.info("Precomputing the first test Gram matrix...")
        test_gram = kernel(test_seqs, train_seqs)
        np.save(os.path.join(output_dir, "testGram.npy"), test_gram)

    logger.info("Done computing Gram matrices.")
    test_set_info = [(test_gram, "test_set", test_labels)]

    # Test set info to log:
    # 1. Training data name
    # 2. CV fold
    # 3. Size of the training data (previous data + the part CV-ed)
    # 4. The test set used
    # 5. Micro, macro, weighted F1 scores, and per-class scores
    performance_metrics = []
    for dataset_name, grouping_info in training_sets.iterrows():
        logger.info(f"Performing cross-validation with the {dataset_name} dataset.")
        dataset_df, prev_idx, cv_df = loaders.prepare_cross_validation(train_df, grouping_info)
        logger.info(f"{len(dataset_df)} sequences available for training.")
        logger.info(f"Of which, {len(cv_df)} will be cross-validated.")
        logger.info(f"Composition of classes in the full dataset:\n"
                    f"{dataset_df[activity_key].value_counts(normalize=True).sort_index()}")
        logger.info(f"Composition of classes in the new data:\n"
                    f"{cv_df[activity_key].value_counts(normalize=True).sort_index()}")

        # Grab the class labels for the new data, since that is what will stratify the CV
        cv_labels = cv_df[activity_key]
        cv = modeling.subset_cross_validation(cv_labels, prev_idx, nfolds, seed=SEED)
        for fold, (full_subset_idx, fake_val_idx) in enumerate(cv, 1):
            logger.info(f"Fold {fold}.")
            dataset_size = len(full_subset_idx)

            # Subset the Gram matrix, get the labels, and fit an SVM
            train_gram_subset = modeling.subset_square_gram(train_gram, full_subset_idx)
            train_labels = dataset_df.loc[full_subset_idx, activity_key]
            svm = SVC(kernel="precomputed", probability=True)
            svm.fit(train_gram_subset, train_labels)
            del train_gram_subset
            fold_dir = os.path.join(output_dir, dataset_name, str(fold))
            os.makedirs(fold_dir, exist_ok=True)
            pickle.dump(svm, open(os.path.join(fold_dir, "svm.pkl"), "wb"))

            # Make predictions on the test sets
            for gram, name, labels in test_set_info:
                test_gram_subset = gram[:, full_subset_idx]
                log_probs = svm.predict_log_proba(test_gram_subset)
                del test_gram_subset
                np.save(os.path.join(fold_dir, f"{name}_log_probs.npy"), log_probs)

                predictions = log_probs.argmax(axis=1)
                fold_data = dict(
                    dataset=dataset_name,
                    fold=fold,
                    nseqs_train=len(full_subset_idx),
                    test_set=name,
                )
                for averaging in ["micro", "macro", "weighted", None]:
                    performance = f1_score(labels, predictions, labels=[0, 1, 2, 3], average=averaging)
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
                confusion = confusion_matrix(labels, predictions)
                logger.info(f"Confusion matrix:\n{confusion}")
                fig, ax = plot_utils.show_confusion_matrix(
                    confusion,
                    class_names=class_name_shorthand,
                    title=f"{dataset_name}, Fold {fold}"
                )
                plot_utils.save_fig(fig, os.path.join(
                    fold_dir, f"confusionMatrix_{name}"), tight_pad=0)
                plt.close(fig)

    # Write all results to file
    loaders.write_data(
        pd.DataFrame(performance_metrics),
        os.path.join(output_dir, "svm_dataset_performance_metrics.txt"),
        index=False
    )


if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir")
    parser.add_argument("--checkpoint", action="store_true", help="If specified, load the pre-computed Gram matrix from the output directory.")
    parser.add_argument("--nfolds", type=int, default=NFOLDS)
    parser.add_argument("--activity_file", default=os.path.join(DATA_DIR, "activity_summary_stats_and_metadata.txt"))
    parser.add_argument("--modeling_splits", default=os.path.join(DATA_DIR, "modeling_splits.txt"))
    args = parser.parse_args()
    main(**vars(args))
