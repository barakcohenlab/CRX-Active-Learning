#!/usr/bin/env python3
"""
Join the big table of processed and harmonized activity measurements with extra metadata information
to report as final data, and to aid in downstream model fitting.
"""
import os
import argparse

import numpy as np
from scipy import stats
import pandas as pd

from mpra_tools import fasta_utils, loaders, log, processing

PATH = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(PATH, "../", "Data")
fasta_dir = os.path.join(data_dir, "Sequences", "Fasta")
logger = log.get_logger()

# Helper functions
def get_immediate_percursor(split):
    """Given a sequence label split on semicolons, determine what sequence in the dataset, if any,
    was used to create the current sequence"""
    if len(split) == 0:
        raise ValueError("Split on an empty string!")

    # If there aren't any semicolons, then this sequence is either from the eLife libraries, is a
    # scrambled sequence, or is a "building block" sequence not used in this study.
    if len(split) == 1:
        # Delim in the original sequence is an underscore
        subsplit = split[0].split("_")
        if subsplit[0] == "building-block":
            # Completely synthetic, no precursor
            return np.nan

        # If not a completely synthetic sequence, format is coordinates_tags_variant-type
        annot = subsplit[2]
        if annot == "WT":
            # Genomic sequence, no precursor
            return np.nan
        elif annot == "MUT-allCrxSites" or annot == "MUT-shape" or annot == "scrambled":
            return subsplit[0] + "_" + subsplit[1] + "_WT"
        else:
            raise ValueError(
                f"Did not recognize sequence mutation type in {split[0]}"
            )

    # If there are semicolons, then the immediate precursor is just everything except for what came
    # after the last semicolon
    else:
        return ";".join(split[:-1])


def get_original_sequence(split):
    """Given a sequence label split on underscores, get the original genomic sequence this sequence
    is derived from, formatted as the coordinates and the 4-letter code used in the eLife paper. If
    a synthetic building block, there is no original sequence."""
    if split[0] == "building-block":
        return np.nan
    else:
        return "_".join(split[:2])


def assign_batch_name(row, r3a_idx):
    """Given a row of data with several columns of metadata, and an index of sequences belonging to
    Round 3A, annotate which active learning batch of data the item belongs to."""
    result = np.nan
    # Each row should only be true for one of the columns in the below list
    columns_to_check = ["original_genomic", "mut_all_crx", "mut_shape",
                        "entropy_sampling", "random_sampling",
                        "high_conf_pilot", "high_conf_cnn"]
    if row[columns_to_check].sum() > 1:
        # Flag for rows that have non-unique tags
        result = "NOT_UNIQUE_TAG"
    elif row["original_genomic"]:
        result = "Genomic"
    elif row[["scrambled", "standard_seq", "ic_scan", "rational_mutagenesis",
              "test_set", "derived_from_test_set_seq", "cnn_validation_set",
              "high_conf_pilot", "l9_repeat_l8", "l9_controls"]].any():
        # Sanity check flag for items that are not training data
        result = "NOT_TRAINING_DATA"
    # For the purposes of the modeling, group the shape mutants in with the motif mutants
    elif row[["mut_all_crx", "mut_shape"]].any():
        result = "CrxMotifMutant"
    elif row["entropy_sampling"]:
        library = row["library"]
        if library == "library3":
            result = "Round2"
        elif library == "library4" or library == "library5":
            if row.name in r3a_idx:
                result = "Round3a"
            else:
                result = "Round3b"
        elif library[:-1] == "library8":
            result = "Round4a"
        else:
            # Flag for rows that are tagged for entropy sampling but cannot be attributed to a batch
            result = "NOT_ID_ACTIVE_ROUND"
    elif row["margin_sampling"]:
        if row["library"][:-1] == "library9":
            result = "Round4b"
        else:
            # Flag for rows tagged as margin sampling but cannot be attributed as such
            result = "NOT_ID_MARGIN"
    elif row["random_sampling"]:
        if row["library"] == "library6" or row["library"] == "library7":
            result = "Round3c"
        else:
            # Flag for rows that are tagged for random sampling but cannot be attributed as such
            result = "NOT_ID_RANDOM"
    elif row["high_conf_cnn"]:
        result = "HighConfidence"
    return result


###################
### MAIN SCRIPT ###
###################
def main(activity_file, output_file):
    activity_df = loaders.load_data(activity_file)
    # Note that the data as-is has the sequence ID on the row index, but after all the processing is
    # done, the row index will be a unique ID
    logger.info(
        f"Loaded in activity summary statistics. There are {len(activity_df)} measurements:\n{activity_df.head()}"
    )
    # Drop sequences not measured
    activity_df = activity_df.dropna()
    logger.info(
        f"After dropping sequences that were not measured, {len(activity_df)} measurements remain."
    )

    # Read in the standards and the scrambled sequences, and annotate for them
    standards = fasta_utils.read_fasta(os.path.join(fasta_dir, "standards.fasta"))
    scrambled = fasta_utils.read_fasta(os.path.join(fasta_dir, "scrambled.fasta"))
    activity_df["standard_seq"] = activity_df.index.isin(standards.index)
    activity_df["scrambled"] = activity_df.index.isin(scrambled.index)

    # Annotate for precursors and original sequence, then remove the building block sequences (not
    # part of this study)
    idx_str = activity_df.index.str
    activity_df["immediate_precursor"] = idx_str.split(
        ";").map(get_immediate_percursor)
    activity_df["original_seq"] = idx_str.split("_").map(get_original_sequence)
    building_block_mask = activity_df["original_seq"].isna()
    logger.info(
        f"Dropping {building_block_mask.sum()} 'building block' sequences not used in this study.")
    activity_df = activity_df[~building_block_mask]
    # Helper handles
    df_idx = activity_df.index
    idx_str = df_idx.str
    idx_isin = df_idx.isin
    lib_str = activity_df["library"].str

    # Convert quantitative measurements into discretized classes
    activity_df["activity_bin"] = activity_df["expression_log2"].apply(processing.discretize_activity)
    logger.info("Number of sequences in each library belonging to each class:")
    logger.info(activity_df.groupby("library")["activity_bin"].value_counts().unstack())

    # Label things that were not designed using the perturbation procedures Anything with the "HAND"
    # tag from libraries 4-7 was designed by hand, i.e. with rational mutagenesis
    activity_df["rational_mutagenesis"] = idx_str.contains(
        "HAND") & lib_str.contains("[4567]")
    # Anything with the "ATAC" tag is a photoreceptor ATAC-seq peak that was scanned with the
    # information content model and used to predict sequence from the genome
    activity_df["ic_scan"] = idx_str.contains("ATAC")
    # If the sequence ID ends in "WT" *and is from L1 or L2* it is an original genomic sequence.
    # Sequences that end in "WT" in other libraries are the standards used in each library
    activity_df["original_genomic"] = idx_str.contains(
        "WT") & lib_str.contains("[12]")
    # "MUT-allCrxSites" means all CRX motifs were mutated by point mutation. "MUT-shape" scrambled a
    # putative shape motif. Sequences ending with these tags are only in the original 2 libraries
    activity_df["mut_all_crx"] = idx_str.contains("MUT-allCrxSites$")
    activity_df["mut_shape"] = idx_str.contains("MUT-shape$")

    # Prepare to annotate sequences generated with the perturbation procedure by loading in the
    # relevant metadata
    logger.info("Loading in various metadata files.")
    # Anything in L3 that is not scrambled or a standard is from round 2
    round_2_mask = lib_str.contains("3") & ~(
        activity_df[["standard_seq", "scrambled", "original_genomic"]].any(axis=1))
    logger.info(f"{round_2_mask.sum()} sequences match Round 2 criteria.")
    # Round 3
    round_3_uncertain_seqs = fasta_utils.read_fasta(os.path.join(
        data_dir, "LibraryDesign", "Round3", "mostUncertain.fasta"))
    logger.info(
        f"Read in {len(round_3_uncertain_seqs)} sequences sampled with entropy in Round 3.")
    round_3_uncertain_seqs = round_3_uncertain_seqs[round_3_uncertain_seqs.index.isin(
        df_idx)]
    logger.info(f"Of which, {len(round_3_uncertain_seqs)} were measured.")
    # Calculate the entropies from round 3 to stratify the 5000 most uncertain from the rest
    round_3_probs = loaders.load_data(os.path.join(
        data_dir, "LibraryDesign", "Round3", "candidateProbabilities.txt.gz"), compression="gzip")
    round_3_probs = round_3_probs.loc[round_3_uncertain_seqs.index]
    round_3_entropies = round_3_probs.apply(
        stats.entropy, axis=1, base=2).sort_values(ascending=False)
    # A small number of high confidence sequences were sampled in round 3
    round_3_high_conf_seqs = pd.concat([
        fasta_utils.read_fasta(os.path.join(
            data_dir, "LibraryDesign", "Round3", f"highConfidence{i}.fasta"))
        for i in ["Silencer", "Inactive", "WeakEnhancer", "StrongEnhancer"]
    ])
    round_3_high_conf_seqs = round_3_high_conf_seqs[round_3_high_conf_seqs.index.isin(
        df_idx)]
    logger.info(
        f"Measured {len(round_3_high_conf_seqs)} high confidence sequences from Round 3."
    )
    # Library 6 and 7 is part of Round 3. Many sequences are not relevant for this study, but those
    # tagged as "Active Learning Controls" are randomly sampled
    sampling_control_tags = loaders.load_data(
        os.path.join(data_dir, "LibraryDesign", "Round3", "library6And7Tags.txt"),
        header=None,
    ).squeeze()
    sampling_control_tags = sampling_control_tags[
        sampling_control_tags.str.contains("ActiveLearningControl") &
        sampling_control_tags.index.isin(df_idx)
    ]
    logger.info(
        f"Measured {len(sampling_control_tags)} randomly sampled sequences from Round 3."
    )
    # Round 4 Library 8
    round_4a_tags = loaders.load_data(
        os.path.join(data_dir, "LibraryDesign", "Round4", "library8Tags.txt"),
        header=None,
    ).squeeze()
    round_4a_tags = round_4a_tags[round_4a_tags.index.isin(df_idx)]
    round_4a_uncertain_tags = round_4a_tags[round_4a_tags.str.contains(
        "MostUncertain")]
    logger.info(
        f"Measured {len(round_4a_uncertain_tags)} sequences sampled with entropy in Round 4.")
    # Anything NOT tagged as MostUncertain in this library is a high confidence sequence
    round_4a_high_conf_tags = round_4a_tags[~round_4a_tags.str.contains(
        "MostUncertain")]
    logger.info(
        f"Measured {len(round_4a_high_conf_tags)} high confidence sequences from Round 4.")

    # Library 9
    round_4b_tags = loaders.load_data(
        os.path.join(data_dir, "LibraryDesign", "Round4", "library9Tags.txt"),
        header=None,
    ).squeeze()
    round_4b_tags = round_4b_tags[round_4b_tags.index.isin(df_idx)]
    round_4b_margin_tags = round_4b_tags[round_4b_tags.str.contains("Margin")]
    logger.info(
        f"Measured {len(round_4b_margin_tags)} sequences sampled with margin uncertainty in Round 4."
    )
    round_4b_repeated_tags = round_4b_tags[round_4b_tags.str.contains("Repeats")]
    logger.info(
        f"Measured {len(round_4b_repeated_tags)} sequences from L8 repeated in L9."
    )
    round_4b_distr = round_4b_tags[round_4b_tags.str.contains("Known")]
    logger.info(
        f"Measured {len(round_4b_distr)} genomic sequences from L9."
    )

    # Annotate for sampling techniques Something is sampled with entropy if it is (1) round 1, (2)
    # round 4 uncertain, L4 or L5, (3) round 4 uncertain, L8
    activity_df["entropy_sampling"] = round_2_mask | \
        (lib_str.contains("[45]") & idx_isin(round_3_uncertain_seqs.index)) | \
        (lib_str.contains("8") & ~(idx_isin(round_4a_high_conf_tags.index)))
    activity_df["margin_sampling"] = lib_str.contains(
        "9") & idx_isin(round_4b_margin_tags.index)
    activity_df["random_sampling"] = lib_str.contains(
        "[67]") & idx_isin(sampling_control_tags.index)
    activity_df["high_conf_pilot"] = lib_str.contains(
        "[45]") & idx_isin(round_3_high_conf_seqs.index)
    activity_df["high_conf_cnn"] = lib_str.contains(
        "8") & idx_isin(round_4a_high_conf_tags.index)
    # Also annotate some of the special case repeats in L9
    activity_df["l9_controls"] = lib_str.contains(
        "9") & idx_isin(round_4b_distr.index)
    activity_df["l9_repeat_l8"] = lib_str.contains(
        "9") & idx_isin(round_4b_repeated_tags.index)

    # Annotate for sequences belonging to the test set, or something derived from a test set genomic
    # sequence. Anything derived from those genomic sequences cannot be included in training or it
    # will cause information leakage.
    test_set_seqs = fasta_utils.read_fasta(os.path.join(fasta_dir, "mutagenesis_test_set.fasta"))
    # Note that there is an edge case where a WT sequence in the test set is also a standard. Don't
    # double count that sequence!
    activity_df["test_set"] = idx_isin(test_set_seqs.index) & lib_str.contains("[45]") & \
        ~(idx_str.contains("chr9-121694789-121694953_CPPE_WT$") & lib_str.contains("5"))
    test_set_wt = test_set_seqs[test_set_seqs.index.str.contains("_WT$")]
    activity_df["derived_from_test_set_seq"] = activity_df["original_seq"].isin(
        test_set_wt.index.str[:-3])

    # Now annotate for anything in the CNN validation set. Since the CNN was developed using data
    # trained from libraries 1-5, this only applies to those sequences. By definition, nothing in
    # library 8 or 9 derives from these sequences. There may be some sequences in L6 or L7 that
    # overlap here, but since they are not used to train subsequent CNNs, this overlap is tolerable.
    # Also note that there are some sequences that are also standards; we don't want to double count
    # those sequences, so we mask out instances of the standards which are in library 3 or greater
    validation_set_seqs = fasta_utils.read_fasta(
        os.path.join(fasta_dir, "cnn_validation.fasta"))
    activity_df["cnn_validation_set"] = idx_isin(validation_set_seqs.index) &\
        lib_str.contains("[1-5]") & ~(activity_df["standard_seq"] & ~lib_str.contains("[12]"))

    # We can now assign training batch names, but first we need to determine which L4-5 sequences
    # are the most uncertain
    round_3_most_uncertain_idx = round_3_entropies[
        round_3_entropies.index.isin(
            activity_df[~activity_df["derived_from_test_set_seq"]].index)
    ].head(5000).index
    activity_df["data_batch_name"] = activity_df.apply(
        assign_batch_name, axis=1, args=(round_3_most_uncertain_idx,))
    logger.info(
        f"Number of sequences in various batches:\n{activity_df['data_batch_name'].value_counts(dropna=False)}")
    if activity_df["data_batch_name"].isna().sum() > 0:
        logger.warning(
            "Warning, some sequences were not assigned a batch name! Head of the dataframe:")
        logger.warning(activity_df[activity_df["data_batch_name"].isna()].head())

    # Explicitly note what is available training data for the SVM and CNN
    activity_df["svm_train"] = ~(activity_df["derived_from_test_set_seq"])
    activity_df["cnn_train"] = ~activity_df[[
        "derived_from_test_set_seq", "cnn_validation_set"]].any(axis=1)

    # Sanity checks
    mask_counts = activity_df[[
        "standard_seq", "scrambled", "rational_mutagenesis", "ic_scan", "original_genomic",
        "mut_all_crx", "entropy_sampling", "margin_sampling", "random_sampling", "high_conf_pilot",
        "high_conf_cnn", "test_set", "derived_from_test_set_seq", "cnn_validation_set", "svm_train",
        "cnn_train"
    ]].sum()
    logger.info(
        f"Number of sequences True for various metadata columns:\n{mask_counts}")

    n_r3_in_validation = activity_df.loc[lib_str.contains("[89]"), "cnn_validation_set"].sum()
    logger.info(
        f"{n_r3_in_validation} sequences from L8/L9 are in the CNN validation set (expect zero).")

    missing_seq = activity_df[activity_df["sequence"].isna()]
    logger.info(f"Rows that are missing a sequence:\n{missing_seq}")

    test_set_df = activity_df[activity_df["test_set"]]
    logger.info(f"Test set has {len(test_set_df)} sequences. Class composition is:")
    logger.info(test_set_df["activity_bin"].value_counts()[[
        "Silencer", "Inactive", "WeakEnhancer", "StrongEnhancer"
    ]])

    # Move the sequence label to a column so that every row has a unique index, then write to file
    loaders.write_data(
        activity_df.reset_index(),
        output_file,
        index_label="unique_index",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        "--activity_file", default=os.path.join(data_dir, "Measurements", "joined_activity_data.txt")
        )
    parser.add_argument(
        "--output_file", default=os.path.join(data_dir, "activity_summary_stats_and_metadata.txt")
        )
    args = parser.parse_args()
    main(**vars(args))
    