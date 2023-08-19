#!/usr/bin/env python3
"""
Perform global importance analysis by dinucleotide shuffling the genomic sequences and calculating the change in
activity when specific, fixed features are injected into those sequences.
"""
import os
import argparse
import itertools

import numpy as np
import torch
from deeplift.dinuc_shuffle import dinuc_shuffle

from mpra_tools import fasta_utils, log, loaders, modeling, predicted_occupancy

PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, "../")
DATA_DIR = os.path.join(PATH, "Data")


def get_consensus(pwm):
    """Helper function to get the most likely base at each position, and return that sequence as a str."""
    return pwm.apply("idxmax", axis=1).str.cat()


def get_motif_combos(motifs, max_motifs=4):
    """Generator to enumerate all possible combinations of motifs from 1 to max_motifs, with replacement."""
    for n in range(1, max_motifs + 1):
        for combo in itertools.product(motifs.index, repeat=n):
            yield list(combo)


def cnn_predict(sequences, model, batch_size=128, use_cuda=True):
    """Helper function to one-hot encode sequences, batch them, make predictions using the provided model,
    and then flatten the result."""
    sequences_hot = modeling.one_hot_encode(sequences)
    batched = [sequences_hot[i:i+batch_size] for i in range(0, len(sequences_hot), batch_size)]
    scores = modeling.cnn_predict_batches(model, batched, use_cuda=use_cuda)
    return scores.flatten()


def add_features(seqs, features, locations):
    """Given a Series of sequences, replace the subsequence starting at a given location with the given sequence. If
    features is a list, they are inserted in the orders listed in locations, which is also a list."""
    # If either the feature or the location is a single element, wrap it in a list
    if type(features) == str:
        features = [features]
    if type(locations) == int:
        locations = [locations]
    # Make sure that both the features and the locations are a list, and that there are enough locations
    if type(features) is not list or type(locations) is not list:
        raise ValueError(f"Expected a list of features or a str, and a list of locations or an int. Got a "
                         f"{type(features)} and a {type(locations)}")
    if len(features) > len(locations):
        raise ValueError("Not enough locations for the provided features.")

    new_seqs = seqs.copy()
    for feat, loc in zip(features, locations):
        new_seqs = new_seqs.str[:loc] + feat + new_seqs.str[loc+len(feat):]
    return new_seqs


def log_sequences(logger, seqs):
    """Log a few full length sequences from the provided series."""
    logger.info(seqs.head().values)


def main(model_dir, output_dir, seed, dev):
    logger = log.get_logger()
    logger.info("Loading in data.")
    activity_df = loaders.load_data(os.path.join(DATA_DIR, "activity_summary_stats_and_metadata.txt"))
    # Get genomic sequences from first 2 libraries
    genomic_seqs = activity_df[activity_df["original_genomic"]]
    # Put the sequence label on the index so we can keep track of the original sequence each scramble came from
    genomic_seqs = genomic_seqs.set_index("label")["sequence"]
    logger.info("Loading in PWMs.")
    pwms = predicted_occupancy.read_pwm_files(os.path.join(DATA_DIR, "Downloaded", "eLifeMotifs.meme"))
    pwms = pwms.rename(lambda x: x.split("_")[0])
    # Get the consensus sequence for each TF
    consensus_seqs = pwms.apply(get_consensus)
    logger.info(f"Consensus sequences to use:\n{consensus_seqs}")
    logger.info("Loading in the CNN.")
    model = loaders.load_cnn(os.path.join(model_dir, "best_model.pth.tar"))
    model.eval()

    logger.info("Dinucleotide shuffling each sequence one time.")
    rng = np.random.RandomState(seed)
    shuffled_seqs = genomic_seqs.apply(dinuc_shuffle, rng=rng)
    if dev:
        shuffled_seqs = shuffled_seqs.head()
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()

    shuffled_df = shuffled_seqs.rename("shuffle_seq").to_frame()
    logger.info("Done shuffling.")
    logger.info(shuffled_df)
    shuffled_df["gc_content"] = fasta_utils.gc_content(shuffled_df["shuffle_seq"])

    logger.info("Making predictions on shuffled sequences.")
    log_sequences(logger, shuffled_seqs)
    if use_cuda:
        model.cuda()

    shuffled_df["background"] = cnn_predict(shuffled_seqs, model, use_cuda=use_cuda)
    crx_motif = "TTAATCCC"
    nrl_motif = "AATTTGCTGAC"
    i = 0
    for crx_position in range(len(shuffled_seqs[0]) - len(crx_motif) + 1): #Assumes all shuffled are of same lengths
        crx_seqs = add_features(shuffled_seqs, crx_motif, crx_position)
        for nrl_position in range(len(shuffled_seqs[0]) - len(nrl_motif) + 1): #Assumes all shuffled are of same lengths
            if nrl_position <= crx_position  - len(nrl_motif) or nrl_position >= crx_position + len(crx_motif):
                i += 1
                crx_nrl_seqs = add_features(crx_seqs, nrl_motif, nrl_position)
                logger.info(f"Testing sequences with the following crx, nrl position : {crx_position}, {crx_motif}, {nrl_position}, {nrl_motif}")
                log_sequences(logger, crx_nrl_seqs)
                encoding = f"{crx_position}crx{nrl_position}nrl"
                shuffled_df[encoding] = cnn_predict(crx_nrl_seqs, model, use_cuda=use_cuda)

    logger.info(f"Tested {i} motif combinations, scores to file.")
    shuffled_df.to_csv(os.path.join(output_dir, "synthetic_insertion_preds.txt"), sep="\t")

if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_dir", help="Directory with the best_model.pth.tar checkpoint of the resnet to use.")
    parser.add_argument("output_dir", help="Directory where all output files will be written.")
    parser.add_argument("--seed", type=int, default=419, help="Seed for RNG.")
    parser.add_argument("--dev", action="store_true", default=False,
                        help="Use a small amount of data and no CUDA for development purposes."
                        )
    args = parser.parse_args()
    main(**vars(args))
