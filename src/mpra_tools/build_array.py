"""Functions for building oligo pools."""
import os
import sys
import itertools

import numpy as np
import pandas as pd

from . import fasta_utils


ecori = "GAATTC"
spei = "ACTAGT"
sphi = "GCATGC"
noti = "GCGGCCGC"


def generate_barcodes(bc_size=9, min_gc=0.125, max_gc=0.875, homopolymer_cutoff=4,  denylist=None):
    """Generate all possible barcodes with a minimum Hamming distance of 2, excluding those with extreme GC content
    or matching something in the denylist. Always exclude barcodes that match any of the following criteria:
    * Homopolymer runs
    * EcoRI, SpeI, SphI, or NotI sites
    * Partial SphI sites (first 4 bp) or partial NotI sites (last 6 bp)

    Parameters
    ----------
    bc_size : int
        Size of barcodes to generate.
    min_gc : float
        Minimum GC content for barcodes.
    max_gc : float
        Maximum GC content for barcodes.
    homopolymer_cutoff : int
       Barcodes with homopolymers of this length or larger will be excluded. 
    denylist : list[str]
        If specified, contains additional sequences to exclude from the list of barcodes (multiplexing sequences, etc.)

    Returns
    -------
    barcodes : pd.Series
        List of all allowed barcodes of the given length
    """
    # If denylist is specified, make sure it is a list
    if denylist and type(denylist) is not list:
        print("Warning, denylist is specified but is not a list. Ignoring!")
        denylist = None
    # If there is no denylist, make an empty list
    if denylist is None:
        denylist = []

    # Generate the list of things to exclude
    denylist += [i * homopolymer_cutoff for i in ["A", "C", "G", "T"]]
    denylist += [ecori, spei, sphi, noti, sphi[:-2], noti[2:]]

    int_to_base = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
    }
    # Generate all possible (k-1)-mers
    barcodes = pd.DataFrame(list(itertools.product([0, 1, 2, 3], repeat=bc_size - 1)))
    # Checksum to get the last position
    barcodes[bc_size - 1] = barcodes.apply(lambda x: x.sum() % 4, axis=1)
    # Convert the barcodes from ints to bases
    barcodes = barcodes.apply(lambda x: "".join([int_to_base[i] for i in x]), axis=1)

    # Check for matches to the denylist
    deny_mask = fasta_utils.has_restriction_sites(barcodes, denylist)
    # Check for extreme GC content
    gc_content = fasta_utils.gc_content(barcodes)
    deny_mask = deny_mask | (gc_content < min_gc) | (gc_content > max_gc)
    barcodes = barcodes[~deny_mask]

    # Report per-position distribution of barcodes
    per_position_distr = barcodes.apply(lambda bc: pd.Series(list(bc))).apply(
        lambda pos: pos.value_counts(normalize=True))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Frequency of each base at each position among all barcodes:")
        print(per_position_distr)

    return barcodes


def build_oligos(sequences, basal, barcodes, fprimer, rprimer, bc_per_seq, nbasal_bc, seed=0):
    """Given a list of sequences, the construct for basal, a list of barcodes, primers, and the number of barcodes to
    assign to a sequence, randomly take a subset of barcodes and then build oligos as follows:
        fprimer - EcoRI - library sequence - SpeI - C - SphI - barcode - NotI - rprimer
    For basal, the construct is:
        fprimer - EcoRI - SpeI - basal - C - SphI - barcode - NotI - rprimer
    The number of available barcodes must be at least as large as nbasal_bc + bc_per_seq * len(sequences).
    Otherwise, nothing is returned.

    Parameters
    ----------
    sequences : pd.Series
        List of library sequences, with index being unique ID and value being the 164 bp sequence
    basal : str
        The filler sequence used to clone the basal promoter
    barcodes : pd.Series
        List of all allowed barcodes
    fprimer : str
        The forward priming sequence
    rprimer : str
        The reverse priming sequence. IMPORTANT: this sequence should be the reverse compliment of the actual primer
        that is used, since this sequence is at the 3' end of the oligo.
    bc_per_seq : int
        Number of unique barcodes to assign to each library sequence.
    nbasal_bc : int
        Number of unique barcodes to assign to the basal construct.
    seed : int
        Random seed for subsampling barcodes.

    Returns
    -------
    oligo_pool : pd.DataFrame
        Index is the barcode, columns are the unique sequence ID name and the oligo
    """
    # Make sure no sequence is repeated
    unique_sequences = sequences.drop_duplicates()
    if len(unique_sequences) < len(sequences):
        n_removed = len(sequences) - len(unique_sequences)
        sequences = unique_sequences
        print(f"Warning, some sequences are repeated! Removed {n_removed} duplicates.")

    # Check to make sure the sequences don't have restriction sites or the priming sequences
    denylist = [ecori, spei, sphi, noti, fprimer, rprimer]
    has_denylist = fasta_utils.has_restriction_sites(sequences, denylist)
    if has_denylist.sum() > 0:
        print(f"Warning, {has_denylist.sum()} sequences were removed for matches to restriction sites or primers.")
        sequences = sequences[~has_denylist]

    # Report how many sequences have long runs -- we don't want this to be too high
    long_run_mask = fasta_utils.has_restriction_sites(sequences, [i * 10 for i in "ACGT"])
    print(f"{long_run_mask.sum()} sequences have mononucleotide runs of 10 or more.")

    nbarcodes = len(sequences) * bc_per_seq + nbasal_bc
    if nbarcodes > len(barcodes):
        raise ValueError("Not enough barcodes provided to build the oligos.")

    # Get a random sample of the available barcodes
    barcodes = barcodes.sample(n=nbarcodes, random_state=seed)
    oligo_pool = {}
    # Counter for looping over the BCs
    bc_idx = 0
    # First make the basal construct
    for i in range(nbasal_bc):
        bc = barcodes.iloc[bc_idx]
        oligo = fprimer + ecori + spei + basal + "C" + sphi + bc + noti + rprimer
        oligo_pool[bc] = ("BASAL", oligo)
        bc_idx += 1

    # Now make oligos of library sequences
    for seq_id, seq in sequences.items():
        for i in range(bc_per_seq):
            bc = barcodes.iloc[bc_idx]
            oligo = fprimer + ecori + seq + spei + "C" + sphi + bc + noti + rprimer
            oligo_pool[bc] = (seq_id, oligo)
            bc_idx += 1

    # Confirm that the right number of oligos were made
    if len(oligo_pool) != nbarcodes:
        raise ValueError("Number of oligos does not match the specified number of barcodes. Something is wrong with "
                         "the code!")

    oligo_pool = pd.DataFrame.from_dict(oligo_pool, orient="index", columns=["label", "oligo"])
    oligo_pool = oligo_pool.rename_axis(index="barcode")

    # Check and make sure there is exactly one instance of each cut site and priming site in each sequence.
    sites_and_names = [
        (ecori, "EcoRI"),
        (spei, "SpeI"),
        (sphi, "SphI"),
        (noti, "NotI"),
        (fprimer, "F primer"),
        (rprimer, "R primer"),
    ]
    for site, name in sites_and_names:
        site_counts = oligo_pool["oligo"].str.count(site)
        if (site_counts == 0).sum() > 0:
            print(f"Warning: sequences are missing a {name} site! The sequences are:")
            print(oligo_pool[site_counts == 0])
        if (site_counts > 1).sum() > 0:
            print(f"Warning, sequences contain multiple {name} sites! The sequences are:")
            print(oligo_pool[site_counts > 1])

    return oligo_pool
