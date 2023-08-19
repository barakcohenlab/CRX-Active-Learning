#!/usr/bin/env python3
"""Process multiple MPRA libraries in parallel. For each library, read in the
counts files and do standard QC based on the provided information. Once all
libraries have been processed to activity scores, join libraries together."""

import os
import argparse

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

from mpra_tools import config, log, processing, plot_utils, loaders, fasta_utils
PATH = os.path.dirname(os.path.abspath(__file__))


def main(data_dir, config_file):
    plot_utils.set_manuscript_params()
    logger = log.get_logger()
    configs = config.load_configs(config_file)
    data_dir = data_dir
    logger.info(f"Changing into the directory {data_dir}")
    os.chdir(data_dir)
    logger.info(f"Read in the following configs:\n{configs}")

    activity_df, seqs, qc_metrics = processing.process_each_library(configs)
    loaders.write_data(qc_metrics,
                       os.path.join(data_dir, configs["output_dir"], "joined_qc_metrics.txt"),
                       index=False,
                       )
    loaders.write_data(activity_df,
                       os.path.join(data_dir, configs["output_dir"], "joined_redundant_activity.txt"),
                       )
    fasta_utils.write_fasta(seqs, os.path.join(data_dir, configs["output_dir"], "joined_libraries.fasta"))
    logger.info(f"There are {len(activity_df)} measurements and {len(seqs)} sequences.")
    # Drop any rows where the seqID is not in the list of sequences. These correspond to cases where
    # two different names were assigned the same sequence variant.
    activity_df = activity_df[activity_df.index.isin(seqs.index)]
    # Double check and make sure there is a one-to-one mapping between the labels assigned to
    # activity measurements nad the labels assigned to sequences.
    seqs = seqs[activity_df.index]
    seqs = seqs.reset_index().drop_duplicates(subset="label").set_index("label").squeeze()
    logger.info(
        f"After removing duplicates, there are {len(activity_df)} measurements and {len(seqs)} unique sequences."
    )

   # Write this complete set of measurements and associated sequences to file
    activity_df["sequence"] = seqs
    loaders.write_data(
        activity_df,
        os.path.join(data_dir, configs["output_dir"], "joined_activity_data.txt"),
    )

    logger.info("Preparing the retinopathy dataset.")
    retinopathy_df = pd.read_parquet(os.path.join(data_dir, "Downloaded", "retinopathy_data.parquet"))
    retinopathy_metadata = loaders.load_data(
        os.path.join(data_dir, "Downloaded", "retinopathy_metadata.txt"),
        index_col=None,
    )
    retinopathy_df = retinopathy_df.set_index("library_id").join(retinopathy_metadata.set_index("library_id"))
    retinopathy_df = retinopathy_df[
        (retinopathy_df["library"] == "rho") &\
        (retinopathy_df["genotype"] == "WT")
    ]
    # Calculate activity vs basal
    retinopathy_df["activity_vs_basal"] = retinopathy_df["activity_mean"] / retinopathy_df.loc["basal", "activity_mean"]
    retinopathy_df = retinopathy_df.drop(index="basal")
    retinopathy_df = retinopathy_df.reset_index()
    retinopathy_df["expression_log2"] = retinopathy_df["activity_vs_basal"].apply(np.log2)
    # Rename activity classes and a few columns
    retinopathy_df["activity_bin"] = retinopathy_df["activity_class"].replace({
        "strong_enhancer": "StrongEnhancer",
        "weak_enhancer": "WeakEnhancer",
        "inactive": "Inactive",
        "weak_silencer": "Silencer",
        "strong_silencer": "Silencer"
    })
    retinopathy_df = retinopathy_df.rename(columns={
        "CRE_sequence": "sequence",
        "library_id": "label",
    })
    # Pull out the WT sequences
    retinopathy_df = retinopathy_df[retinopathy_df["variant_type"].str.contains("wildtype")]
    loaders.write_data(retinopathy_df, os.path.join(data_dir, "retinopathy_reformatted.txt"))


if __name__ == "__main__":
    data_dir = os.path.join(PATH, "../", "Data")
    config_file = os.path.join(data_dir, "processing_config.yml")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--config_file", default=config_file)
    args = parser.parse_args()
    main(**vars(args))
