"""Functions for loading in data, splitting different subsets, and other useful manipulations."""
import os
import pandas as pd
import torch


def load_data(file, index_col=0, **kwargs):
    """Wrapper for reading in an arbitrary tab-delimited file as a DataFrame/Series. Assumes the first column is the
    index column. Extra arguments for pd.read_csv."""
    return pd.read_csv(file, sep="\t", index_col=index_col, na_values="NaN", **kwargs)


def write_data(df, file, **kwargs):
    """Wrapper for writing a Series or DataFrame to file with consistent sep and na_rep"""
    df.to_csv(file, sep="\t", na_rep="NaN", **kwargs)
    
    
def get_strong_cutoff():
    """Wrapper to get the strong enhancer cutoff, which is a constant value."""
    return 2.8384278422293634


def activity_to_int(val):
    """Encode Silencer, Inactive, WeakEnhancer, and StrongEnhancer as an int value."""
    if val == "Silencer":
        return 0
    elif val == "Inactive":
        return 1
    elif val == "WeakEnhancer":
        return 2
    elif val == "StrongEnhancer":
        return 3
    else:
        raise ValueError(f"Did not recognize activity name: {val}")


def train_test_split(df, model):
    """
    Split the data based on the following columns:
    * The test set is True for test_set
    * If `model` == "svm", the training data is True for `svm_train`. If `model` == "cnn", it is True for `cnn_train`.
    * Additionally, if the model is the CNN, the validation set is `cnn_validation_set`.

    Returns
    -------
    (train_df, test_df, validate_df)
    Each entry in the Tuple is a subset of the original df. If validate=False, then validate_df=None
    """
    if model == "cnn":
        train_df = df[df["cnn_train"]]
        validate_df = df[df["cnn_validation_set"]]
    elif model == "svm":
        train_df = df[df["svm_train"]]
        validate_df = None
    else:
        raise ValueError(f"Model type not recognized. Model must be `cnn` or `svm`, saw {model}.")

    test_df = df[df["test_set"]]
    return train_df, test_df, validate_df


def prepare_cross_validation(df, grouping_info):
    """
    Split the data for cross-validation, where only a subset should be cross-validated.
    * df should have a column called `data_batch_name`
    * Index of grouping_info are possible values in `data_batch_name`
    * If a data batch is `old` in grouping_info, it is considered part of the "previous" data. If a batch is `new`, it
    is part of the current data to do cross-validation on. If a batch is `mix`, it is a special case of `new` where
    first 2500 sequences should be sampled at random from that batch. Otherwise a batch is NaN and not part of training.

    Returns
    -------
    dataset_df: the subset of df that is True for full_data
    prev_idx: the Index of dataset_df that is part of the old data, if there is anything.
    cv_df: the subset of dataset_df that is not True for prev_data -- what we will use for CV.
    """
    batch_name_col = "data_batch_name"
    dataset_df = df.dropna(axis=0, subset=batch_name_col)

    batches = grouping_info.index.values
    old_batches = batches[grouping_info == "old"]
    if len(old_batches) > 0:
        prev_idx = dataset_df[
            dataset_df[batch_name_col].str.contains("|".join(old_batches))
        ].index
    else:
        prev_idx = pd.Index([])
    # New batch should only be denoted by new or mix, no combination of the two. No other values allowed.
    if grouping_info.str.contains("new").sum() > 0:
        new_batches = batches[grouping_info == "new"]
        cv_df = dataset_df[
            dataset_df[batch_name_col].str.contains("|".join(new_batches))
        ]
    elif grouping_info.str.contains("mix").sum() > 0:
        new_batches = batches[grouping_info == "mix"]
        n = 2500
        seed = 0
        subsample_idx = pd.Index([])
        for batch in new_batches:
            # Get sequences in the batch
            batch_df = dataset_df[dataset_df[batch_name_col].str.contains(batch)]
            # Get a sample
            batch_idx = batch_df.sample(n=n, random_state=seed).index
            subsample_idx = subsample_idx.append(batch_idx)
        cv_df = dataset_df.loc[subsample_idx]
    else:
        raise ValueError("No batches were annotated with `new` or `mix`!")

    return dataset_df, prev_idx, cv_df


def load_cnn(checkpoint, model=None, seq_len=164, eval_mode=True):
    """
    Load in a fit model given a path to a checkpoint with its saved state dictionary. Assumes the model is a
    ..selene_files.enhancer_resnet_regression.EnhancerResnet instance with 164 bp input, but other model classes can
    be loaded as long as an instance is provided as input.

    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint.
    model : EnhancerResnet
        If specified, the model is already instantiated and it just needs to be initialized. If None, create a new
        instance of the object.
    seq_len : int
        Length of the sequence.
    eval_mode : bool
        If True, set model to eval mode. If False, it is in training mode.

    Returns
    -------
    model : EnhancerResnet
        The model with its fit parameters.
    """
    from selene_sdk.utils import load_model_from_state_dict
    checkpoint = torch.load(checkpoint, map_location=lambda storage, location: storage)
    if model is None:
        PATH = os.path.dirname(os.path.abspath(__file__))
        PATH = os.path.join(PATH, "../")
        from selene_files.enhancer_resnet_regression import EnhancerResnet
        model = EnhancerResnet(seq_len)

    model = load_model_from_state_dict(checkpoint["state_dict"], model)
    if eval_mode:
        model.eval()
        
    return model
