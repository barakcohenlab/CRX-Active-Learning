"""Useful functions for machine learning modeling and to preprocess data as necessary for modeling.

"""
import random
import numpy as np
import scipy.io as scio
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import selene_sdk


def subset_cross_validation(cv_labels, fixed_idx, nfolds, seed=0):
    """
    Generator to perform cross-validation on a defined subset of the data. Each fold is joined with a fixed set of items. If nfolds is 1, then it just returns the full input index as the first value.

    Parameters
    ----------
    cv_labels : pd.Series
        Index should be a unique integer value from a larger pandas object, values are interger class labels. These
        are the items which should be cross-validated.
    fixed_idx : pd.Index
        The unique indices for the data that is held constant in each fold.
    nfolds : int
        Number of folds for CV.
    seed : int
        Seed for the cross-validation object.

    Yields
    -------
    full_subset_idx : pd.Index
        The unique integer indices of the cross-validation fold and the fixed data.
    fake_val_idx : pd.Index
        The unique integer indices for the validation fold. This is unlikely to be used.
    """
    if nfolds == 1:
        for i in range(1):
            yield fixed_idx.append(cv_labels.index), None
    
    else:
        cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
        for subset_idx, fake_val_idx in cv.split(cv_labels, cv_labels):
            subset_idx = cv_labels.iloc[subset_idx].index
            full_subset_idx = fixed_idx.append(subset_idx)
            yield full_subset_idx, fake_val_idx


def subset_square_gram(gram, idx):
    """Get a submatrix of a symmetric (square) Gram matrix given an array (not an Index) of indices"""
    return gram[idx][:, idx]


def one_hot_encode(seqs):
    """Given a list of sequences, one-hot encode them.

    Parameters
    ----------
    seqs : list-like
        Each entry is a DNA sequence to be one-hot encoded

    Returns
    -------
    seqs_hot : ndarray, shape (number of sequences, 4, length of sequence)
    """
    seqs_hot = list()
    for seq in seqs:
        seqs_hot.append(
            selene_sdk.sequences.Genome.sequence_to_encoding(seq).T
        )
    seqs_hot = np.stack(seqs_hot)
    return seqs_hot


def prepare_data_for_selene(seqs, activity, activity_key, filename):
    """Given a list of sequences and their corresponding labels, one-hot encode the sequences and write the data to a
    .mat file for Selene.

    Parameters
    ----------
    seqs : pd.Series
        FASTA of DNA sequences to be used as input. Assumes all sequences are the same length.
    activity : pd.Series
        Labels corresponding to the activity of each sequence. Can be either quantitative measurements or numerical
        representation of categorical groups.
    activity_key : str
        Key to assign the activity labels in the .mat file.
    filename : str
        Name of the file to write.

    Returns
    -------
    None
    """
    seqs_hot = one_hot_encode(seqs)
    # Make the labels a column vector
    activity = activity.values[:, np.newaxis]
    # Write to file
    data = {}
    for k, v in [("sequence", seqs_hot), (activity_key, activity)]:
        data[k] = v
    scio.savemat(filename, data)


def seed_selene(seed):
    """Seed all the random number generators Selene utilizes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cnn_predict_batches(model, data_in_batches, use_cuda=False, seq_len=164):
    """
    Use a trained CNN to make predictions on a dataset, split into batches.

    Parameters
    ----------
    model : nn.Module
        The trained CNN
    data_in_batches : list(tuple(ndarray, ndarray)) or list(ndarray)
        Each entry of the list is a batch of the data. If the entry is a tuple, then the first array is the input
        sequences and the second are the associated labels. If the entry is an array, then it is just the sequences
        and no labels are provided. Each ndarray is formatted as <batch size> x 4 (number of channels) x 164 (seq len)
    use_cuda : bool
        Indicates whether to use a GPU or not. Do this when there are a lot of sequences to make predictions over.
    seq_len : int
        Expected size of sequence, should be on the last axis

    Returns
    -------
    preds : ndarray
        The predictions made by the CNN. Shape is <total number of sequences> x <number of outputs being predicted>
    """
    preds = []
    # Figure out whether there are associated labels or not. It only matters for how we loop over the data.
    if type(data_in_batches[0]) is tuple:
        unpack = lambda x: x[0]
    elif type(data_in_batches[0]) is np.ndarray:
        unpack = lambda x: x
    else:
        raise ValueError("Did not recognize format of input data.")

    with torch.no_grad():
        for inputs in data_in_batches:
            inputs = unpack(inputs)
            inputs = torch.Tensor(inputs)
            if use_cuda:
                inputs = inputs.cuda()
            if inputs.shape[1] == seq_len:
                inputs = inputs.transpose(1, 2)
            batch_preds = model(inputs)
            preds.append(batch_preds.data.cpu().numpy())
    return np.vstack(preds)


def batch(seqs, batch_size=128):
    """Given a list of one-hot encoded sequences, split them into batches for prediction.

    Parameters
    ----------
    seqs : ndarray, shape (number of sequences, 4, length of sequence)
        List of sequences to batch.
    batch_size : int
        The batch size to use. Defaults to 128.

    Returns
    -------
    batched : list
        Each value of the list is a (batch_size, 4, length of sequence) array.
    """
    batched = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]
    return batched