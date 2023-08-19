"""Functions to compute saliency maps with the Majdandzic correction OR to run 
DeepLIFT on a sequence to get the hypothetical and actual importance scores OR."""
import numpy as np
import pandas as pd
import torch

from deeplift.dinuc_shuffle import dinuc_shuffle
import shap

from . import modeling

SEQ_LENGTH = 164


def dinuc_background(input_seq, seed=0, nshuf=100):
    """
    Dinucleotide shuffle a one-hot encoded sequence several times to generate a
    background distribution with shap. Note that the dimensionality of the input
    sequence will be shape (4, seq_len) but the function to shuffle expects the
    sequence to be shape (seq_len, 4). Internally, the input sequence gets
    transposed, then the shuffled sequences get transposed back.

    Parameters
    ----------
    input_seq : list[Tensor] or None
        If a list, it should be of length one. The sole value should be a
        Tensor with shape (4, seq_len) representing the one-hot encoded
        sequence to shuffle. This is a list of length one for compatibility
        reasons with shap.
    seed : int
        Seed for the numpy RNG. Defaults to 0.
    nshuf : int
        Number of unique shuffles to generate. Defaults to 100.

    Returns
    -------
    result : list[Tensor]
        The dinucleotide shuffled sequences, wrapped within a list of size 1 for
        compatibility reasons. The Tensor will be shape (nshuffle, 4, seq_len).
        If the input_seq was None, the tensor is all zeros and the first
        dimension is size 1.
    """
    # Case for initializing the explainer
    if input_seq is None:
        return torch.zeros((1, 4, SEQ_LENGTH))
    assert len(input_seq) == 1
    onehot_seq = input_seq[0].transpose(0, 1).numpy()
    rng = np.random.RandomState(seed)
    result = torch.Tensor(dinuc_shuffle(
        onehot_seq,
        num_shufs=nshuf,
        rng=rng
    )).transpose(1, 2)
    return [result]


def hypothetical_importance_scores(mult, orig_inp, bg_data):
    """
    Get the hypothetical contribution scores of a sequence against the provided
    background given the DeepLIFT multipliers. Note that all arguments are lists
    of length equal to the number of input modes, which in our case is only one.
    Slide 27: https://docs.google.com/presentation/d/1JCLMTW7ppA3Oaz9YA2ldDgx8ItW9XHASXM1B3regxPw/edit#slide=id.g65244ca07a_0_273

    Parameters
    ----------
    mult : list[np.array]
        The DeepLIFT multipliers for each of the background sequences. Shape is
        (n_bg, 4, seq_len)
    orig_inp : list[np.array]
        One-hot encoding of the original input sequence, shape (4, seq_len).
        Only used for compatibility reasons.
    bg_data : list[np.array]
        One-hot encoding of the background distribution of sequences, shape
        (n_bg, 4, seq_len).

    Returns
    -------
    attr_map : list[np.array]
        The Majdandzic-corrected contribution of every base at every position, averaged
        across all background sequences.
    """
    assert len(orig_inp) == 1
    assert len(orig_inp[0].shape) == 2
    # Shape starts out as (n_bg, 4, seq_len) corresponding to the hypothetical
    # contribution of all bases against all background sequences
    attr_map = np.zeros_like(bg_data[0]).astype("float")
    # For each base...
    for nt in range(orig_inp[0].shape[0]):
        # Generate a dummy sequence where all of one base is hot
        hyp_imp = np.zeros_like(orig_inp[0]).astype("float")
        hyp_imp[nt] = 1.0
        # Take difference from reference. Hypothetical imp gets broadcasted to
        # do the subtraction, so we end with a 3D array
        hyp_diff_from_ref = hyp_imp - bg_data[0]
        # Get importance scores
        hyp_contribs = hyp_diff_from_ref * mult[0]
        # FIXME reverted for thesis
        # Sum importance scores across all bases at a position
        attr_map[:, nt] = np.sum(hyp_contribs, axis=1)
        # Correct for the off-simplex noise according to Majdandzic et al.
        # hyp_contribs -= np.mean(hyp_contribs, axis=1, keepdims=True)
        # Get the contribution to this base
        # attr_map[:, nt] = hyp_contribs[:, nt]

    # Average across all background sequences
    attr_map = np.mean(attr_map, axis=0)
    return [attr_map]


def actual_importance_scores(hypothetical_imp, onehot_seq):
    """Helper function to get the actual importance score for a sequence, given
    the hypothetical importance score relative to background. This is done by
    simply multiplying the one-hot encoded sequence element-wise by the
    hypothetical scores."""
    return hypothetical_imp * onehot_seq


def score_seqs(seqs, model, seed=0, nshuf=100):
    """
    Given a list of DNA sequences, one-hot encode it, then generate both the
    hypothetical and actual importance scores for each sequence.

    Parameters
    ----------
    seqs : list[str] or str
        The DNA sequences of interest. Each sequence is assumed to be of size
        SEQ_LENGTH. A single sequence can be provided as a string, in which case the
        outputs will have 2 dimensions instead of 3.
    model : torch.Module
        An instance of the trained model.
    seed : int
        Seed for random number generator for generating the background. Defaults to 0.
    nshuf : int
        Number of background sequences to generate. Defaults to 100.

    Returns
    -------
    hypotheticals : ndarray
        Hypothetical importance score for each sequence.
    actuals : ndarray
        Actual importance score for each sequence.
    preds : ndarray or float
        Model output prediction for each sequence.
    """
    one_seq = type(seqs) is str
    if one_seq:
        seqs = [seqs]
    seqs_hot = modeling.one_hot_encode(seqs)
    explainer = shap.DeepExplainer(
        model,
        data=lambda x: dinuc_background(x, seed=seed, nshuf=nshuf),
        combine_mult_and_diffref=hypothetical_importance_scores
    )
    seqs_tensor = torch.Tensor(seqs_hot)
    hypotheticals = explainer.shap_values(seqs_tensor)
    actuals = np.array([
        actual_importance_scores(hyp, seq)
        for hyp, seq in zip(hypotheticals, seqs_hot)
    ])
    preds = model(seqs_tensor).detach().numpy()
    if one_seq:
        hypotheticals = hypotheticals[0]
        actuals = actuals[0]
        # Assuming only one prediction task
        preds = preds[0][0]
    return hypotheticals, actuals, preds


def saliency_map(seq, model):
    """
    Given a single DNA sequence, one-hot encode it, then generate both the
    hypothetical and actual importance score using the Majdandzic correction.

    Parameters
    ----------
    seq : str
        The DNA sequence of interest, assumed to be of size SEQ_LENGTH.
    model : torch.Module
        An instance of the trained model.

    Returns
    -------
    hypothetical : ndarray
        Hypothetical importance score.
    actual : ndarray
        Actual importance score (saliency map).
    pred : float
        Model output prediction.
    """
    seq_hot = modeling.one_hot_encode([seq])
    seq_tensor = torch.Tensor(seq_hot).requires_grad_()
    pred = model(seq_tensor)
    hypothetical = torch.autograd.grad(pred, seq_tensor)[0]
    hypothetical = hypothetical.data.cpu().numpy()
    # Correction step
    hypothetical -= np.mean(hypothetical, axis=1, keepdims=True)
    actual = actual_importance_scores(hypothetical, seq_hot)
    
    hypothetical = hypothetical[0]
    actual = actual[0]
    # Assuming only one prediction task
    pred = pred.detach().numpy()[0][0]
    
    return hypothetical, actual, pred
