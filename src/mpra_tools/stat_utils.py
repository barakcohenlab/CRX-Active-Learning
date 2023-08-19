"""Functions to aid in statistical analysis.

"""
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def _get_lognormal_params(mean, std):
    """Helper function for lognormal_params.

    Parameters
    ----------
    mean : float
        The sample mean.
    std : float
        The sample standard deviation.

    Returns
    -------
    pd.Series
        The first and second moments of the lognormal distribution.

    """
    if mean == 0:
        mu = - np.inf
        sigma = np.nan
    else:
        cov = std / mean
        mu = np.log(mean / np.sqrt(cov**2 + 1))
        sigma = np.sqrt(np.log(cov**2 + 1))

    return pd.Series({
        "mu": mu,
        "sigma": sigma
    })


def lognormal_params(params):
    """Given the mean and standard deviation of an MPRA measurement in linear space, compute the parameters of the
    log-normal distribution based on the following definitions:
    mean = exp(mu + sigma**2 / 2)
    variance = [exp(sigma**2 -1] * exp(2 * mu + sigma**2)

    Parameters
    ----------
    params : tuple (float, float) or pd.DataFrame
        The sample mean and standard deviation, or a set of them. If a DataFrame, rows are samples. First value is
        the mean, second value is the std.

    Returns
    -------
    pd.Series or pd.DataFrame
        The first and second moments of the lognormal distribution. If mean and std are a float, returns a Series. If
        they are Series, return a DataFrame
    """
    dtype = type(params)
    if dtype is tuple:
        return _get_lognormal_params(*params)
    elif dtype is pd.DataFrame:
        return params.apply(lambda x: _get_lognormal_params(*x), axis=1)
    else:
        raise ValueError(f"Did not recognize input type, got {dtype}.")


def fdr(pvals):
    """Correct for multiple hypotheses using Benjamini-Hochberg FDR and return q-values for each observation. Ties
    are assigned the largest possible rank.

    Parameters
    ----------
    pvals : pd.Series
        A set of p-value from many statistical tests.

    Returns
    -------
    qvals : pd.Series
        The FDR-corrected q-values.
    """
    n = pvals.notna().sum()
    ranks = pvals.rank(method="max")
    qvals = pvals * n / ranks
    return qvals


def variation(data, ddof=1, pseudocount=0):
    """Calculate the mean and coefficient of variation from the given data. Optionally add a
    pseudocount beforehand to avoid divide by zero errors. Sample CV is calculated by default.

    Parameters
    ----------
    data : pd.Series
        A list of replicate measurements.
    ddof : int
        Degrees of freedom. If one (default), calculates the sample CV. If zero, calculates the
        population CV.
    pseudocount : float
        Optional pseudocount to add to all observations before calculating CV, to avoid divide by
        zero errors. Defaults to 0.

    Returns
    -------
    pd.Series
        The mean, CV, and n
    """
    adjusted = data + pseudocount
    mean = adjusted.mean()
    cv = adjusted.std(ddof=ddof) / mean
    n = data.notna().sum()
    return pd.Series({
        "mean": mean,
        "cv": cv,
        "count": n,
    })
