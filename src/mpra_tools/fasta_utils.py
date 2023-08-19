"""Functions for reading, writing, manipulating, and searching FASTA files and other DNA sequence operations.

"""

import itertools

import numpy as np
import pandas as pd


def fasta_iter(fin, sep=""):
    """A generator function to parse through one entry in a FASTA or FASTA-like file.

    Parameters
    ----------
    fin : file input stream
        Handle to the file to parse
    sep : str
        Delimiter for adjacent bases in the file

    Yields
    -------
    header : str
        Name of the sequence
    sequence : str
        The sequence
    """
    # Generator yields True if on a header line
    generator = itertools.groupby(fin, lambda x: len(x) > 0 and x[0] == ">")
    for _, header in generator:
        # Syntactic sugar to get the header string
        header = list(header)[0].strip()[1:]
        # Get all the lines for this sequence and concatenate together
        sequence = sep.join(i.strip() for i in generator.__next__()[1])
        yield header, sequence


def read_fasta(filename):
    """Parse through a FASTA file and store the sequences as a Series.

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    seq_series : pd.Series, dtype=str
        Index is the FASTA header, values are the sequence strings.
    """
    seq_series = {}
    with open(filename) as fin:
        # Add each sequence to the series
        for header, sequence in fasta_iter(fin):
            sequence = sequence.upper()
            seq_series[header] = sequence
    
    seq_series = pd.Series(seq_series)
    seq_series.index.name = "label"
    return seq_series


def write_fasta(fasta_ser, filename):
    """Write the given series to a file in FASTA format.

    Parameters
    ----------
    fasta_ser : pd.Series
        Index is the FASAT header, values are the sequence strings.
    filename : str
        Name of the file to write to.

    Returns
    -------
    None
    """
    with open(filename, "w") as fout:
        for header, seq in fasta_ser.items():
            fout.write(f">{header}\n{seq}\n")


def rev_comp(seq):
    """Take the reverse compliment of a sequence

    Parameters
    ----------
    seq : str
        The original sequence.

    Returns
    -------
    new_seq : str
        The reverse compliment.
    """
    compliment = {"A": "T", "C": "G", "G": "C", "T": "A"}
    new_seq = seq[::-1]
    new_seq = "".join([compliment[i] for i in new_seq])
    return new_seq


def gc_content(fasta_ser):
    """Calculate the GC content of every sequence.

    Parameters
    ----------
    fasta_ser : pd.Series
        Series representation of a FASTA file.

    Returns
    -------
    gc_ser : pd.Series
        GC content of every sequence, index matching fasta_ser.
    """
    gc_ser = fasta_ser.str.count("G|C")
    gc_ser /= fasta_ser.str.len()
    gc_ser.name = "GC_content"
    return gc_ser


def has_restriction_sites(fasta_ser, restrictions):
    """Generate a boolean mask indicating which sequences contain restriction sites.

    Parameters
    ----------
    fasta_ser : pd.Series
        Series representation of a FASTA file.
    restrictions : list-like
        List of strings indicating restriction sites to search for.

    Returns
    -------
    mask : pd.Series
        Boolean mask for fasta_ser indicating which sequences have restriction sites.
    """
    pattern = "|".join(restrictions)
    mask = fasta_ser.str.contains(pattern)
    return mask
