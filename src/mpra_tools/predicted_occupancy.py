"""Functions for computing the predicted occupancy of TFs across a sequence, as well as the information content.

"""
import re
import numpy as np
import pandas as pd
from scipy.special import gamma

from . import fasta_utils


def peek(fin):
    """ Peek at the next line in a file.

    Parameters
    ----------
    fin : file input stream

    Returns
    -------
    line : str
    """
    pos = fin.tell()
    line = fin.readline()
    fin.seek(pos)
    return line


def gobble(fin):
    """Gobble up lines in the file until we have reached the start of a motif or EOF.

    Parameters
    ----------
    fin : file input stream

    Returns
    -------
    lines : str
        The lines that got gobbled, including newline characters.
    """
    lines = ""
    while True:
        line = peek(fin)
        if len(line) == 0 or line[:5] == "MOTIF":
            break
        else:
            lines += fin.readline()

    return lines


def read_pwm_files(filename):
    """Given a MEME file, read in all PWMs. PWMs are stored as DataFrames, and the list of PWMs is represented as a
    Series, where keys are primary motif identifiers and values are the DataFrames.

    Parameters
    ----------
    filename : str
        Name of the file to read in.

    Returns
    -------
    pwm_ser : pd.Series
        The list of PWMs parsed from the file.
    """
    pwm_ser = {}
    with open(filename) as fin:
        # Lines before the first motif is encountered
        gobble(fin)

        # Do-while like behavior to read in the data
        # Do <read in motif> while not EOF
        while True:
            # MOTIF <motif name> [alternate ID]
            motif_id = fin.readline().split()[1]

            # Empty line
            fin.readline()
            # "letter-probability matrix: [other info]"
            fin.readline()

            # Every line that starts with a space, zero, or one is a new position in the PWM, if the first character is not a space
            # it is not part of the PWM.
            pwm = []
            pat = re.compile("[01 ]")
            while pat.search(peek(fin)[0]):
            # while peek(fin)[0] == " " or peek(fin)[0] == "0":
                pwm.append(fin.readline().split())

            # Make a DataFrame and add to the list
            pwm = pd.DataFrame(pwm, dtype=float, columns=["A", "C", "G", "T"])
            pwm_ser[motif_id] = pwm
            # Read up any extra info such as the URL
            gobble(fin)

            # Check if EOF
            if len(peek(fin)) == 0:
                break

    pwm_ser = pd.Series(pwm_ser)
    return pwm_ser


def ewm_from_letter_prob(pwm_df, pseudocount=0.0001, rt=2.5):
    """Compute an energy weight matrix from a letter probability matrix. Normalize the PWM to the maximum letter
    probability at each position and then compute relative free energies using the formula ddG = -RT ln(p_b,i / p_c,
    i), where p_b,i is the probability of base b, p_c,i is the probability of the consensus base, and ddG is relative
    free energy.

    Parameters
    ----------
    pwm_df : pd.DataFrame
        The letter probability matrix, where each row is a position of the motif and columns represent A, C, G, T.
    pseudocount : float
        Pseudocount value to add to every value to account for zeros in the PWM.
    rt : float
        The value of RT to use in the formula in kJ/mol.

    Returns
    -------
    ewm_df : pd.DataFrame
        The weight matrix of free energies relatives to the consensus sequence.
    """
    pwm_df = pwm_df.copy()
    pwm_df += pseudocount
    # Normalize each position by the most frequent letter to get relative Kd
    pwm_df = pwm_df.apply(lambda x: x / x.max(), axis=1)
    # Convert to EWM
    ewm_df = -rt * np.log(pwm_df)
    ewm_df.columns = ["A", "C", "G", "T"]
    return ewm_df


def ewm_to_dict(ewm):
    """Convert a DataFrame representation of an EWM to a dictionary for faster indexing.

    Parameters
    ----------
    ewm : pd.DataFrame

    Returns
    -------
    ewm_dict : {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are values of the matrix
    """
    ewm_dict = ewm.to_dict(orient="index")
    return ewm_dict


def read_pwm_to_ewm(filename, pseudocount=0.0001, rt=2.5):
    """Read in a file of letter probability matrices, convert them to EWMs, and then convert the DataFrames to
    dictionaries for faster indexting.

    Parameters
    ----------
    filename : str
        Name of the file to read in.
     pseudocount : float
        Pseudocount value to add to every value to account for zeros in the PWM.
    rt : float
        The value of RT to use in the formula in kJ/mol.

    Returns
    -------
    ewm_dict : {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are values of the matrix
    """
    # Wrapper function handle to convert each PWM to an EWM
    # Read in the file
    pwms = read_pwm_files(filename)
    # Convert to EWM dicts
    ewms = pwms.apply(ewm_from_letter_prob, args=(pseudocount, rt)).apply(ewm_to_dict)
    return ewms


def energy_landscape(seq, ewm):
    """Scans both strands of a sequence with energy matrix

    Parameters
    ----------
    seq : str
        The sequence to scan.
    ewm : dict {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are delta delta G relative to the consensus sequence.

    Returns
    -------
    fscores, rscores: np.array, dtype=float
        Represents the EWM scores for each subsequence on the forward and reverse strand.
    """
    motif_len = len(ewm.keys())
    # Number of positions where the motif can be scored
    n_scores = len(seq) - motif_len + 1
    fscores = np.zeros(n_scores)
    # Reverse compliment scores
    rscores = fscores.copy()
    r_seq = fasta_utils.rev_comp(seq)

    # Calculate occ for forward and reverse k-mer at every position
    for pos in range(n_scores):
        f_kmer = seq[pos:pos + motif_len]
        r_kmer = r_seq[pos:pos + motif_len]

        # Initialize energy score
        fscore = 0
        rscore = 0
        
        # This is faster than using the enumerate function
        # Calculate the EWM score for the k-mer starting at pos
        for i in range(motif_len):
            fscore += ewm[i][f_kmer[i]]
            rscore += ewm[i][r_kmer[i]]

        fscores[pos] = fscore
        rscores[pos] = rscore

    # rscores needs to be reversed so the indexing corresponds to the appropriate position in the original sequence (
    # i.e. just the compliment, not the reverse compliment)
    rscores = rscores[::-1]

    return fscores, rscores


def occupancy_landscape(seq, ewm, mu):
    """Compute the occupancy landscape by scanning sequence with the energy matrix and then calculate the relative
    free energy for each k-mer subsequence on the forward and reverse strand at chemical potential mu.

    Parameters
    ----------
    seq : str
        The sequence to scan.
    ewm : dict {int: {str: float}}
        Dictionary of dictionaries, where the outer keys are positions, the inner keys are letters, and the values
        are delta delta G relative to the consensus sequence.
    mu : int
        Chemical potential of the TF

    Returns
    -------
    fscores, rscores: np.array, dtype=float
        Represents the occupancy scores for each subsequence on the forward and reverse strand.
    """
    fscores, rscores = energy_landscape(seq, ewm)
    # Convert EWM scores to occupancies
    fscores = 1 / (1 + np.exp(fscores - mu))
    rscores = 1 / (1 + np.exp(rscores - mu))
    return fscores, rscores


def total_landscape(seq, ewms, mu):
    """Compute the occupancy landscape for each TF and join it all together into a DataFrame. Pad the ends of the
    positional information so every TF occupancy landscape is the same length.

    Parameters
    ----------
    seq : str
        The DNA sequence.
    ewms : pd.Series or dict {str: {int: {str: float}}}
        Keys/index are TF names and values are dictionary representations of the EWMs.
    mu : int or float
        TF chemical potential.

    Returns
    -------
    landscape : pd.DataFrame, dtype=float
        The occupancy of each TF at each position in each orientation. Rows are positions, columns are TFs and
        orientations, values indicate the predicted occupancy starting at the position.
    """
    landscape = {}
    seq_len = len(seq)
    # For each TF
    for name, ewm in ewms.items():
        # Get the predicted occupancy and add it to the list
        fscores, rscores = occupancy_landscape(seq, ewm, mu)
        landscape[f"{name}_F"] = fscores
        landscape[f"{name}_R"] = rscores

    # Pad the ends of the lists to the length of the sequence
    for key, val in landscape.items():
        amount_to_add = seq_len - len(val)
        landscape[key] = np.pad(val, (0, amount_to_add), mode="constant", constant_values=0)

    landscape = pd.DataFrame(landscape)
    return landscape


def total_occupancy(seq, ewms, mu):
    """For each TF, calculate its predicted occupancy over the sequence given the energy matrix and chemical
    potential. Then, summarize the information as the total occupancy of each TF over the entire sequence.

    Parameters
    ----------
    seq : str
        The DNA sequence.
    ewms : pd.Series or dict {str: {int: {str: float}}}
        Keys/index are TF names and values are dictionary representations of the EWMs.
    mu : int or float
        TF chemical potential.

    Returns
    -------
    occ_profile : pd.Series, dtype=float
        The total occupancy profile of each TF on the sequence.
    """
    occ_landscape = total_landscape(seq, ewms, mu)
    occ_profile = {}
    # Add together F and R strand
    if type(ewms) is dict:
        keys = ewms.keys()
    else:
        keys = ewms.index
    for tf in keys:
        occ_profile[tf] = occ_landscape[[f"{tf}_F", f"{tf}_R"]].sum().sum()

    occ_profile = pd.Series(occ_profile)
    return occ_profile


def all_seq_total_occupancy(seq_ser, ewm_ser, mu, convert_ewm=True):
    """Calculate the total predicted occupancy of each TF over each sequence.

    Parameters
    ----------
    seq_ser : pd.Series, dtype=str
        Representation of FASTA file, where each value is a different sequence. Index is the FASTA header.
    ewm_ser : pd.Series, dtype=pd.DataFrame
        Each value of the series is an energy matrix for a different TF.
    mu : int or float
        TF chemical potential.
    convert_ewm : bool
        If True, convert the EWMs from a DataFrame to a dictionary representation. If False, assumes that the EWMs
        have already been converted.

    Returns
    -------
    occ_df : pd.DataFrame, shape=[n_seq, n_tf]
        Total predicted occupancy of each TF over each sequence. Rows are sequences with same index as seq_ser,
        columns represent different TFs.
    """
    # Convert the EWMs to dictionary representations for speedups
    if convert_ewm:
        ewm_ser = {name: ewm_to_dict(ewm) for name, ewm in ewm_ser.iteritems()}

    seq_ser = seq_ser.str.upper()
    occ_df = seq_ser.apply(lambda x: total_occupancy(x, ewm_ser, mu))

    return occ_df


def information_content(occupancies, diversity_cutoff=0.5, log=np.log2):
    """Given a list of TF occupancies, compute total occupancy, diversity, and information content.

    Parameters
    ----------
    occupancies : pd.Series
        Predicted occupancy for a collection of TFs on a given sequence.
    diversity_cutoff : float
        Cutoff to call a TF "occupied" on the sequence.
    log : Function handle
        Function to use for computing the log. Default is log2 so information content is in bits, natural log should be used for
        biophysical applications.

    Returns
    -------
    result : pd.Series
        The total occupancy, diversity, and information content of the provided sequence.
    """
    # Calculate total occupancy of all TFs on the sequence
    total_occ = occupancies.sum()
    # Count how many of the TFs are occupied, i.e. have motifs present in the sequence
    diversity = (occupancies > diversity_cutoff).sum()
    # Since the occupancies are continuous values, we need to use the Gamma function to compute entropy. Gamma(n+1)=n!
    # W = N! / prod(N_i!)
    microstates = gamma(total_occ + 1) / (occupancies + 1).apply(gamma).product()
    # S = log W
    info_content = log(microstates)

    result = pd.Series({
        "total_occupancy": total_occ,
        "diversity": diversity,
        "info_content": info_content
    })
    return result


def get_occupied_sites_and_tfs(occupancy_df, cutoff=0.5):
    """Given an occupancy landscape for a sequence, identity the positions that are occupied and which TF+orientation is
    occupying that site. Assumes that at any given position, only one TF can be occupied.

    Parameters
    ----------
    occupancy_df : pd.DataFrame
        Occupancy landscape of a sequence. Rows are positions, columns are TFs+orientations, values are the predicted
        occupancy.
    cutoff : float
        Cutoff for calling something occupied.

    Returns
    -------
    sites_to_tf : {int: str}
        Mapping of starting positions to the TF+orientation occupied at that position
    """
    occupancy_df = occupancy_df > cutoff
    sites = occupancy_df.index[occupancy_df.any(axis=1)].values
    sites_to_tf = {i: occupancy_df.columns[occupancy_df.loc[i]].values[0] for i in sites}
    return sites_to_tf


def get_num_motifs(seq, ewms, mu, cutoff=0.5):
    """For a given sequence, get the number of motifs for each TF, rather than the total predicted occupancy of that TF.

    Parameters
    ----------
    seq : str
        The sequence of interest.
    ewms : pd.Series or dict {str: {int: {str: float}}}
        Keys/index are TF names and values are dictionary representations of the EWMs.
    mu : int or float
        TF chemical potential.
    cutoff : float
        Cutoff for calling something occupied.

    Returns
    -------
    occupied_sites : pd.Series
        Number of occupied sites for each TF.
    """
    occupied_sites = total_landscape(seq, ewms, mu) > cutoff
    # Count the number of occupied sites in each column
    occupied_sites = occupied_sites.sum()
    # Add together strand effects
    occupied_sites = occupied_sites.groupby(occupied_sites.index.str.split("_").str[0]).sum()
    return occupied_sites
