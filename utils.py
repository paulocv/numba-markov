"""
Miscellaneous auxiliary functions for the numba_markov package.
Some may be taken from my "toolbox" folder.
"""

import json
import numba as nb
import numpy as np
import pandas as pd


MACH_EPSILON = np.finfo(float).eps


def str_to_list_json(s):
    """Converts a string to a python list using json.loads().
    Will only work for simple lists with objects supported by json.

    s : str
    Returns : list
    """
    s.replace("'", '"')
    return json.loads(s)


def load_edgl(fname):
    """Uses pandas to load an edgelist file and returns it as a list of tuples with pairs of connected nodes."""
    # Reads edges
    df = pd.read_csv(fname, sep=" ", header=None, usecols=[0, 1])
    # Convert to list of tuples
    return list(df.itertuples(index=False, name=None))


def guess_num_nodes_from(edgelist):
    """
    Gets the greatest node index 'maxi' from an edgelist. Returns 1 + maxi as an estimate of the network size.
    """
    raise NotImplementedError


def nav_ipr(x):
    """Returns the inverse participation ratio of a vector (first normalized by its sum of squares)."""
    x2 = x*x
    s2 = np.sum(x2)
    if s2 < MACH_EPSILON:
        # Zero sum. Could happen for veeery small overall prevalence.
        return 0.
    else:
        return np.sum(x2 * x2 / (s2 * s2))


def bla_ipr(x):
    """Returns the inverse participation ratio of a vector (first normalized by its sum of squares)."""
    phi = x / np.sqrt(np.sum(x**2))
    return np.sum(phi**4)
    # if s2 < MACH_EPSILON:
    #     # Zero sum. Could happen for veeery small overall prevalence.
    #     return 0.
    # else:
    #     return np.sum(x2 * x2 / (s2 * s2))
