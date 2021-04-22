"""
Miscellaneous auxiliary functions for the numba_markov package.
Some may be taken from my "toolbox" folder.
"""

import json
# import numba as nb
import numpy as np
import pandas as pd

from .types import np_ncount_t


MACH_EPSILON = np.finfo(float).eps


# ---------------------------------------
# OPERATIONS BETWEEN VARIABLE TYPES, ETC

def str_to_list_json(s):
    """Converts a string to a python list using json.loads().
    Will only work for simple lists with objects supported by json.

    s : str
    Returns : list
    """
    s.replace("'", '"')
    return json.loads(s)


def str_to_bool_safe(s, truelist=("True", "true", "T"), falselist=("False", "false", "F")):
    """
    Converts a boolean codified as a string. Instead of using 'eval', compares with lists of accepted strings for
    both true and false bools, and raises an error if the string does not match any case.

    Parameters
    ----------
    s : str
        The string to be read from
    truelist : tuple or list
        Tuple or list of strings interpreted as True.
    falselist : tuple or list
        Tuple or list of strings interpreted as False.

    Returns
    -------
    res : bool
    """
    if s in truelist:
        return True
    elif s in falselist:
        return False
    elif isinstance(s, bool):
        # In case the input is already a boolean.
        return s
    else:
        raise ValueError("Hey, the string '{}' could not be understood as a boolean.".format(s))


# ----------------------------------------
# NETWORK OPERATIONS
# TODO: FOR THE NEXT TWO FUNCTIONS, ADD FUNCTIONALITY TO READ THE NUMBER OF NODES AS THE FIRST ARGUMENT
def load_edgl(fname):
    """Uses pandas to load an edgelist file and returns it as a list of tuples with pairs of connected nodes."""
    # Reads edges
    df = pd.read_csv(fname, sep=" ", header=None, usecols=[0, 1])
    # Convert to list of tuples
    return list(df.itertuples(index=False, name=None))


def load_edgl_as_array(fname):
    """
    Uses pandas to load an edgelist file and returns it as a 2D array with pairs of connected nodes.
    Signature: a[i, 0] and a[i, 1] are two connected vertices.
    """
    df = pd.read_csv(fname, sep=" ", header=None, usecols=[0, 1])
    return df.to_numpy(dtype=np_ncount_t)


def guess_num_nodes_from(edgelist):
    """
    Gets the greatest node index 'maxi' from an edgelist. Returns 1 + maxi as an estimate of the network size.
    """
    return np.max(edgelist) + 1


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
