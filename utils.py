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
    both true and false bools, and raises an error if the string does not match any case. Also, if s is already a
    boolean, returns as it is.

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


def cast_to_export(value, float_fmt="{:12.6f}", int_fmt="{:12d}"):
    """
    Converts a given variable to a string in an adequate format for tabular files.
    """
    if isinstance(value, (float, np.floating)):
        out = float_fmt.format(value)
    elif isinstance(value, (int, np.integer)):
        out = int_fmt.format(value)
    else:
        out = str(value)
    return out


def cast_to_export_list(values, float_fmt="{:12.6f}", int_fmt="{:12d}", sep="\t") -> str:
    """
    Converts a list of values into strings to be exported as a fixed width string, using cast_to_export in each value.
    Adds a sep character after each entry.
    """
    out_str = ""
    for value in values:
        out_str += cast_to_export(value, float_fmt, int_fmt) + sep
    return out_str


def list_to_csv(parlist, sep=", "):
    """Returns a csv string with elements from a list."""
    result_str = ""
    for par in parlist:
        result_str += str(par) + sep
    return result_str[:-len(sep)]

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


# ----------------------------------
# MISC

def make_counter(mode, param):
    """
    Returns a generator that produces numbers in a given sequence.
    For this project, it is used to get the steps at which time series is stored.

    In all modes, the first returned value is always zero. Return type is always int.

    Supported modes
    ---------------
    "linear", "lin"
        Arithmetic progression - sums 'param' at each call.
    "exp", "exponential"
        Geometric progression - starting from 1, multiplies by 'param' at each call

    """
    param = float(param)

    if mode in ("linear", "lin"):
        # LINEAR COUNTER (Arithmetic Progression)
        def gen():
            i = 0.
            while True:
                yield round(i)
                i += param

    elif mode in ("exp", "exponential"):
        # EXP COUNTER (Geometric Progression)
        def gen():
            yield 0
            i = 1.
            while True:
                yield round(i)
                i *= param
    else:
        raise ValueError("Hey, mode '{}' passed to 'make_store_counter' was not understood.".format(mode))

    return gen


def calc_num_counts_from_iter(gen, max_steps, upperlim=10000):
    """
    Calculates the number of times a generator gen() is called until its return value is greater (or equal) than
    max_steps.

    The integer upperlim avoids an infinite loop due to some error.
    """
    for i in range(upperlim):
        if next(gen) >= max_steps:
            return i + 1
    else:
        raise RuntimeError("Hey, tried to measure the length of this generator but it did not reach max_steps after "
                           "{:d} calls.".format(upperlim))
