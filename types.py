"""Defines numeric types for use in this package, unifying these through
the models.
"""
import awkward1 as ak
import numpy as np
import numba as nb

# numpy types
np_int_t = np.int64  # General use int type
np_float_t = np.float64  # General use float type
np_ncount_t = np.int64  # Node indexing and counting
np_p_t = np.float64  # Float type for probabilities
np_steps_t = np.int64  # Count of simulation steps

# Dev remainder: Change these numba types along with the above types
# Numba does not define short/long
nb_int_t = nb.int64  # General use int type
nb_float_t = nb.float64  # General use float type
nb_ncount_t = nb.int64  # Node indexing and counting
nb_p_t = nb.float64  # Float type for probabilities
nb_steps_t = nb.int64  # Count of simulation steps

# Awkward array: Foolish way to define the numba type of an adjacency list as an awkward array (using a dummy array).
awk_adjlist_t = ak.Array([[np_ncount_t(0), np_ncount_t(1)], [np_ncount_t(2)]]).numba_type
