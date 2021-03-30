"""Allocator class to store possible results from a Markov chain simulation."""


import numpy as np
from .types import np_p_t, np_steps_t


class SimResults:
    """
    Bunch class to hold results from various Markov calculation routines.
    Contains both simple immutable variables and large mutable data structures.
    For the latter, a smart allocation procedure is employed.

    This class is also responsible for the allocation of its members via 'alloc'
    method. It is smart so to avoid unnecessary reallocation after successive calls
    for alloc.

    The allocation signature of each member is defined at _alloc dictionary.

    Devnote: Python 3.7 alternative = dataclasses
    """
    __slots__ = ["num_steps", "converged"]
    # num_steps, converged,

    # Dictionary of allocation callables for execution data structures.
    #    Must be updated if a new structure is defined in this class.
    _alloc = {
        # "p_state": lambda dg, model: np.empty((model.num_states, len(dg)), dtype=np_p_t),
        # "p_next": lambda dg, model: np.empty((model.num_states, len(dg)), dtype=np_p_t),
        # "f_trans": lambda dg, model: np.empty((model.num_infec_trans, len(dg)), dtype=np_p_t),
        # "q_trans": lambda dg, model: np.empty((model.num_infec_trans, len(dg)), dtype=np_p_t),
    }

    def __init__(self, num_steps=None, converged=None):
        """
        Creates a SimResults bunch to aggregate usual results from different types of Markov chain calculations.

        Parameters
        ----------
        num_steps : int
            Total number of steps a simulation had.
        converged : bool
            True if the simulation ended before max_steps.
        """
        self.num_steps = num_steps
        self.converged = converged
        # Future: self.p_state_tseries or something alike

    def alloc(self, name, dg, model):
        """Smart allocation for a structure given by name.
        Allocation only occurs if the required attribute is None.
        """
        if self.__getattribute__(name) is None:
            if name not in self._alloc:
                raise KeyError("Hey, allocation method for '{}' was not defined."
                               "".format(name))
            self.__setattr__(name, self._alloc[name](dg, model))

    def alloc_for(self, names, dg, model):
        """Smart allocation for a set of structures given as 'names'.
        Allocation only occurs if the attribute is None.
        """
        for name in names:
            self.alloc(name, dg, model)

    def assert_alloc(self, name):
        """Checks if a given object is not set to None, raising an error if not."""
        if self.__getattribute__(name) is None:
            raise ValueError("Hey, the execution data structure '{}' is not "
                             "allocated.".format(name))

    def assert_alloc_for(self, names):
        """Checks if a set of objects are not set to None, raising an error if so."""
        for name in names:
            self.assert_alloc(name)