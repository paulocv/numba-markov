import numpy as np

from .types import np_p_t


class ExecData:
    """
    Bunch class to hold structures that are essential for the execution of a Markov chain multiplex model.
    The data defined here can (and should!) be reused along different executions of
    a given model in a given graph.

    This class is also responsible for the allocation of its members via 'alloc'
    method. It is smart so to avoid unnecessary reallocation after successive calls
    for alloc.

    The allocation signature of each member is defined at _alloc dictionary.

    Devnote: Python 3.7 alternative = dataclasses
    """
    __slots__ = ["p_state", "p_next", "f_trans", "q_trans", "aux_i"]

    # Dictionary of allocation callables for execution data structures.
    #    Must be updated if a new structure is defined in this class.
    _alloc = {
        "p_state": lambda dg, model: np.empty((model.num_states, len(dg)), dtype=np_p_t),
        "p_next": lambda dg, model: np.empty((model.num_states, len(dg)), dtype=np_p_t),
        "f_trans": lambda dg, model: np.empty((model.num_infec_trans, len(dg)), dtype=np_p_t),
        "q_trans": lambda dg, model: np.empty((model.num_infec_trans, len(dg)), dtype=np_p_t),
        "aux_i": lambda dg, model: np.empty((len(dg),), dtype=np_p_t) if model.require_aux else None,
    }

    def __init__(self, p_state=None, p_next=None, f_trans=None, q_trans=None, aux_i=None):
        self.p_state = p_state
        self.p_next = p_next
        self.f_trans = f_trans
        self.q_trans = q_trans
        self.aux_i = aux_i

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
