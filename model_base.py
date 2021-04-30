"""
A simple generic data bunch to hold model parameters in a single object.

Author: Paulo Cesar Ventura da Silva (https://github.com/paulocv)

devnote - Things that could further improve CPU (or RAM) performance:
    * Calculation of (1 - q) instead of q
    * Calculation of q can be incorporated on the main iteration loop (using the "gather" implementation).
        - This prevents the creation of q-factor arrays.
    * Check if f_trans (with numpy notation) is generating large intermediate arrays (or if numba workarrounds this).
"""

import numba as nb
import numpy as np
import os
# import time

from .exec_data import ExecData
from .graph import NLayerMultiplex
from .sim_results import SimResults
from .types import nb_p_t, nb_ncount_t, awk_adjlist_t, nb_int_t, nb_float_t
from .utils import str_to_list_json, str_to_bool_safe, make_counter, calc_num_counts_from_iter


NUMBA_PARALLEL = str_to_bool_safe(os.getenv("NUMBA_PARALLEL", default=True))  # Reads from environment variable


# # This can't change the parameter sent to numba decorators after this module is imported.
# def set_numba_parallel(val: bool):
#     """
#     Sets the global variable NUMBA_PARALLEL, which controls if the numbaed that can be parallelized should actually
#     be. Call to this must be before any call to the numbaed functions, which do the JIT.
#     """
#     global NUMBA_PARALLEL
#     NUMBA_PARALLEL = val


class ModelBase:
    """Generic class to be used as a data bunch of model parameters and attributes.

    Any instance should override all class variables with values of the implemented model.

    Class variables
    ---------------
    num_states : int
        Number of states to be used as dynamical variables. In this formulation (the AB notation), it is the total
        number of possible states according to the model.
    num_layers : int
        Number of layers that the model should expect from a multiplex object. Even if the model can run with less
        layers or even single-layer graphs, just repeat those that should be equal.
    state_names : list
        A list with the names of the dynamical states. Should have num_states entries.
    state_id : dict
        Reverse container for state_names. Converts state names into their respective indexes.
    prevalences : list
        A list with the state combinations that should be recognized as "prevalences".
        For instance, in a model with two diseases and states SS, SI, IS and II, one could use:
        prevalences = {("I1", ["IS", "II"]), ("I2", ["SI", "II"])}
        Meaning that the prevalence called I1 should collect states IS and II.
    num_infec_trans : int
        Number of state changes that occur via contact with individuals in one or more states.
    """

    num_states = None  # Number states used for the dynamics (not necessarily the dimension of the dyn. system)
    num_layers = None  # Number of layers expected from a multiplex object.
    state_names = []   # Labels of each dynamical states. Must have len() = num_states.
    state_id = {}      # A dictionary to convert from state name to its index
    prevalences = []   # List of tuples (PrevName, [States])
    num_infec_trans = None  # Number of transitions that occur via contact
    require_aux = False  # If the model needs an auxiliary iarray

    # methods = {}  # Update methods, to be defined by the children classes.
    # std_method = None

    __slots__ = []  # Override this on children classes to include names of allowed attributes.

    # -----------------------------
    # Generic functions that need implementation by children classes
    # -----------------------------

    def initialize_states(self, pop, exd, init_mode, init_data):
        """Initializes the state probabilities of all nodes, based on information given as strings via
        init_mode and init_data.
        """
        raise NotImplementedError

    def iterate_model(self, pop, exd, dt=1.):
        """
        Generic model iteration function.
        Must be implemented with the model rules to calculate the next-step Markov probabilities given the previous
        ones.
        """
        raise NotImplementedError

    # --------------------------
    # Concrete execution functions, which should work for any model
    # --------------------------
    def check_states_are_close(self, pop, exd, tol):
        """
        Check if all markov probabilities from the current and the (already calculated) next step are all close
        up to a given tolerance tol. The operation is executed state-wise and nodewise. Any entry that exeeds the
        tolerance implies a 'False' return.
        """
        return check_states_are_close(pop.size, self.num_states, exd.p_state, exd.p_next, tol)

    @staticmethod
    def renormalize_node_probabilites(pop, exd):
        """
        Calculates the sum of the state probabilities of a node, then divide them by the result. This keeps
        the probabilities adding up to one, fixing locally for numerical errors.
        """
        return renormalize_node_probabilities(pop.size, exd.p_array)

    def _check_and_init_pop(self, pop, exd, initialize_pop, init_mode, init_data):
        if initialize_pop:
            if None not in (init_mode, init_data):
                self.initialize_states(pop, exd, init_mode, init_data)
            else:
                raise ValueError("Hey, both 'init_mode' and 'init_data' must be informed. To simulate without "
                                 "initializing the state probabilities, use 'initialize_pop = False'.")

    # @staticmethod
    # def get_allloc_list(**kwargs):
    #     # Basic containers (for any kind of Markov simulation)
    #     l = ["p_state", "p_next", "f_trans", "q_trans", "aux_i"]
    #
    #     # Additional containers
    #     if kwargs.get("calc_tseries_state", False):
    #         l.append("")

    def prepare_sim_results_obj(self, tbuffer_size, pop_size, res=None, **kwargs):
        """
        Parameters
        ----------
        tbuffer_size : int
        pop_size : int
        res : SimResults

        Keyword arguments
        -----------------
        calc_tseries_state : bool
        calc_tseries_nodestate : bool
        """
        if res is None:
            res = SimResults()

        # Time series of overall state densities
        if kwargs.get("calc_tseries_state", False):
            res.alloc_t_tseries(tbuffer_size)
            res.alloc_rho_tseries(self, tbuffer_size)

        # Future: time series of nodewise state probabilities

        res.converged = True  # Set to false after for loop exhaustion

        return res

    @staticmethod
    def store_to_time_series(exd, res,  t, calc_tseries_state, calc_tseries_nodestate):
        """TODO DOCS PLEASE

        Parameters
        ----------
        exd : ExecData
        res : SimResults
        t : float
        calc_tseries_state : bool
        calc_tseries_nodestate : bool
        """
        # Stores current time
        if res.t_tseries is not None:
            res.t_tseries[res.tbuffer_index] = t

        if calc_tseries_state:
            # Mean p_state over the nodes
            res.rho_tseries[res.tbuffer_index, :] = np.mean(exd.p_state, axis=1)

        res.tbuffer_index += 1

    def calc_stationary_densities(self, pop, exd, max_steps, init_mode=None, init_data=None, initialize_pop=True,
                                  dt=1., tol=1.E-6, persist_steps=5, error_check_period=25, check_convergence=True,
                                  calc_tseries_state=False, calc_tseries_nodestate=False,
                                  store_mode="linear", store_period=10, tbuffer_size=None,
                                  res=None):
        """
        Employs the Microscopic Markov Chain Approach to calculate the stationary probabilities of a model via
        simple fixed-point iterations of the model.

        The population (a multiplex network) is informed as 'pop', while 'exd' must be an ExecData instance to
        hold the necessary data structures for the calculations. The 'exd' structures do not need
        to be preallocated.

        The stationary probabilities of the last step will be stored in 'exd.p_state' array. Other results, such
        as the number of iteration steps and if the process ended before 'max_steps', are returned as a SimResults
        object.

        Parameters
        ----------
        pop : NLayerMultiplex
            Initialized multiplex object instance.
        exd : ExecData
            Execution data bunch, with containers not necessarily allocated.
        max_steps : int
            Maximum number of Markov iterations. If convergence is not detected after this number of steps,
            the result is returned anyway, but a warning is exhibited.
        init_mode : str
            Initialization mode string, passed to self.init_states.
        init_data : str or something else
            Initialization data, passed to self.init_states.
        initialize_pop : bool
            Whether the routine should initialize the population (i.e., the state probabilities) before calculations.
            If False, init_data and init_mode are ignored.
        dt : float
            Fixed time step length. Multiplies the dynamic probabilities of the model.
        tol : float
            Convergence tolerance. All states in all nodes must not differ from this amount.
        persist_steps : int
            Number of consecutive steps in which the convergence criterion must be met to declare overall convergence.
        error_check_period : int
            Number of time steps between numerical error checks (i.e., node probability renormalizations).
            Set to None to avoid error checking.
        res : SimResults
            If informed, it is used as the bunch for simulation results.
            Otherwise, a new one is created.

        Return
        ------
        res : SimResults
        """
        # ------------------------------------------
        # PREAMBLE OF THE SIMULATION
        if error_check_period is None:
            # Never checks for numerical errors
            error_check_period = max_steps + 2

        # Execution structure allocation (if not performed yet)
        exd.alloc_for(["p_state", "p_next", "f_trans", "q_trans", "aux_i"], pop, self)

        # State probability initializations
        self._check_and_init_pop(pop, exd, initialize_pop, init_mode, init_data)
        exd.p_next[:] = exd.p_state

        # Time series periodicity definition
        store_counter = make_counter(store_mode, store_period)()
        i_t_store = next(store_counter)
        if tbuffer_size is None:
            # Infers from the counter object (creates a dummy instance and transverse it).
            tbuffer_size = calc_num_counts_from_iter(make_counter(store_mode, store_period)(), max_steps)

        # Bunch to aggregate results of the simulation. It is returned by the function.
        res = self.prepare_sim_results_obj(tbuffer_size, pop.size, res,
                                           calc_tseries_state=calc_tseries_state,
                                           calc_tseries_nodestate=calc_tseries_nodestate)

        # ------------------------------------------------
        # MAIN SIMULATION LOOP
        persist_count = i_t = 0  # Number of consecutive steps declared as convergent / Regular time index
        res.tbuffer_index = 0  # Index of time in the time series, which is no greater than i_t due to periodic storing.
        t = 0.
        for i_t, t in enumerate(np.arange(0., max_steps*dt, dt)):
            # Periodic time series storing
            if i_t >= i_t_store:
                i_t_store = next(store_counter)
                self.store_to_time_series(exd, res, t,
                                          calc_tseries_state=calc_tseries_state,
                                          calc_tseries_nodestate=calc_tseries_nodestate)

            # ------  MODEL UPDATE HERE  ----------
            self.iterate_model(pop, exd, dt=dt)

            # Periodic numerical error check
            if (i_t + 1) % error_check_period == 0:
                renormalize_node_probabilities(pop.size, exd.p_next)
                # self.renormalize_node_probabilites(pop, exd)  # devnote: Overhead?

            # Persistent convergence check (if requested)
            if check_convergence:
                if self.check_states_are_close(pop, exd, tol):
                    persist_count += 1
                    if persist_count == persist_steps:
                        # Stop by convergence
                        break
                else:
                    persist_count = 0

            # Update (consolidation) of changes for the next step
            exd.p_state[:] = exd.p_next[:]

        # -----------------------------------
        # END OF SIMULATION
        else:
            # Enters here if max_steps is reached
            res.converged = False

        # Updates everything (including the last step to the time series) and returns.
        exd.p_state[:] = exd.p_next[:]
        self.store_to_time_series(exd, res, t, calc_tseries_state=calc_tseries_state,
                                  calc_tseries_nodestate=calc_tseries_nodestate)
        res.num_steps = i_t + 1
        return res

    # - - SNAPSHOT FROM BEFORE TIME SERIES HANDLING - TO BE DELETED
    def calc_stationary_densities_legacy(self, pop, exd, max_steps, init_mode=None, init_data=None, initialize_pop=True,
                                  dt=1., tol=1.E-6, persist_steps=5, error_check_period=25,
                                  res=None, **kwargs):
        """
        Employs the Microscopic Markov Chain Approach to calculate the stationary probabilities of a model via
        simple fixed-point iterations of the model.

        The population (a multiplex network) is informed as 'pop', while 'exd' must be an ExecData instance to
        hold the necessary data structures for the calculations. The 'exd' structures do not need
        to be preallocated.

        The stationary probabilities of the last step will be stored in 'exd.p_state' array. Other results, such
        as the number of iteration steps and if the process ended before 'max_steps', are returned as a SimResults
        object.

        Parameters
        ----------
        pop : NLayerMultiplex
            Initialized multiplex object instance.
        exd : ExecData
            Execution data bunch, with containers not necessarily allocated.
        max_steps : int
            Maximum number of Markov iterations. If convergence is not detected after this number of steps,
            the result is returned anyway, but a warning is exhibited.
        init_mode : str
            Initialization mode string, passed to self.init_states.
        init_data : str or something else
            Initialization data, passed to self.init_states.
        initialize_pop : bool
            Whether the routine should initialize the population (i.e., the state probabilities) before calculations.
            If False, init_data and init_mode are ignored.
        dt : float
            Fixed time step length. Multiplies the dynamic probabilities of the model.
        tol : float
            Convergence tolerance. All states in all nodes must not differ from this amount.
        persist_steps : int
            Number of consecutive steps in which the convergence criterion must be met to declare overall convergence.
        error_check_period : int
            Number of time steps between numerical error checks (i.e., node probability renormalizations).
            Set to None to avoid error checking.
        res : SimResults
            If informed, it is used as the bunch for simulation results.
            Otherwise, a new one is created.

        Return
        ------
        res : SimResults
        """
        DeprecationWarning("Hey, legacy function (with no time series handling) will be deleted.")

        if error_check_period is None:
            # Never checks for numerical errors
            error_check_period = max_steps + 2

        # Execution structure allocation (if not performed yet)
        exd.alloc_for(["p_state", "p_next", "f_trans", "q_trans", "aux_i"], pop, self)

        # Bunch to aggregate results of the simulation. It is returned by the function.
        if res is None:
            res = SimResults()
        res.converged = True  # Set to false after for loop exhaustion

        # State probability initializations
        self._check_and_init_pop(pop, exd, initialize_pop, init_mode, init_data)
        exd.p_next[:] = exd.p_state

        # Main loop
        persist_count = 0  # Number of consecutive steps declared as convergent
        for i_t, t in enumerate(np.arange(0., max_steps*dt, dt)):
            # MODEL UPDATE HERE
            self.iterate_model(pop, exd, dt=dt)

            # Periodic numerical error check
            if (i_t + 1) % error_check_period == 0:
                renormalize_node_probabilities(pop.size, exd.p_next)
                # self.renormalize_node_probabilites(pop, exd)  # devnote: Overhead?

            # Persistent convergence check
            if self.check_states_are_close(pop, exd, tol):
                persist_count += 1
                if persist_count == persist_steps:
                    # Updates things and returns results
                    exd.p_state[:] = exd.p_next[:]
                    res.num_steps = i_t + 1
                    return res
            else:
                # Persist count reset to zero again.
                persist_count = 0

            # Update (consolidation) of changes for the next step
            exd.p_state[:] = exd.p_next[:]

        # At this point, it means that convergence was not achieved during desired number of steps.
        res.converged = False
        res.num_steps = max_steps
        return res

    # def _timed_calc_stationary_densities(self, pop, exd, max_steps, init_mode=None, init_data=None, initialize_pop=True,
    #                                      dt=1., tol=1.E-6,
    #                                      persist_steps=5, error_check_period=25, res=None):
    #     """
    #     Copy of self.calc_stationary_densities with timed core operations and an execution time report at the end.
    #     """
    #     if error_check_period is None:
    #         # Never checks for numerical errors
    #         error_check_period = max_steps + 2
    #
    #     # Execution structure allocation (if not performed yet)
    #     exd.alloc_for(["p_state", "p_next", "f_trans", "q_trans", "aux_i"], pop, self)
    #
    #     # Bunch to aggregate results of the simulation. It is returned by the function.
    #     if res is None:
    #         res = SimResults()
    #     res.converged = True  # Set to false after for loop exhaustion
    #
    #     # State probability initializations
    #     self._check_and_init_pop(pop, exd, initialize_pop, init_mode, init_data)
    #     exd.p_next[:] = exd.p_state
    #
    #     # Main loop
    #
    #     # Execution time counters
    #     iter_xt = 0.
    #     numerr_xt = 0.
    #     convcheck_xt = 0.
    #     consolid_xt = 0.
    #
    #     def _report_xt():
    #         print("Execution times (sum during total loop):\n"
    #               "----------\n"
    #               "Model iteration: \t{:0.5f} s\n"
    #               "Numerical error check: {:0.5f} s\n"
    #               "Convergence check:\t{:0.5f} s\n"
    #               "Consolidation:\t{:0.5f}\n"
    #               "-------------\n".format(iter_xt, numerr_xt, convcheck_xt, consolid_xt)
    #               )
    #
    #     persist_count = 0  # Number of consecutive steps declared as convergent
    #     for i_t, t in enumerate(np.arange(0., max_steps*dt, dt)):
    #         # MODEL UPDATE HERE
    #         xt0 = time.time()
    #         self.iterate_model(pop, exd, dt=dt)
    #         iter_xt += time.time() - xt0
    #
    #         # Periodic numerical error check
    #         xt0 = time.time()
    #         if (i_t + 1) % error_check_period == 0:
    #             renormalize_node_probabilities(pop.size, exd.p_next)
    #             # self.renormalize_node_probabilites(pop, exd)  # devnote: Overhead?
    #         numerr_xt += time.time() - xt0
    #
    #         # Persistent convergence check
    #         xt0 = time.time()
    #         if self.check_states_are_close(pop, exd, tol):
    #             persist_count += 1
    #             if persist_count == persist_steps:
    #                 # Updates things and returns results
    #                 exd.p_state[:] = exd.p_next[:]
    #                 res.num_steps = i_t + 1
    #                 _report_xt()
    #                 return res
    #         else:
    #             # Persist count reset to zero again.
    #             persist_count = 0
    #         convcheck_xt += time.time() - xt0
    #
    #         # Update (consolidation) of changes for the next step
    #         xt0 = time.time()
    #         exd.p_state[:] = exd.p_next[:]
    #         consolid_xt += time.time() - xt0
    #
    #     # At this point, it means that convergence was not achieved during desired number of steps.
    #     res.converged = False
    #     res.num_steps = max_steps
    #     _report_xt()
    #     return res


# ----------------------------------------------------------------
# SAMPLE FUNCTIONS TO INITIALIZE MARKOV CHAIN STATES
# ----------------------------------------------------------------
def initialize_states_basic(p_array, num_nodes, init_mode, init_data, i_healthy_state, i_infective_state):
    """
    Initialize state probabilities for a single infective state.

    Changes are applied to p_array in place, so no returns.

    Parameters
    ----------
    p_array : np.ndarray
    num_nodes : int
    init_mode : str
        Initialization mode string. Supported modes:

        * "fixed_frac", "frac", "p", "fixed_p":
            All nodes are initialized with equal probability of being on the infective state.
            The init_data must be a float or float-like string with the probability of being infected at start.
        * "fixed_count", "count", "nodecount":
            All nodes are initialized with equal probability of being infective, but the input data is in
            number of nodes (i.e., the expectancy of the number of infectious nodes). Then p = num / pop_size.
            The init_data must be a float or float-like string with the number of infectious nodes.
        *  "nodelist", "infec_nodelist":
            A give list of nodes is initialized with 100% probability of being infective, with all other nodes
            having 100% of being susceptible.
            The init_data must be a list or a string that's parseable to a list via json.loads. Entries are the
            integer ids of the infective nodes.

    init_data : str or something else
        Data used to initialize states. Must follow the directives as explained for init_mode.
    i_healthy_state : int
        Index of the healthy state, according to p_array convention (model.states).
    i_infective_state : int
        Index of the infective state, according to p_array convention (model.states).
    """

    # Starts by seting all p to zero, regardless of the mode
    p_array[:] = 0.

    # Initialization modes elif ladder
    if init_mode in ["fixed_frac", "frac", "p", "fixed_p"]:
        # Initializes all nodes with uniform probability of being infected
        p = float(init_data)
        p_array[i_infective_state, :] = p
        p_array[i_healthy_state, :] = 1. - p

    elif init_mode in ["fixed_count", "count", "nodecount"]:
        # Same as fixed_frac (previous), but by providing the expectancy of the number of infected nodes
        p = float(init_data) / num_nodes
        p_array[i_infective_state, :] = p
        p_array[i_healthy_state, :] = 1. - p

    elif init_mode in ["nodelist", "infec_nodelist"]:
        # Specifies a list of nodes to be initialized as infectious with probability 1
        if isinstance(init_data, str):
            infec_nodes = str_to_list_json(init_data)
        else:
            infec_nodes = init_data

        # First sets all nodes as susceptibles
        p_array[i_healthy_state] = 1.0

        # For the chosen nodes, set them as infected
        p_array[i_infective_state, infec_nodes] = 1.
        p_array[i_healthy_state, infec_nodes] = 0.

    else:
        # Unrecognized init mode
        raise ValueError("Hey, mode '{}' to initialize the "
                         "state probabilities was not recognized. "
                         "Please check 'initialize_states_basic()' "
                         "documentation.".format(init_mode))


# ----------------------------------------------------------------
# GENERIC FUNCTIONS FOR MARKOV CHAIN CALCULATIONS
# ----------------------------------------------------------------

@nb.njit(nb.void(nb_ncount_t, nb_p_t[:], nb_p_t, nb_p_t[:]), parallel=NUMBA_PARALLEL)
def calc_f_trans(num_nodes, p_state_s, tp, f_trans_tr):
    """
    Calculates each node's f-factor for a given transition tr, with individual probability tp per contact.
    The f-factor of i is the probability that node i does not cause a state transition to a "susceptible" neighbor.

    Results are written on f_trans_tr.
    """
    f_trans_tr[:] = 1. - tp * p_state_s  # One-line numpy vectorized implementation
    # for i in nb.prange(num_nodes):
    #     f_trans_tr[i] = 1. - tp * p_state_s[i]


@nb.njit(nb.void(nb_ncount_t, nb_p_t[:, :], nb_p_t, nb_p_t[:]), parallel=NUMBA_PARALLEL)
def calc_f_trans_statelist(num_nodes, p_state_sub, tp, f_trans_tr):
    """
    Calculates each node's f-factor for a given transition tr, with individual probability tp per contact, in which
    more than one of the basic model states can cause the transition.
    The f-factor of i is the probability that node i does not cause a state transition to a "susceptible" neighbor.
    Parameter p_state_sub is assumed to be a 2D array with signature: p[state, i], and must only contain the states
    that cause the given transition.

    Results are written on f_trans_tr.
    """
    f_trans_tr[:] = 1. - tp * np.sum(p_state_sub, axis=0)


# # Do not parallelize this function/implementation, as it writes by random access.
# # For larger networks (say, > 100k), it may be slower than the parallelized "reversed loop" version.
# @nb.njit(nb.void(nb_ncount_t, awk_adjlist_t, nb_p_t[:], nb_p_t[:]))
# def calc_q_trans(num_nodes, g_neighbors, f_trans_tr, q_trans_tr):
#     """
#     Calculates each node's q-factor for a given transition re, based on the precalculated f-factors.
#     The q-factor of node i is the probability that it will undergo a given infection transition, given that
#     it is prone (i.e., susceptible) to it, and based on each neigbor's probability of not-promoting the transition.
#
#     The f-factors are expected to be calculated (an up-to-date) on f_trans_tr (array of shape (num_nodes,)).
#     The resulting q factors are written in q_trans_tr.
#
#     ** Broadcast implementation **
#     In this version of the function, nodes are visited to broadcast their f-factors to their neighbors, which update
#     their q-factors. Due to this random-access writing, its main loop cannot be parallelized. But it is slightly
#     faster than its counterpart method for sequential execution.
#
#     Parameters
#     ----------
#     num_nodes : int
#         Number of nodes on each layer of the multiplex.
#     g_neighbors : ak.Array
#         Adjacency list (i.e., list of neighbors) as an awkward array.
#     f_trans_tr : np.ndarray
#         1D array with the precalculated f-factors of the given transition. Signature: f_trans_tr[i]
#     q_trans_tr : np.ndarray
#         1D array that receives the q-factors as they're calculated. Signature: q_trans_tr[i]"""
#     # Initialize all q-factors as 1
#     q_trans_tr[:] = 1.
#
#     for i in range(num_nodes):
#         f_i = f_trans_tr[i]  # Gets the f-factor of the current node
#         for j in g_neighbors[i]:
#             # Broadcasts the f-factor of node i to all its neighbors
#             q_trans_tr[j] *= f_i
#
#     # Finally reverts, as q-factor is defined as the probability of _not_ being infected by any neighbor.
#     q_trans_tr[:] = 1. - q_trans_tr[:]


@nb.njit(nb.void(nb_ncount_t, awk_adjlist_t, nb_p_t[:], nb_p_t[:]), parallel=NUMBA_PARALLEL)
def calc_q_trans(num_nodes, g_neighbors, f_trans_tr, q_trans_tr):
    """
    Calculates each node's q-factor for a given transition re, based on the precalculated f-factors.
    The q-factor of node i is the probability that it will undergo a given infection transition, given that
    it is prone (i.e., susceptible) to it, and based on each neigbor's probability of not-promoting the transition.

    The f-factors are expected to be calculated (an up-to-date) on f_trans_tr (array of shape (num_nodes,)).
    The resulting q factors are written in q_trans_tr.

    ** Gather implementation **
    In this version of the function, nodes are visited to calculate their own q-factor at once, by collecting its
    neighbors' f-factors. This allows the for loop to be parallelized.

    Parameters
    ----------
    num_nodes : int
        Number of nodes on each layer of the multiplex.
    g_neighbors : ak.Array
        Adjacency list (i.e., list of neighbors) as an awkward array.
    f_trans_tr : np.ndarray
        1D array with the precalculated f-factors of the given transition. Signature: f_trans_tr[i]
    q_trans_tr : np.ndarray
        1D array that receives the q-factors as they're calculated. Signature: q_trans_tr[i]
    """
    # q_trans_tr[:] = 1.  # It is faster to set inside the loop

    for i in nb.prange(num_nodes):
        q_trans_tr[i] = 1.
        for j in g_neighbors[i]:
            # Catches the f-factor from all its neighbors
            q_trans_tr[i] *= f_trans_tr[j]
        # Inverts about 1, to take the _infection_ probability
        q_trans_tr[i] = 1. - q_trans_tr[i]


# ---------------
# Convergence (consecutive states close to each other) and numerical error checking

# # Convergence check - numpy version
# def check_states_are_close(p_state, p_next, tol):
#     return np.amax(np.abs((p_next - p_state))) < tol

# Convergence check - numba version
@nb.njit(nb.bool_(nb_ncount_t, nb_int_t, nb_p_t[:, :], nb_p_t[:, :], nb_float_t))
def check_states_are_close(num_nodes, num_states, p_state, p_next, tol):
    """
    Check if all markov state probabilities from the current and the (already calculated) next step are all close
    up to a given tolerance tol. The operation is executed for all nodes and all states. Any entry that exceeds the
    tolerance immediately implies a 'False' return.
    """
    for i_s in range(num_states):
        for i in range(num_nodes):
            if abs(p_next[i_s, i] - p_state[i_s, i]) >= tol:
                return False

    return True


@nb.njit(nb.void(nb_ncount_t, nb_p_t[:, :]))
def renormalize_node_probabilities(num_nodes, p_array):
    """
    Calculates the sum of the state probabilities of a node, then divide them by the result. This keeps
    the probabilities adding up to one, locally fixing numerical errors.
    """
    for i in range(num_nodes):
        p_array_i = p_array[:, i]
        p_sum = p_array_i.sum()
        p_array_i /= p_sum


# -------------------------------
# MISC - Other functions related to Markov modeling
@nb.njit((nb_p_t, nb_p_t))
def abo_event_probs(p1, p2):
    """Probabilities for two events that cannot occur simultaneously.

    Context:
      Two events (1 and 2), if executed independently, have probabilities
      p1 and p2 of success; if they are run simultaneously, however,
      no more than one of the events can succeed (i.e., the two events
      cannot occur at the same trial).

    The random process:
      First, a random choice defines if one of the events will
      occur; with probability (1-p1)*(1-p2), none of them succeed, and
      with complementary probability another random choice is made
      between the two events. The second random choice has renormalized
      probability: with chance p1/(p1+p2), the event 1 occur, and
      with chance p2/(p1+p2), event 2 suceeds.

    Parameters
    ----------
    p1 : float
        Individual probability of event 1.
    p2 : float
        Individual probability of event 2.

    Returns
    -------
    p(0), p(A), p(B)
        Probabilities that neither occur, event 1 occurs or event 2
        occurs, respectively.
    """
    p0 = (1. - p1) * (1. - p2)  # Probability that neither happens
    renorm = (1. - p0) / (p1 + p2)  # Renorm. for knowing that A or B happens
    return p0, renorm * p1, renorm * p2
