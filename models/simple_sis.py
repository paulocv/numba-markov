"""
Microscopic Markov chain SIS model.

Author: Paulo Cesar Ventura da Silva (https://github.com/paulocv)

"""

import numba as nb
# import numpy as np

from ..model_base import ModelBase, calc_f_trans, calc_q_trans, initialize_states_basic, NUMBA_PARALLEL
from ..types import nb_ncount_t, awk_adjlist_t, nb_p_t, nb_float_t


class SimpleSIS(ModelBase):

    # Child class (model-specific) attributes
    num_states = 2
    num_layers = 1
    state_names = ["S", "I"]
    state_id = {s: i for i, s in enumerate(state_names)}
    prevalences = []  # [("I", "I")]  # Maybe not needed for SIS
    num_infec_trans = 1  # Only S->I transition

    __slots__ = ["beta", "mu"]

    def __init__(self, beta, mu):
        """Creates an instance of a simple SIS model, carrying its parameters:

        Parameters
        ----------
        beta : float
            Probability of contagion for each S-I contact at each time step.
        mu : float
            Probability of healing, for each I node at each time step.
        """
        self.beta = beta
        self.mu = mu

    def initialize_states(self, pop, exd, init_mode, init_data):
        """Initializes state probabilities using model_base.initialize_states_basic function."""
        return initialize_states_basic(exd.p_state, len(pop), init_mode, init_data,
                                       i_healthy_state=0, i_infective_state=1)

    def iterate_model(self, pop, exd, dt=1.):
        """
        Applies SIS model rules to calculate the next-step state probabilities given the current ones
        """
        return _iterate_model(pop.size, pop.g[0].neighbors_awk, self.beta, self.mu, dt, exd.p_state,
                              exd.f_trans, exd.q_trans, exd.p_next)


# SIS model time step function.
# Here is where the model's rules are applied.
@nb.njit(nb.void(nb_ncount_t, awk_adjlist_t, nb_p_t, nb_p_t, nb_float_t, nb_p_t[:, :], nb_p_t[:, :],
                 nb_p_t[:, :], nb_p_t[:, :]), parallel=NUMBA_PARALLEL)
def _iterate_model(num_nodes, g_neighbors, beta, mu, dt, p_state, f_trans, q_trans, p_next):

    # Aliases for better readability
    p_state_s = p_state[0]
    p_state_i = p_state[1]
    p_next_s = p_next[0]
    p_next_i = p_next[1]
    f_trans_inf = f_trans[0]
    q_trans_inf = q_trans[0]

    # Time step factor
    beta *= dt
    mu *= dt

    # Aliases that might avoid some repeated calculations
    om_mu = 1. - mu

    # Calculate f and q factors for disease infection transition
    calc_f_trans(num_nodes, p_state_i, beta, f_trans_inf)
    calc_q_trans(num_nodes, g_neighbors, f_trans_inf, q_trans_inf)

    # Model rules applied to calculate next probabilities
    for i in nb.prange(num_nodes):
        # Aliases for the node properties, to avoid repeated indexing operation.
        p_s = p_state_s[i]
        p_i = p_state_i[i]
        q_inf = q_trans_inf[i]

        # SIS Markov equations here
        p_next_s[i] = p_s * (1. - q_inf) + p_i * mu
        p_next_i[i] = p_s * q_inf + p_i * om_mu

