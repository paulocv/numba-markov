"""
Interacting diseases via susceptibility change, with two SIS models.
Multiplex Markov chain approach

Author: Paulo Cesar Ventura da Silva (https://github.com/paulocv)

"""
import numba as nb
import numpy as np

from ..exec_data import ExecData
from ..graph import NLayerMultiplex
from ..model_base import ModelBase, calc_f_trans, calc_q_trans, initialize_states_basic, abo_event_probs, PARALLEL_NUMBA
from ..types import nb_ncount_t, awk_adjlist_t, nb_p_t, nb_float_t, np_p_t, np_float_t


class DoubleSIS(ModelBase):
    """ TODO DOCS

    Indexes of the 4 model states
    -----------------------------
    These indexes refer to the first axis of p_state, p_next

    * [0] | SS
    * [1] | IS
    * [2] | SI
    * [3] | II

    Indexes of the 4 model transitions
    ----------------------------------
    These indexes regard the first axis of f_trans and q_trans arrays.

    * [0] | i1 | SS -> IS  = infection of a fully susceptible (SS) by disease 1.
    * [1] | i2 | SS -> SI  = infection of a fully susceptible (SS) by disease 2.
    * [2] | i1m | SI -> II = infection of a disease-2 infected (IS) by disease 1.
    * [3] | i2m | IS -> II = infection of a disease-1 infected (IS) by disease 2.
    """

    # Child class attributes
    num_states = 4
    state_names = ["SS", "IS", "SI", "II"]
    state_id = {s: i for i, s in enumerate(state_names)}
    prevalences = [("I1", ["IS", "II"]), ("I2", ["SI", "II"])]  # Named prevalences
    num_infec_trans = 4  # SS -> IS, SS -> SI, IS -> II, SI -> II
    require_aux = True  # Requires an auxiliary array of node probabilities

    __slots__ = ["beta1", "beta2", "mu1", "mu2", "gamma1", "gamma2",
                 "om_mu1", "om_mu2", "gamma_beta1", "gamma_beta2"]

    def __init__(self, beta1, beta2, mu1, mu2, gamma1, gamma2):
        """TODO docs

        DUE TO AUXILIARY DEPENDENT PARAMETERS, do not reuse a model instance with different parameters.
        """
        # Input parameters
        # TODO: perform type checks to prevent awful numba errors later when something wrong is passed
        self.beta1 = np_p_t(beta1)
        self.beta2 = np_p_t(beta2)
        self.mu1 = np_p_t(mu1)
        self.mu2 = np_p_t(mu2)
        self.gamma1 = np_float_t(gamma1)
        self.gamma2 = np_float_t(gamma2)

        # Useful (but not necessarily used) dependent parameters
        self.om_mu1 = 1. - mu1
        self.om_mu2 = 1. - mu2
        self.gamma_beta1 = gamma1 * beta1
        self.gamma_beta2 = gamma2 * beta2

    def initialize_states(self, pop, exd, init_mode, init_data):
        """
        Initializes state probabilities using model_base.initialize_states_basic function.
        II is regarded as the infectious state, SS is the susceptible one.
        """
        return initialize_states_basic(exd.p_state, len(pop), init_mode, init_data,
                                       i_healthy_state=0, i_infective_state=3)

    def iterate_model(self, pop, exd, dt=1.):
        """
        Applies one time step (iteration) of the model's Markov chain method.

        Parameters
        ----------
        pop : NLayerMultiplex
        exd : ExecData
        dt : float
        """
        return _iterate_model(pop.size, pop.g[0].neighbors_awk, pop.g[1].neighbors_awk,
                              self.beta1, self.beta2, self.mu1, self.mu2, self.gamma_beta1, self.gamma_beta2,
                              dt, exd.p_state, exd.f_trans, exd.q_trans, exd.aux_i, exd.p_next)


@nb.njit(nb.void(nb_ncount_t, awk_adjlist_t, awk_adjlist_t, nb_p_t, nb_p_t, nb_p_t, nb_p_t, nb_p_t, nb_p_t,
                 nb_float_t, nb_p_t[:, :], nb_p_t[:, :], nb_p_t[:, :], nb_p_t[:], nb_p_t[:, :]),
         parallel=PARALLEL_NUMBA)
def _iterate_model(num_nodes, g1_neighbors, g2_neighbors, beta1, beta2, mu1, mu2, gamma_beta1, gamma_beta2,
                   dt, p_state, f_trans, q_trans, aux_i, p_next):
    """Double SIS model iteration (compiled version)."""

    # ------------------------------------
    # Aliases for better readability of the core code
    p_state_ss = p_state[0]
    p_state_is = p_state[1]
    p_state_si = p_state[2]
    p_state_ii = p_state[3]
    p_next_ss = p_next[0]
    p_next_is = p_next[1]
    p_next_si = p_next[2]
    p_next_ii = p_next[3]

    f_trans_i1 = f_trans[0]   # SS -> IS
    f_trans_i2 = f_trans[1]   # SS -> SI
    f_trans_i1m = f_trans[2]  # SI -> II
    f_trans_i2m = f_trans[3]  # IS -> II

    q_trans_i1 = q_trans[0]
    q_trans_i2 = q_trans[1]
    q_trans_i1m = q_trans[2]
    q_trans_i2m = q_trans[3]

    # --------------------------------
    # Time step factor application
    beta1 *= dt
    beta2 *= dt
    mu1 *= dt
    mu2 *= dt
    gamma_beta1 *= dt
    gamma_beta2 *= dt

    # -------------------------
    # f and q-factors calculation (probabilities of transitions)
    # Disease 1 transitions f-factors
    np.add(p_state[1], p_state[3], aux_i)  # Auxiliary calc of sum of states IS + II = I1
    calc_f_trans(num_nodes, aux_i, beta1, f_trans_i1)  # SS -> IS f-factors
    calc_f_trans(num_nodes, aux_i, gamma_beta1, f_trans_i1m)  # SS -> IS f-factors

    # Disease 2 transitions f-factors
    np.add(p_state[2], p_state[3], aux_i)  # Auxiliary calc of sum of states SI + II = I2
    calc_f_trans(num_nodes, aux_i, beta2, f_trans_i2)  # SS -> IS f-factors
    calc_f_trans(num_nodes, aux_i, gamma_beta2, f_trans_i2m)  # SS -> IS f-factors

    # devnote: this fails due to the use of list indexing of an array.
    # calc_f_trans_statelist(num_nodes, p_state[[1, 3]], beta1, f_trans_i1)  # Sums IS + II
    # calc_f_trans_statelist(num_nodes, p_state[[1, 3]], gamma_beta1, f_trans_i1m)  # Sums IS + II
    # calc_f_trans_statelist(num_nodes, p_state[[2, 3]], beta2, f_trans_i2)  # Sums SI + II
    # calc_f_trans_statelist(num_nodes, p_state[[2, 3]], gamma_beta2, f_trans_i2m)  # Sums SI + II

    # Diseases 1 and 2 q-factors from f-factors
    calc_q_trans(num_nodes, g1_neighbors, f_trans_i1, q_trans_i1)
    calc_q_trans(num_nodes, g2_neighbors, f_trans_i2, q_trans_i2)
    calc_q_trans(num_nodes, g1_neighbors, f_trans_i1m, q_trans_i1m)
    calc_q_trans(num_nodes, g2_neighbors, f_trans_i2m, q_trans_i2m)

    # ------------------------
    # Main loop over the nodes
    for i in nb.prange(num_nodes):
        # Aliases of node i variables
        p_ss = p_state_ss[i]
        p_is = p_state_is[i]
        p_si = p_state_si[i]
        p_ii = p_state_ii[i]

        # ABO probabilities for the pairs of events starting from each state
        # Notation: [s]_[tr] -> A transition tr leaving state s
        ss_o, ss_i1, ss_i2 = abo_event_probs(q_trans_i1[i], q_trans_i2[i])  # SS: Infec by 1 or infec by 2
        is_o, is_i2m, is_h1 = abo_event_probs(q_trans_i2m[i], mu1)          # IS: InfecMod by 2 or heal from 1
        si_o, si_i1m, si_h2 = abo_event_probs(q_trans_i1m[i], mu2)          # IS: InfecMod by 1 or heal from 2
        ii_o, ii_h1, ii_h2 = abo_event_probs(mu1, mu2)                      # II: Heal from 1 or heal from 2

        # Model dynamical equations, after all. They're simplified with all these alias defined above.
        p_next_ss[i] = p_ss * ss_o + p_si * si_h2 + p_is * is_h1
        p_next_is[i] = p_ss * ss_i1 + p_is * is_o + p_ii * ii_h2
        p_next_si[i] = p_ss * ss_i2 + p_si * si_o + p_ii * ii_h1
        p_next_ii[i] = p_is * is_i2m + p_si * si_i1m + p_ii * ii_o
