
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from numba_markov.graph import Layer, NLayerMultiplex, nx_to_layer
from numba_markov.exec_data import ExecData
from numba_markov.model_base import *
from numba_markov.models.simple_sis import *
from numba_markov.models.double_sis import *
from numba_markov.utils import load_edgl_as_array, guess_num_nodes_from
from toolbox.network_gen_tools import generate_layer

from numba_markov.model_base import calc_f_trans, calc_q_trans


# Pop paramenters
# N = int(1E4)  # int(5E5)
# kmean = 5.
# p = kmean / N

# Model parameters - SIS
beta1 = 0.0105  # 0.005
beta2 = 0.022

mu1 = 0.5
mu2 = 0.5

gamma1 = 0.8
gamma2 = 1.2

# Initial conditions?
init_mode = "p"  # "nodelist"
init_data = 0.1  # [0, 1, 2, 3]

# Simulation parameters
max_steps = 3000
tol = 1.E-6

# ETC
plot_histo = False

# ---------------


def load_edgl(fname):
    # Reads edges
    df = pd.read_csv(fname, sep=" ", header=None, usecols=[0, 1])
    # Convert to list of tuples
    return list(df.itertuples(index=False, name=None))


# Population creation
# print("Creating graphs with networkx...")
# g = nx.erdos_renyi_graph(N, p, seed=10)  # NETWORKX TAKES JUST TOO LONG
# # h = nx.erdos_renyi_graph(N, p, seed=20)
# gl = nx_to_layer(g)

print("Loading network edgelist from file")
# edges = load_edgl("networks/ER/ER_10000_k10_1.edgl")
# edges = load_edgl("networks/assortat_SF-CM/N10k/g2p00_k2_002.edgl")
# edges = load_edgl("networks/assortat_SF-CM/N100k/g2p00_k2_000.edgl")
# edges = load_edgl("networks/assortat_SF-CM/N500k/g2p00_k2_000.edgl")
edges = load_edgl_as_array("networks/SF_samples/g2p20_k2_N1E5/sample_0000.edgl")

N = guess_num_nodes_from(edges)
gl = Layer(N, edges, keep_neighbors_as=["awk"])


# Single-layer
print("Creating Layers and multiplex object...")
# pop = NLayerMultiplex([gl])   # FOR SIMPLE SIS
pop = NLayerMultiplex(gl, num_layers=2)  # FOR DOUBLE SIS and etc

# # Two-layer
# gl_list = [nx_to_layer(g) for g in [g, h]]
# pop = NLayerMultiplex(gl_list, num_layers=2)


# --- Simulation infrastructure allocation
# model = SimpleSIS(beta1, mu1)
model = DoubleSIS(beta1, beta2, mu1, mu2, gamma1, gamma2)


print("Now calculating...")
exd = ExecData()
# exd.alloc("p_state", pop, model)
exd.alloc_for(exd.__slots__, pop, model)


# -- Initialization
# exd.p_state[0, :] = 1. - init_I
# exd.p_state[1, :] = init_I
# initialize_states_basic(pop, exd, init_mode="frac", init_data=init_I, i_healthy_state=0,
#                         i_infective_state=1)
# initialize_states_basic(exd.p_state, N, init_mode="nodelist", init_data="[0, 1, 2]", i_healthy_state=0,
#                         i_infective_state=1)
# exd.p_next[:] = exd.p_state[:]
# print(exd.p_state)


# # -- Manual run - to test the model's core functions
# model.initialize_states(pop, exd, init_mode, init_data)
# exd.p_state[1, 0] = 0.5
# exd.p_state[3, 0] = 0.0
# exd.p_state[0, 0] = 0.5
# exd.p_next[:] = exd.p_state[:]
#
# # One step
# model.iterate_model(pop, exd)
# for i in range(10):
#     print("k = {:d}\tq = {:0.8f}".format(pop.g[0].degree(i), exd.q_trans[0, i]))
# # print(exd.f_trans[0, :10])
# exit()

# First run to compile everything
# model.calc_stationary_densities(pop, exd, max_steps=2, init_mode=init_mode, init_data=init_data)
model.calc_stationary_densities(pop, exd, max_steps=2, init_mode=init_mode, init_data=init_data)

# SINGLE COMMAND RUN
xt0 = time.time()
# model.calc_stationary_densities(pop, exd, max_steps, init_mode=init_mode, init_data=init_data)
res = None
for rep in range(1):
    # res = model._timed_calc_stationary_densities(pop, exd, max_steps, init_mode=init_mode, init_data=init_data)
    res = model.calc_stationary_densities_legacy(pop, exd, max_steps, init_mode=init_mode, init_data=init_data, tol=tol,
                                          store_mode="linear", store_period=10, calc_tseries_state=True)
xtf = time.time()
print("Number of iterations = {:d} steps".format(res.num_steps))
print("Exec time: {:0.5f} s".format(xtf - xt0))
#
# print(res.rho_tseries[:res.tbuffer_index])
# print(res.t_tseries[:res.tbuffer_index])


# ---
# PLOT A HISTOGRAM OF THE STATIONARY PREVALENCES
if plot_histo:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(exd.p_state[1], color="blue", alpha=0.25, density=True, stacked=True)
    ax.hist(exd.p_state[2], color="green", alpha=0.25, density=True, stacked=True)
    ax.set_xscale("log")
    plt.show()

# # _ TEST +_ Calculation of f factors
# def calc_f_trans(p_state_s, tp, f_trans_tr):
#     f_trans_tr[:] = 1. - tp * p_state_s

# calc_f_trans(exd.p_state[1], beta, exd.f_trans[0])
# calc_f_trans(pop.size, exd.p_state[1], beta, exd.f_trans[0])
# calc_q_trans(pop.size, pop.g[0].neighbors_awk, exd.f_trans[0], exd.q_trans[0])

# # _iterate_model(N, pop.g[0].neighbors_awk, beta, mu, 1., exd.p_state, exd.f_trans, exd.q_trans, exd.p_next)
# for t in range(max_steps):
#     model.iterate_model(pop, exd)
#     exd.p_state[:] = exd.p_next[:]

# for i in range(N):
#     # print(pop.g[0].degree(i), exd.q_trans[0, i])
#     print(pop.g[0].degree(i), exd.p_next[1, i])
#     # print(pop.g[0].degree(i), exd.p_next[0, i] + exd.p_next[1, i])
# #
# print(check_states_are_close(N, model.num_states, exd.p_state, exd.p_next, tol=tol))
# print(check_states_are_close(N, model.num_states, exd.p_state, exd.p_state, tol=tol))
#
# print(renormalize_node_probabilities(N, model.num_states, exd.p_next))
#
# print()
# for i in range(N):
#     # print(pop.g[0].degree(i), exd.q_trans[0, i])
#     # print(pop.g[0].degree(i), exd.p_next[1, i])
#     print(pop.g[0].degree(i), exd.p_next[0, i] + exd.p_next[1, i])
#

