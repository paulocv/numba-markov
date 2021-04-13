""" MULTIPLE RUNS OF THE MARKOV CHAIN CALCULATION - as devtest"""

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from numba_markov.graph import Layer, NLayerMultiplex, nx_to_layer
from numba_markov.model_base import *
from numba_markov.models.double_sis import *
from numba_markov.utils import load_edgl


# # Pop paramenters
# kmean = 5.
# p = kmean / N

# Model parameters - SIS
num_beta = 50
beta1_list = np.linspace(0.002, 0.100, num_beta)  # 0.045  # 0.005
beta2_list = np.linspace(0.002, 0.100, num_beta)  # 0.045  # 0.005

mu1 = 0.5
mu2 = 0.5

gamma1 = 0.5
gamma2 = 2.0

# Initial conditions?
init_mode = "p"  # "nodelist"
init_data = 0.05  # [0, 1, 2, 3]

# Simulation parameters
t_max = 3000
tol = 1.E-6


# --------------------------------------------
# Population / Network loading


N = int(1E4)  # int(5E5)
print("Loading network edgelist from file")
# edges = load_edgl("networks/ER/ER_10000_k10_1.edgl")
edges = load_edgl("networks/SF-CM/N10k/g2p00_k2_000.edgl")
# edges = load_edgl("networks/SF-CM/N100k/g2p00_k2_000.edgl")
# edges = load_edgl("networks/SF-CM/N500k/g2p00_k2_000.edgl")
gl = Layer(N, edges, keep_neighbors_as=["awk"])

# Another layer
edges = load_edgl("networks/SF-CM/N10k/g2p00_k2_002.edgl")
hl = Layer(N, edges, keep_neighbors_as=["awk"])


print("Creating Layers and multiplex object...")
# pop = NLayerMultiplex(gl, num_layers=2)   # Same layer repeated
pop = NLayerMultiplex([gl, hl])  # MuLtIpLeX

# -------------------------------------------------
# Model infrastructure
exd = ExecData()

z_array = np.empty((num_beta, num_beta), dtype=float)
x_array = np.empty((num_beta, num_beta), dtype=float)
y_array = np.empty((num_beta, num_beta), dtype=float)


# ---------------------------------------------
# MODEL EXECUTION

for ((i_b1, beta1), (i_b2, beta2)) in itertools.product(enumerate(beta1_list), enumerate(beta2_list)):

    print("{:0.4f}\t{:0.4f}".format(beta1, beta2), end="\t")
    model = DoubleSIS(beta1, beta2, mu1, mu2, gamma1, gamma2)
    res = model.calc_stationary_densities(pop, exd, t_max, init_mode=init_mode, init_data=init_data,
                                          tol=tol)
    print(res.num_steps)

    # Store data
    x_array[i_b1, i_b2] = beta1
    y_array[i_b1, i_b2] = beta2
    z_array[i_b1, i_b2] = res.num_steps



# ----------------
# For regular data sets, use rectangular grid.

#
# for i in range(num_beta):
#     for j in range(num_beta):
#         z_array[i][j] = z_data[l*i+j]
#         x_array[i][j] = beta1_list[num_beta*i+j]
#         y_array[i][j] = beta2_list[num_beta*i+j]

fig, ax = plt.subplots()
ax.pcolormesh(x_array, y_array, z_array,
             cmap=plt.get_cmap("RdBu"))
plt.tight_layout()
plt.show()
