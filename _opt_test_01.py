"""
Optimization attempt of the graph construction before markov chain calculations.
"""

import numpy as np
import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg
import sys
import time
from numba.typed import List as TypedList

from numba_markov.graph import Layer, NLayerMultiplex, nx_to_layer, _construct_adjacency_matrix
from numba_markov.model_base import *
from numba_markov.models.double_sis import *
from numba_markov.utils import load_edgl, load_edgl_as_array
from numba_markov.types import *

size = np_ncount_t(10)


N = int(1E5)  # Change the path too!!!!
g_fname = "networks/SF-CM/N100k/g2p00_k2_000.edgl"
# g_fname = "networks/SF_samples/g2p20_k2_N5E6/sample_0000.edgl"

# Dummy construction for compilation.
gl = Layer(N, np.array([[0, 1], [1, 2]], dtype=np_ncount_t), keep_neighbors_as=[])

# -----------------------------------
# Edgelist from file
print("Loading edgelist from file:", end="\t")
sys.stdout.flush()
xt0 = time.time()

# edges = load_edgl(g_fname)  # Loads as a list of tuples. Yikes.
edges = load_edgl_as_array(g_fname)
num_edges = len(edges)

xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))

print("Graph size: {:d} nodes".format(N))

# ----------------------------
# Graph from edgelist
print("Constructing graph object:", end="\t")
sys.stdout.flush()
xt0 = time.time()
gl : Layer = Layer(N, edges, keep_neighbors_as=["awk"])  #["alist"])#, "awk"])
xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))


# print(gl.neighbors_awk[-5:])
# print(gl.neighbors_alist[-5:])
# exit()


# # ----------------------------
# # Multiplex from graphs
# print("Constructing multiplex object:", end="\t")
# sys.stdout.flush()
# xt0 = time.time()
# pop = NLayerMultiplex([gl, gl])  # MuLtIpLeX
# xtf = time.time()
# print("{:0.6f} s".format(xtf - xt0))


# --------------------------------
# Adjacency matrix construction
print("Calculating adjacency matrices:", end="\t")
sys.stdout.flush()
xt0 = time.time()
gl.get_adjmat(sparse=True)
xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))


# print(gl.get_adjmat())

# -------------------------------
# Adjacency matrix diagonalization
print("Calculating leading eigenv.:", end="\t")
sys.stdout.flush()
xt0 = time.time()
g_eigval, g_eigvec = gl.get_eig()
xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))

print("\n--------------\n")
print("Eigenvalue: ", g_eigval)
print("Eigenvector: ", g_eigvec)


