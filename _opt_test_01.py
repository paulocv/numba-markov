"""
Optimization attempt of the graph construction before markov chain calculations.
"""

import numpy as np
import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
import sys
import time
from numba.typed import List as TypedList

from numba_markov.graph import Layer, NLayerMultiplex, nx_to_layer
from numba_markov.model_base import *
from numba_markov.models.double_sis import *
from numba_markov.utils import load_edgl, load_edgl_as_array
from numba_markov.types import *

size = np_ncount_t(10)


N = int(1E5)  # Change the path too!!!!
g_fname = "networks/SF-CM/N100k/g2p00_k2_000.edgl"

# -----------------------------------
# Edgelist from file
print("Loading network edgelist from file:", end="\t")
sys.stdout.flush()
xt0 = time.time()

# edges = load_edgl(g_fname)  # Loads as a list of tuples. Yikes.
edges = load_edgl_as_array(g_fname)

xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))


gl = Layer(N, edges, keep_neighbors_as=[])

# ----------------------------
# Graph from edgelist
print("Constructing graph object:", end="\t")
sys.stdout.flush()
xt0 = time.time()
gl = Layer(N, edges, keep_neighbors_as=["awk"])  #["alist"])#, "awk"])
xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))


# print(gl.neighbors_awk[-5:])
# print(gl.neighbors_alist[-5:])
# exit()


# ----------------------------
# Multiplex from graphs
print("Constructing multiplex object:", end="\t")
sys.stdout.flush()
xt0 = time.time()
pop = NLayerMultiplex([gl, gl])  # MuLtIpLeX
xtf = time.time()
print("{:0.6f} s".format(xtf - xt0))


# --------------------------------
# Adjacency matrix construction
# TODO
