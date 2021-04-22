"""
A static graph for optimized Markov chain calculations.
Graphs (i.e., layers) can be grouped into a multiplex structure.

The graph must be initialized with all vertices and edges. Once done, its structure can't be changed.

Author: Paulo Cesar Ventura da Silva (https://github.com/paulocv)
"""

import awkward as ak
import numba as nb
from numba.typed import List as TypedList
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from .types import np_ncount_t, np_float_t  # , nb_ncount_t
from .utils import load_edgl_as_array, guess_num_nodes_from


class Layer:
    """
    A simple graph class with a fixed structure.
    Nodes and edges are informed at initialization and held fixed during
    the existence of the object.

    From: fastepidem module
    """

    __slots__ = ["size", "edges", "num_edges", "neighbors_list", "neighbors_alist", "neighbors_awk",
                 "_adjmat", "_eigv_centrality", "_eigval"]

    # @profile
    def __init__(self, size, edges,
                 keep_neighbors_as=("list", "alist", "awk")):
        """
        Initializes a static graph with fixed structure.
        The parameter edges is necessarily a 2D numpy array of pairs a[i] = [node_index, neighbor_index].

        For use with different iteration methods, various types of adjacency structures
        (i.e., lists of neighbors for each node) can be kept by the graph. This is defined
        by the `keep_neighbors_as` parameter. Some may take much longer to be constructed though, depending
        on the current implementation.

        Devnote: the "list", "alist", ... names can be annoying to be flying around.
          If repeated use of these names is needed, consider making a list of such names
          and dict of methods to allocate these structures.

        Devnote future feature: a shortcut for when only "awk" is asked to be kept. The code could construct the
          awkward array directly from the array of edges.

        Devnote future feature: after migrating from python list to numba typed list, the non-numba construction of
          neighbors_alist is quite slow. Maybe using a numba typed list and a numbaed function could speed this up,
          if possible.

        Parameters
        ----------
        size : int
        edges : a container of tuples.
        keep_neighbors_as : iterable
        """
        super().__init__()
        self.size = np_ncount_t(size)

        # ----------------------------------------------------------
        # List of edges - type check and attribution
        # Check: edges now must be a numpy array of shape (num_edges, 2)
        if type(edges) is not np.ndarray:
            raise TypeError("Hey, now the argument 'edges' is required to be a numpy array of shape (num_edges, 2). "
                            "If you used 'load_edgl', consider using 'load_edgl_as_array' instead.")
        self.edges = edges
        self.num_edges = edges.shape[0]

        # ------------------------------------------------------
        # Construction of the adjacency list (using python lists) from a list of edges
        # self.neighbors_list = [[] for _ in range(size)]  # Python list. Can't be numbaed.
        self.neighbors_list = _make_typed_list_for_neighbors(self.size)  # Numba typed list.

        # self._add_edges_from(edges)  # Pure python construction
        _add_edges_from(edges, self.size, self.neighbors_list)  # Numba construction with numba typed list

        # ------------------------------------------------------
        # Creation of other neighbor containers for optimal operation.
        self.neighbors_alist = None
        self.neighbors_awk = None

        # List of numpy arrays, for semi-vectorized methods
        if "alist" in keep_neighbors_as:
            # This is currently slow, specially because neighbors_list is a numba typed list, slower outside numba.
            self.neighbors_alist = [np.array(neighs, dtype=np_ncount_t) for neighs in self.neighbors_list]

        # Awkward array for optimal handling of the ragged structure with numba.
        if "awk" in keep_neighbors_as:
            # ----
            # Low-level build using ak.ArrayBuilder. Can be numbaed.
            builder = ak.ArrayBuilder()
            _make_awk_from_typed_list(self.neighbors_list, builder)
            self.neighbors_awk = builder.snapshot()
            del builder

            # ----
            # High-level build. Will be slower with numba typed lists than with regular lists.
            # self.neighbors_awk = ak.Array(self.neighbors_list)  # Old formulation, slower

        # Optionally deletes the original list-of-lists.
        if "list" not in keep_neighbors_as:
            del self.neighbors_list
            self.neighbors_list = None

        # -----------------------------------------------------
        # Extra data structures, constructed and returned uppon request
        self._adjmat = None
        self._eigv_centrality = None
        self._eigval = None

    def __len__(self):
        return self.size

    def assert_node(self, i):
        """Checks if index i is valid as a node index."""
        if i < 0 or i >= self.size:
            raise ValueError("Hey, node index {} does not fit into graph with size {}."
                             "".format(i, self.size))

    def _add_edge(self, i, j):
        self.assert_node(i)
        self.assert_node(j)
        self.neighbors_list[i].append(np_ncount_t(j))
        self.neighbors_list[j].append(np_ncount_t(i))

    # def _add_edges_from(self, edges):
    #     for i, j in edges:
    #         self._add_edge(i, j)

    # Slightly more performant than the previous (with edges as an array)
    def _add_edges_from(self, edges):
        for edge in edges:
            self._add_edge(edge[0], edge[1])

    # ----------------------------------
    # Convenience methods
    # ----------------------------------
    # These are not focused on performance.

    def neighbors(self, i):
        """Gets a sequence of neighbors from node i.
        This is intended for simple high-level access, not for performance.
        It tries to dodge the differences in the allocated structures.
        """
        for suffix in ["list", "alist", "awk"]:
            attr = self.__getattribute__("neighbors_" + suffix)
            if attr is not None:
                return attr[i]
        else:
            raise AttributeError("Hey, graph {} oddly doesn't have any allocated adjacency"
                                 " structure. This may be an internal error, but you can"
                                 " check the parameter 'keep_neighbors_as' at the"
                                 " initialization of this class.".format(self))

    def degree(self, i):
        return len(self.neighbors(i))

    def get_adjmat(self, sparse=True, recalc=False):
        """
        Returns the adjacency matrix of the graph (with float datatype).
        Each entry (i, j) is 1. if i and j are connected and 0. otherwise.
        The matrix is allocated and calculated by the first call to this function. In the following ones,
        the stored result is returned. A new allocation/calculation is done if recalc == True.

        The construction of the matrix is not optimized.
        """
        if self._adjmat is None or recalc:
            # --- scipy.sparse matrix construction
            if sparse:
                # # ---------
                # # Old: constructs LIL format first within for loops, then converts to CSR
                # mat = scipy.sparse.lil_matrix((self.size, self.size))  # LIL (list-of-lists) representation
                # for i in range(self.size):
                #     for j in self.neighbors(i):
                #         mat[i, j] = 1.
                # # Converts to the Compressed Sparse Row representation, which is better for matmul operations.
                # self._adjmat = mat.tocsr()
                # del mat

                # # ---------
                # # New: direct construction of CSR from list of edges (assuming it is a 2D array of pairs). Way faster.
                # Temporarily duplicates the edges to include the reciprocal entries
                tmp_row_idx = np.concatenate((self.edges[:, 0], self.edges[:, 1]))
                tmp_col_idx = np.concatenate((self.edges[:, 1], self.edges[:, 0]))
                self._adjmat = scipy.sparse.csr_matrix((np.ones(2 * self.num_edges, dtype=np_float_t),  # Data: 1
                                                       (tmp_row_idx, tmp_col_idx)),  # Indexes of non-null entries
                                                       shape=(self.size, self.size))
                del tmp_row_idx, tmp_col_idx

            # --- Non-sparse matrix construction
            else:
                # Allocates and constructs the adjacency matrix
                self._adjmat = np.zeros((self.size, self.size), dtype=float)

                for i in range(self.size):
                    for j in self.neighbors(i):
                        self._adjmat[i, j] = 1.

        return self._adjmat

    def get_eig(self, sparse=True, recalc=False):
        """
        Returns the greatest eigenvalue and the corresponding eigenvector of the adjacency matrix.

        Set sparse to False to prevent using scipy.sparse diagonalization function (optimized for sparse matrix).
        """
        if self._eigv_centrality is None or recalc:
            mat = self.get_adjmat()

            if sparse:
                self._eigval, self._eigv_centrality = scipy.sparse.linalg.eigs(mat, k=1)
            else:
                self._eigval, self._eigv_centrality = scipy.linalg.eig(mat)

            # Reshapes and rearranges the array, then normalizes by its sum
            self._eigv_centrality = self._eigv_centrality.real.flatten().ravel()
            self._eigv_centrality /= np.sum(self._eigv_centrality)
            self._eigval = self._eigval.real[0]

        return self._eigval, self._eigv_centrality


class NLayerMultiplex:
    """A simple bunch class that groups same-sized layers to form a multiplex. Created from Layer objects,
    promotes type and size checks on initialization.
    """

    __slots__ = ["g", "num_layers", "size"]

    def __init__(self, g_list, num_layers=None):
        """Creates a multiplex object from a list of Layer objects.

        g_list must be a list of Layer objects (not networkx graphs!).
        Optionally, you can pass a single Layer object and specify num_layers to create a multiplex with
        repeated layers (i.e., a single-layer network with multiplex compatibility). If num_layers is not
        informed, it is treated as 1.


        Parameters
        ----------
        g_list : list, Layer
            List of Layer objects. If a single Layer, repeats it over num_layers.
        num_layers : int
            Number of layers for repeated multiplex. Ignored if g_list is not a single Layer object.
        """

        # --------------
        # Convenience routines
        # --------------
        # Shorthand to create a n-layer multiplex from the same layer
        if isinstance(g_list, Layer):
            if num_layers is None:
                num_layers = 1
            g_list = [g_list for _ in range(num_layers)]

        # Now, regardless of the g_list original type, num_layers is set to its size.
        num_layers = len(g_list)

        # ---------------
        # Construction checkouts
        # ---------------
        # Type check. We do not create Layers from nx networks on the fly.
        for g in g_list:
            if not isinstance(g, Layer):
                raise TypeError("Hey, to initialize a multiplex, each element of g_list must be a Layer object. "
                                "Currently, one of the items is a '{}' object.".format(type(g)))

        # Size check:
        size = g_list[0].size
        for i_g, g in enumerate(g_list[1:]):
            if g.size != size:
                raise ValueError("Hey, the multiplex received layers with different sizes.\n"
                                 "Size of the first layer:   {:d}\n"
                                 "Size of the {:d}-th layer: {:d}".format(self.size, i_g + 1, g.size))

        # ---------------
        # Object attributes
        # ---------------
        self.g = g_list
        self.num_layers = num_layers
        self.size = size

    def __len__(self):
        """Gets the number of nodes, not layers."""
        return self.size


# ----------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------


def nx_to_layer(g_nx, keep_neighbors_as=None):
    """Shorthand to create a Markov Layer from an nx Graph.
    Only use nx Graphs that contain sequential indexes as names!! Otherwise, the edges can be wrongly set.
    """
    if keep_neighbors_as is None:
        return Layer(len(g_nx), list(g_nx.edges()))
    else:
        return Layer(len(g_nx), list(g_nx.edges()), keep_neighbors_as=keep_neighbors_as)


def layer_to_nx(g_layer):
    """Shorthand to create an nx Graph from a Markov Layer."""
    raise NotImplementedError


def get_layer_paths_from_dict(input_dict, num_layers, layer_path_fmt="layer{:1d}_path"):
    """Reads the layer paths from input_dict, as the parameters "layer[]_path". """
    return list((input_dict[layer_path_fmt.format(i + 1)] for i in range(num_layers)))


def make_multiplex_from_dict(input_dict, num_layers, layer_path_fmt="layer{:1d}_path", num_nodes=None,
                             keep_neighbors_as=("awk", )):
    """
    Creates an NLayerMultiplex instance taking arguments from input_dict and loading layer from edgelist files.

    Parameters
    ----------
    input_dict : dict
        Dictionary with (at least but not only) the paths of each layer's edgelist file.
    num_layers : int
        Number of layers to be used.
    layer_path_fmt : str
        Format string for the input_dict keys that contain the layer paths.
        Defaults to layer{:1d}_path. Must be formatable by .format(i), where i is an integer STARTING FROM 1.
    num_nodes : int
        Previously informed number of nodes of each layer.
        If not informed, it is guessed from each layer's edgelist. In this case, an error is raised if the layers
        do not have the same number of nodes.
    keep_neighbors_as : tuple
        Types of containers to keep the list of neighbors (adjacency lists) in each layer. Refer to Layer class
        documentation (__init__ function).

    Returns
    -------
    pop : NLayerMultiplex
    """
    layer_paths = get_layer_paths_from_dict(input_dict, num_layers, layer_path_fmt)
    g_list = []
    for path in layer_paths:
        edges = load_edgl_as_array(path)
        if num_nodes is None:
            # An error will be raised later if the size of each layer is different
            num_nodes = guess_num_nodes_from(edges)
        g_list.append(Layer(num_nodes, edges, keep_neighbors_as=keep_neighbors_as))

    return NLayerMultiplex(g_list)


# -----------------------------------------
# NUMBA FUNCTIONS
# -----------------------------------------

@nb.njit
def _make_typed_list_for_neighbors(size):
    """
    Creates an empty nested typed list, with two levels and 'size' elements.

    As of 2021/04/13, numba typed List is experimental and cannot receive its type in advance. Thus a very ugly
    append-and-pop solution had to be used.
    """
    res = TypedList()

    # Creates a prototype for the inner lists, with type inference via append-and-pop.
    proto_list = TypedList()
    proto_list.append(np_ncount_t(0))
    proto_list.pop()
    for i in range(size):
        res.append(proto_list.copy())  # Just copy the empty prototype
    return res


@nb.njit
def _assert_node_index(i, size):
    """Checks and if a node index is compatible with a graph's size."""
    if i >= size:
        # Error message must be a compile-time constant.
        raise ValueError("Hey, a node index is greater than the graph's size. I cannot know which index is because "
                         "I am a numba-compiled function, but you can try constructing the list of neighbors with "
                         "Layer._add_edges_from, which calls the non-numba Layer._assert_node. This will report the "
                         "invalid index to you.")


@nb.njit
def _add_edges_from(edges, size, neighbors_list):
    """
    Numba function that constructs a list of neighbors from a numpy array of edges.
    NOTICE: edges must be a numpy array, so read it from file using 'load_edgl_as_array' instead of 'load_edgl'.
    The neighbors_list must have been previously allocated as a numba typed list. Use _make_typed_list_for_neighbors.
    """
    for i, j in edges:
        _assert_node_index(i, size)
        _assert_node_index(j, size)
        neighbors_list[i].append(np_ncount_t(j))
        neighbors_list[j].append(np_ncount_t(i))


# https://awkward-array.readthedocs.io/en/stable/_auto/ak.Array.html
# "(...) The only limitation is that Awkward Arrays cannot be created inside the Numba-compiled function;
# to make outputs, consider ak.ArrayBuilder"

# See this topic for a detailed discussion from a user with more or less the same problem:
# https://github.com/scikit-hep/awkward-1.0/discussions/328
@nb.njit
def _make_awk_from_typed_list(neighbors_list, builder):
    """
    Constructs an awkward array from a numba typed list using awkward's ArrayBuilder.
    No returned object. The actual awkward array must be extracted from builder.snapshot().
    """
    for neighs in neighbors_list:
        # Creates a new line ("list") in the builder
        # # with builder.list():  # Numba incompatible
        builder.begin_list()
        for j in neighs:
            builder.integer(j)  # Appends as int64, unavoidably...
        builder.end_list()


@nb.njit
def _construct_adjacency_matrix(neighbors, size, lil_mat):
    """"""
    for i in range(size):
        for j in neighbors[i]:
            lil_mat[i, j] = 1.
