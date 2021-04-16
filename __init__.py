
# When the package structure is more or less defined, set this file to define what is loaded as:
# # import numba_markov
# # from numba_markov import [something]
# # from numba_markov import *

# # EXAMPLE - FROM fastepidem
# # ---------
from .graph import Layer, NLayerMultiplex
# from model_base import ModelBase  # model_update functions are not private, yet not meant to be imported here.
from .models.simple_sis import SimpleSIS
from .models.double_sis import DoubleSIS
from .exec_data import ExecData
from .sim_results import SimResults
#
#
# __all__ = [
#     "StaticGraph",
#     "ModelSIS",
#     "ModelBase",
# ]
