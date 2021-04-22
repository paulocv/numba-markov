"""
This file defines what is loaded as:
import numba_markov  # then use numba_markov.Layer
from numba_markov import [something]  # then simply call [something]
from numba_markov import *   # Extracts from the __all__ variable
"""

# # ---------
from .graph import Layer, NLayerMultiplex
# from model_base import ModelBase  # model_update functions are not private, yet not meant to be imported here.
from .models import SimpleSIS, DoubleSIS  # These are exposed on models/__init__.py
from .models import get_model_instance
from .exec_data import ExecData
from .sim_results import SimResults

#
# __all__ = [
#     "StaticGraph",
#     "ModelSIS",
#     "ModelBase",
# ]
