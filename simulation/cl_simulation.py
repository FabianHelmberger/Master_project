"""
    Complex Langevin simulation module
"""

import numpy as np

# from src.numba_target import use_cuda, myjit, my_parallel_loop, prange, threadsperblock

from .field import *
from .config import *
from .langevin_dynamics import *
from .observables import *
from .lattice import *

class ComplexLangevinSimulation:

    def __init__(self, config):
        self.config = config
        self.lattice = Lattice(self.config)
        self.field = Field(self.config, self.lattice)
        self.dynamics = LangevinDynamics(self.config, self.field, self.lattice)
        # self.observables = Observables(self.config, self.field)