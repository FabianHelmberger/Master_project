"""
    Complex Langevin simulation module
"""

import cmath, math
import numba
import numpy as np
from numpy import abs as arr_abs
from numpy import around, amax, mean, imag, real, reshape

import src.lattice as l
import src.scal as scal
from   src.numba_target import use_cuda, myjit, my_parallel_loop, prange, threadsperblock

from config import *
from field import *
from langevin_dynamics import *
from observables import *

class ComplexLangevinSimulation:
    
    def __init__(self, config):
        self.config = config
        self.field = Field(config)
        self.dynamics = LangevinDynamics(config, self.field)
        self.observables = Observables(config, self.field)