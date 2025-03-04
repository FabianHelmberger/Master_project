# simulation/field.py
import numpy as np

import src.scal as scal
from .lattice import Lattice
from .config import Config
from src.numba_target import my_act_parallel_loop
from src.utils import evolve_kernel, swap_kernel

class Field(Lattice):
    def __init__(self, config):
        super().__init__(config)

        # self.phi0 = scal.SCAL_TYPE(np.random.normal(self.n_cells))
        # self.phi1 = scal.SCAL_TYPE(np.random.normal(self.n_cells))

        self.phi0 = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE)
        self.phi1 = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE)
        # rand = np.random.normal(size=(4, 2)).astype(scal.SCAL_TYPE)
        # self.phi0 = rand[0,:]
        # self.phi1 = rand[2,:]

