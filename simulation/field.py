# simulation/field.py
import numpy as np

import src.scal as scal
from .lattice import Lattice
from .config import Config
from src.numba_target import my_parallel_loop
from src.utils import evolve_kernel

class Field(Lattice):
    def __init__(self, config):
        super().__init__(config)

        self.phi0 = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE)
        self.phi1 = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE)

    def swap(self):
        self.phi0, self.phi1 = self.phi1, self.phi0