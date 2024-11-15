# simulation/field.py
import numpy as np

import src.scal as scal
from .lattice import Lattice
from .config import Config
from src.numba_target import my_parallel_loop
from src.utils import evolve_kernel

class Field:
    def __init__(self, config: Config, latt: Lattice):
        self.latt = latt
        self.config = config
        self.initialize_field()

    def initialize_field(self):
        self.phi0 = np.zeros(self.latt.n_cells, dtype=scal.SCAL_TYPE)
        self.phi1 = np.zeros(self.latt.n_cells, dtype=scal.SCAL_TYPE)
        self.dS   = np.zeros(self.latt.n_cells, dtype=scal.SCAL_TYPE)
        self.eta  = np.zeros(self.latt.n_cells, dtype=scal.SCAL_TYPE_REAL)

    def update_field(self):
        my_parallel_loop(
            evolve_kernel,
            self.latt.n_cells,
            self.phi0,
            self.phi1,
            self.dS,
            self.eta,
            self.config.dt,
            )

    def swap(self):
        self.phi0, self.phi1 = self.phi1, self.phi0