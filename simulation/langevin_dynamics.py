# simulation/langevin_dynamics.py
import math
import numpy as np

from .config import Config
from .field import Field
from .lattice import Lattice
from src.numba_target import my_parallel_loop
from src.utils import evolve_kernel

class LangevinDynamics:
    def __init__(self, config: Config, field: Field, latt: Lattice):
        self.config = config
        self.field = field
        self.latt = latt

        self.sqrt2 = config.sqrt2
        self.noise_factor = self.config.noise_factor
        
    def update_noise(self):
        self.field.eta = self.noise_factor * self.sqrt2 * np.random.normal(size=self.field.eta.shape)

    def update_field(self):
        my_parallel_loop(
                    evolve_kernel,
                    self.latt.n_cells,
                    self.field.phi0,
                    self.field.phi1,
                    self.field.dS,
                    self.field.eta,
                    self.config.dt,
                    )