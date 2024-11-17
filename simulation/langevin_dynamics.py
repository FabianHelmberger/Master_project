# simulation/langevin_dynamics.py
import math
import numpy as np

import src.scal as scal
from .field import Field
from src.numba_target import my_parallel_loop
from .config import Config

class LangevinDynamics(Field):
    """
    A class that takes care of the dynamics of the stochastic process.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.dS   = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE)
        self.eta  = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE_REAL)
        
    def update_noise(self):
        self.eta = scal.SCAL_TYPE_REAL(self.noise_factor * self.sqrt2 * np.random.normal(size=self.eta.shape))

    def update_drift(self, drift_kernel):
        my_parallel_loop(
            drift_kernel,
            self.n_cells,
            self
            )