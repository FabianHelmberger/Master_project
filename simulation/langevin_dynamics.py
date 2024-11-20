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
        self.langevin_time: scal.SCAL_TYPE_REAL = 0.0
        self.dt_ada = self.dt

    def update_drift(self, drift_kernel, *kernel_param):
        my_parallel_loop(
            drift_kernel,
            *kernel_param
            )

    def update_noise(self, noise_kernel, *kernel_param):
        my_parallel_loop(
            noise_kernel,
            *kernel_param
            )

    def update_field(self, evolve_kernel, *kernel_param):
        my_parallel_loop(
            evolve_kernel,
            *kernel_param
            )
        self.langevin_time += self.dt_ada