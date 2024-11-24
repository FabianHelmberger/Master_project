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
        self.dS = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE)
        self.dS_norm = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE_REAL)
        self.eta = np.zeros(self.n_cells, dtype=scal.SCAL_TYPE_REAL)
        self.langevin_time: scal.SCAL_TYPE_REAL = 0.0
        self.dt_ada = self.dt
        self.dS_max: scal.SCAL_TYPE_REAL = 0.0
        self.mean_dS_max: scal.SCAL_TYPE_REAL = 5
        self.dS_mean: scal.SCAL_TYPE_REAL = 0.0
        self.DS_MAX_UPPER = 1e12
        self.DS_MAX_LOWER = 1e-12

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

    
    def set_apative_stepsize(self):
        # TODO: if multiple trajectories are run in parallel and this is implemented as another lattice dimension,
        # the mean and max of dS should not be calculated across different trajectories
        
        self.dt_ada = self.dt
        self.dS_max = max(self.dS_norm)
        self.dS_mean = np.mean(self.dS_norm)

        if self.dS_max > self.DS_MAX_LOWER:
            self.dt_ada = (
                (self.dt_ada * self.mean_dS_max / self.dS_mean)
                if self.mean_dS_max < self.dS_max
                else self.dt_ada
            )