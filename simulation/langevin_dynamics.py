# simulation/langevin_dynamics.py
import math
import numpy as np
from numpy import abs as arr_abs

import src.scal as scal
from .field import Field
from src.numba_target import my_parallel_loop, use_cuda, threadsperblock
from .config import Config

amax = np.max
mean = np.mean
if use_cuda: 
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states

    import cupy as cp
    from cupy import add, multiply, real, imag
    from cupy import abs as arr_abs

    def amax(arr):
        return cp.amax(cp.asarray(arr)).get().item()

    def mean(arr):
        return cp.mean(cp.asarray(arr)).get().item()


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


        if use_cuda:
            n_blocks = math.ceil(self.n_cells / threadsperblock)
            self.rng = create_xoroshiro128p_states(
                threadsperblock * n_blocks, seed=self.noise_seed
            )

    def update_drift(self, drift_kernel, *kernel_param):
        my_parallel_loop(
            drift_kernel,
            *kernel_param
            )
        if use_cuda: cuda.synchronize()
        
    def update_noise(self, noise_kernel, *kernel_param):
        my_parallel_loop(
            noise_kernel,
            *kernel_param
            )
        if use_cuda: cuda.synchronize()
        
    def update_field(self, evolve_kernel, *kernel_param):
        my_parallel_loop(
            evolve_kernel,
            *kernel_param
            )
        if use_cuda: cuda.synchronize()
        self.langevin_time += self.dt_ada
    
    def set_apative_stepsize(self):
        # TODO: if multiple trajectories are run in parallel and this is implemented as another lattice dimension,
        # the mean and max of dS should not be calculated across different trajectories
        
        self.dt_ada = self.dt
        
        self.dS_norm = arr_abs(self.dS)
        self.dS_max = amax(self.dS_norm)
        self.dS_mean = mean(self.dS_norm)

        if self.dS_max > self.DS_MAX_LOWER:
            self.dt_ada = (
                (self.dt_ada * self.mean_dS_max / self.dS_mean)
                if self.mean_dS_max < self.dS_max
                else self.dt_ada
            )