# simulation/langevin_dynamics.py
import math
import numpy as np
import numba
from numpy import abs as arr_abs


import src.scal as scal
from simulation.field import Field
from src.numba_target import my_parallel_loop, use_cuda, threadsperblock
from simulation.config import Config
from src.utils import KernelBridge, update_langevin_time, chunk_max_kernel, adaptive_step_kernel

from numpy import max as amax
from numpy import mean
from numpy import abs as arr_abs

if use_cuda: 
    from numba.cuda.random import create_xoroshiro128p_states

    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from src.utils import arr_abs_kernel
    from numba import cuda
    def arr_abs(in_array: DeviceNDArray, out_array: DeviceNDArray) -> None:
        my_parallel_loop(arr_abs_kernel, in_array.size, in_array, out_array)
        cuda.synchronize()

    import cupy as cp
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
        self.langevin_time = np.zeros(self.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.DS_MAX_UPPER = 1e12
        self.DS_MAX_LOWER = 1e-12

        # for adaptive stepsize (in parallel trajs mode)
        self.adaptive_step_kernel = adaptive_step_kernel
        self.dS_max = np.zeros(self.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.ada = np.ones(self.trajs, scal.SCAL_TYPE_REAL)
        self.mean_dS_max: scal.SCAL_TYPE_REAL = 5
        self.dS_mean: scal.SCAL_TYPE_REAL = 0.0
        self.langevin_steps = 0


        # buffer for kernel args
        self.cldyn_kernel_args = None
        self.cldyn_kernel_bridge = KernelBridge(self, [self.noise_kernel, self.drift_kernel, self.evolve_kernel, self.adaptive_step_kernel], const_param={})

        if use_cuda:
            n_blocks = math.ceil(self.n_cells / threadsperblock)
            self.rng = create_xoroshiro128p_states(
                threadsperblock * n_blocks, seed=self.noise_seed
            )

    def update_drift(self, *kernel_param):
        my_parallel_loop(
            self.drift_kernel,
            *kernel_param
            )
        if use_cuda: cuda.synchronize()
        
    def update_noise(self, *kernel_param):
        my_parallel_loop(
            self.noise_kernel,
            *kernel_param
            )
        if use_cuda: cuda.synchronize()
        
    def update_field(self, *kernel_param):
        my_parallel_loop(
            self.evolve_kernel,
            *kernel_param
            )
        if use_cuda: cuda.synchronize()
        
        my_parallel_loop(
            update_langevin_time, 
            self.trajs,
            self.langevin_time,
            self.ada,
            self.dt
        )
        if use_cuda: cuda.synchronize()
        self.langevin_steps += 1
    
    def set_apative_stepsize(self):
        if self.ada_step:

            # calculate the max drift of every traj
            my_parallel_loop(chunk_max_kernel, self.trajs, self.dS_norm, self.dS_max, self.adims[1])
            if use_cuda: cuda.synchronize()

            args = self.cldyn_kernel_bridge.get_current_params()[self.adaptive_step_kernel]
            my_parallel_loop(self.adaptive_step_kernel, *args.values())
            if use_cuda: cuda.synchronize()

    def step(self):
        self.cldyn_kernel_args = self.cldyn_kernel_bridge.get_current_params()
        self.update_noise(*self.cldyn_kernel_args[self.noise_kernel].values())
        self.update_drift(*self.cldyn_kernel_args[self.drift_kernel].values())
        if self.ada_step: self.set_apative_stepsize()
        self.update_field(*self.cldyn_kernel_args[self.evolve_kernel].values())
        
        self.swap()