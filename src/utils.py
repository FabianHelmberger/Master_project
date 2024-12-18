import math
import numpy as np
import inspect
from typing import TYPE_CHECKING, Callable, Dict, Any, List

import src.scal as scal
from src.numba_target import myjit, use_cuda
from simulation.constants import SQRT2

if use_cuda: 
    from numba.cuda.random import xoroshiro128p_normal_float32 # type: ignore
    from cupy import multiply # type: ignore

if TYPE_CHECKING:
    # Import only for type checking
    from simulation.langevin_dynamics import LangevinDynamics


@myjit
def shift(index, dir, amount, dims, adims):
    res = index
    di = index // adims[dir + 1]
    wdi = di % dims[dir]
    
    if amount > 0:
        if wdi == dims[dir] - 1:
            res = res - adims[dir]
        res = res + adims[dir + 1]
    else:
        if wdi == 0:
            res = res + adims[dir]
        res = res - adims[dir + 1]
    return int(res)  # needs explicit cast, otherwise 'res' promoted to a float otherwise. this is new behaviour.

@myjit
def get_index(pos, dims, traj):
    index = traj
    for d in range(0, len(dims)):
        index = index * dims[d] + pos[d]
    return index

@myjit
def noise_kernel(idx, eta, noise_factor):
    eta[idx] = SQRT2 * noise_factor * scal.SCAL_TYPE_REAL(np.random.normal())

@myjit
def cuda_noise_kernel(idx, eta, noise_factor, rng):
    r = xoroshiro128p_normal_float32(rng, idx)
    eta[idx] = SQRT2 * noise_factor * r

@myjit
def evolve_kernel(idx, phi0, phi1, dS, eta, ada, dt, adims):
    # TODO: move dt_sqrt
    traj_idx = idx // adims[1]
    ada_dt = ada[traj_idx] * dt
    etaterm = eta[idx] * math.sqrt(ada_dt)
    update = etaterm - ada_dt * dS[idx]
    phi1[idx] = phi0[idx] + update
    
@myjit
def chunk_max_kernel(idx, array, max_array, chunk_size):
    start_idx = idx * chunk_size
    stop_idx  = start_idx + chunk_size
    max_val = -1.0e20  # Initialize to a very large negative number

    for i in range(start_idx, stop_idx):  # Loop within the chunk
        if array[i] > max_val:
            max_val = array[i]
        max_array[idx] = max_val  # Store the result in max_array

@myjit
def update_langevin_time(traj_idx, langevin_time, ada, dt):
    langevin_time[traj_idx] += ada[traj_idx]*dt

@myjit
def euclidean_drift_kernel(idx, field, dims, adims, dS_out, mass_real, mass_imag):
    """
    Computes and returns the action drift term on the euclidean branch at lattice site `idx`.
    This has to be completely imaginary (by convention):
    The field stays real, update uses 1j*ds

    :param idx:         lattice site index
    :param field:       scalar field array
    :param dims:        lattice dimensions
    :param adims:       cumulative product of lattice dimensions
    :param dS_out:      drift term arrayvpn.tuwien.ac.at
    :param mass_real:   bare mass_real
    :param mass_imag:   bare mass_imag
    """
    n_dims = len(dims)
    out = 0

    # temporal
    idx_plus  = shift(idx, 0, +1, dims, adims)
    idx_minus = shift(idx, 0, -1, dims, adims)
    phi_idx = field[idx]
    out += field[idx_minus] + field[idx_plus]-2*phi_idx

    # spacial
    for i in range(1, n_dims):
        idx_plus  = shift(idx, i, +1, dims, adims)
        idx_minus = shift(idx, i, -1, dims, adims) 
        out += field[idx_minus]+ field[idx_plus]-2*phi_idx

    out += mass_real**2 * phi_idx
    dS_out[idx] = out

@myjit
def mexican_hat_kernel_real(idx, phi0, dS, dS_norm, mass_real, interaction):
    phi_idx = phi0[idx]
    out = 0
    out += mass_real * phi_idx
    out += interaction * phi_idx*phi_idx*phi_idx
    dS[idx] = out
    dS_norm[idx] = abs(dS[idx])

@myjit 
def adaptive_step_kernel(idx, dS_max, ada, DS_MAX_LOWER, mean_dS_max):
    this_dS_max = dS_max[idx]
    # TODO: use mask and activated parallel loop
    if this_dS_max > DS_MAX_LOWER and mean_dS_max < this_dS_max:
        ada[idx] = mean_dS_max / this_dS_max

@myjit
def arr_abs_kernel(idx, in_array, out_array):
    out_array[idx] = abs(in_array[idx])

@myjit
def mark_equilibrated_trajs_kernel(traj_idx, meas_time, langevin_time, marker_array, thermal_time, auto_corr):
    """Mark a given trajectory as ready to be observes. marker_array[traj_idx] is set true if: 
    the observable is equilibratedelated from its last observation and the observables is thermalized. 

    Args:
        traj_idx (int): Index of the current trajectory
        meas_time (array-like): array containing langevin times of last measurement (of this obs)
        langevin_time (array-like): array of current langevin times (all trajectories)
        marker_array (array-like): resulting array that indicates readiness to be measured
        thermal_time (float): thermalization time of this obs
        auto_corr (float): auto-correlation time of this obs
    """
    delta = langevin_time[traj_idx] - meas_time[traj_idx]
    if delta >= auto_corr and langevin_time[traj_idx] >= thermal_time: 
        marker_array[traj_idx] = True

    else: 
        marker_array[traj_idx] = False

@myjit
def fill_history_kernel(traj_idx, equilibrated_traj, in_array, out_array, adims):
    start_idx = traj_idx * adims[1]
    stop_idx = start_idx + adims[1]

    for i in range(start_idx, stop_idx):
        out_array[i] = in_array[i]

@myjit
def update_rolling_stats_scal_kernel(traj_idx, act_matrix, result, rolling_mean, rolling_sqr_mean, counter):
    rolling_mean[0] += result[traj_idx]
    rolling_sqr_mean[0] += math.pow(abs(result[traj_idx]), 2)
    counter[0] += 1

@myjit
def get_rolling_stats_scal_kernel(traj_idx, rolling_mean, rolling_sqr_mean, counter, mean, std):
    var = rolling_sqr_mean[traj_idx] / counter[traj_idx] - math.pow(abs(rolling_mean[traj_idx]/counter[traj_idx]), 2)
    std[traj_idx] = math.sqrt(var)
    mean[traj_idx] = rolling_mean[traj_idx] / counter[traj_idx]
    

# @myjit
# def fill_result_kernel(traj_idx, equilibrated_traj, in_array, out_array):

    
class KernelBridge:
    """
    Interface to the `my_parallel_loop` function in numba_target. 
    Automates the generation of parameter dictionaries for kernel functions.

    Attributes:
        sim: Instance of ComplexLangevinSimulation providing simulation parameters.
        kernel_funcs: Dictionary mapping kernel functions to their parameter lists.
        current_params: Dictionary of current kernel parameters for each function.
        const_param: Constant parameters that don't change during simulation.
    """
    def __init__(self, instance, kernel_funcs: List[Callable[..., Any]], 
                 result: np.ndarray = None, const_param: Dict[Callable, Dict] = None):
        for key, val in const_param.items(): self.__setattr__(key, val)
        
        self.instance = instance
        self.kernel_funcs: Dict[Callable, list] = {}
        self.const_param = const_param
        self.result = result

        # Validate and process kernel functions
        for kf in kernel_funcs:
            kernel_params = inspect.signature(kf).parameters.keys()
            self.kernel_funcs[kf] = [param for param in kernel_params]

    def get_current_params(self) -> Dict[Callable, Dict[str, Any]]:
        """
        Generates a dictionary of current kernel parameters based on the simulation state.
        
        Returns:
            A dictionary mapping each kernel function to its resolved parameter dictionary.
        """
        current_params: Dict[Callable, list] = {}

        for kernel_func, params in self.kernel_funcs.items():
            param_dict = {}

            for param in params:
                if param == 'result': 
                    # result is always tied to self.result (observables) and is unique
                    param_dict[param] = self.result; continue 
                
                if param == 'idx':
                    # idx is always tied to self.n_cells (parallel for loop)
                    param_dict[param] = self.instance.n_cells; continue 
                
                

                # check if param is instance of sim (eg. field)
                elif hasattr(self.instance, param): param_dict[param] = getattr(self.instance, param) 

                if self.const_param is not None:
                    if param in self.const_param.keys(): 
                        # constant parameters may be passed (eg. order of moment)
                        param_dict[param] = self.const_param[param]

            current_params[kernel_func] = param_dict
        return current_params
    


from collections import deque

import numpy as np


class RollingStats:
    """
    Computes rolling mean, variance, and error statistics over a sample stream.
    """

    def __init__(self):
        self.sample_counter = 0
        self.shape = None
        self.rolling_mean = None
        self.rolling_sqr_mean = None
        self.data = []

    def update(self, next_sample_array: np.ndarray) -> None:
        """
        Update the rolling statistics with the next sample.
        Dynamically determines shape on the first update.
        """
        # if np.isnan(next_sample_array).all():
        #     return 
        if self.sample_counter == 0:
            self.shape = next_sample_array.shape
            self.rolling_mean = next_sample_array.copy()
            self.rolling_sqr_mean = np.power(abs(next_sample_array), 2)
        else:
            if next_sample_array.shape != self.shape:
                raise ValueError(
                    f"Shape of new sample ({next_sample_array.shape}) does not match ({self.shape})!"
                )
            self.rolling_mean += next_sample_array
            self.rolling_sqr_mean += np.power(abs(next_sample_array), 2)

        self.sample_counter += 1
        # self.store_data(next_sample_array)

    def store_data(self, next_sample_array: np.ndarray) -> None:
        """
        Optionally stores the raw data for additional analysis.
        """
        if self.data is not None:
            self.data.append(next_sample_array.copy())

    def get_data(self) -> list:
        """
        Return all stored data samples.
        """
        return self.data

    def get_rolling_mean(self) -> np.ndarray:
        """
        Return the rolling mean of the data samples.
        """
        if self.sample_counter == 0:
            raise ValueError("No samples provided yet.")

        return self._to_numpy(self.rolling_mean / self.sample_counter)

    def get_rolling_std(self) -> np.ndarray:
        """
        Return the rolling standard deviation.
        """
        if self.sample_counter == 0:
            raise ValueError("No samples provided yet.")

        rolling_mean = self.rolling_mean / self.sample_counter
        rolling_variance = (
            self.rolling_sqr_mean / self.sample_counter - np.power(abs(rolling_mean), 2)
        )
        rolling_std = np.sqrt(rolling_variance)
        return self._to_numpy(rolling_std)

    def get_rolling_err_mean(self) -> np.ndarray:
        """
        Return the rolling relative error of the mean.
        """
        if self.sample_counter == 0:
            raise ValueError("No samples provided yet.")

        rolling_mean = self.rolling_mean / self.sample_counter
        rolling_variance = (
            self.rolling_sqr_mean / self.sample_counter - np.power(abs(rolling_mean), 2)
        )
        rolling_err_mean = np.sqrt(rolling_variance) / np.sqrt(self.sample_counter)
        return self._to_numpy(rolling_err_mean)

    def reset(self) -> None:
        """
        Reset the rolling statistics to start over.
        """
        self.sample_counter = 0
        self.shape = None
        self.rolling_mean = None
        self.rolling_sqr_mean = None
        self.data = []

    
    def _to_numpy(self, array: np.ndarray) -> np.ndarray:
        """
        Convert to numpy array, handling GPU support if enabled.
        """
        return np.array(array)