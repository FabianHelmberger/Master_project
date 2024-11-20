import math
import numpy as np
import inspect
from typing import TYPE_CHECKING, Callable, Dict, Any, List

import src.scal as scal
from src.numba_target import myjit


if TYPE_CHECKING:
    # Import only for type checking
    from simulation.langevin_dynamics import LangevinDynamics
    from simulation.cl_simulation import ComplexLangevinSimulation

from simulation.constants import SQRT2

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
def get_index(pos, dims):
    index = pos[0]
    for d in range(1, len(dims)):
        index = index * dims[d] + pos[d]
    return index

@myjit
def noise_kernel(idx, eta):
    eta[idx] = SQRT2 * scal.SCAL_TYPE_REAL(np.random.normal())

@myjit
def evolve_kernel(idx, phi0, phi1, dS, eta, dt):
    # TODO: move dt_sqrt
    dt_sqrt = math.sqrt(dt)
    etaterm = eta[idx] * dt_sqrt
    update = etaterm - dt * dS[idx]
    phi1[idx] = phi0[idx] + update


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
def mexican_hat_kernel_real(idx, phi0, dS, mass_real, interaction):
    phi_idx = phi0[idx]
    out = 0
    out += mass_real * phi_idx
    out += interaction/6 * phi_idx*phi_idx*phi_idx
    dS[idx] = out


@myjit
def mexican_hat_kernel_complex(idx, field, dS_out, mass_real, interaction):
    phi_idx = field[idx]
    out = 0
    out += mass_real * phi_idx
    out += interaction/6 * phi_idx*phi_idx*phi_idx
    dS_out[idx] = out


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
    def __init__(self, sim: 'LangevinDynamics', kernel_funcs: List[Callable[..., Any]], 
                 result: np.ndarray = None, const_param: Dict[Callable, Dict] = None,):
        self.sim = sim
        self.kernel_funcs: Dict[Callable, list] = {}
        # self.const_param:  Dict[Callable, Dict] = {}
        self.const_param = const_param
        self.result = result

        # Validate and process kernel functions
        for kf in kernel_funcs:
            kernel_params = inspect.signature(kf).parameters.keys()
            self.kernel_funcs[kf] = [param for param in kernel_params]

            # fill constant parameters for the current kernel function
            # if kf in const_param: 
            #     self.const_param[kf] = const_param[kf]


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
                    param_dict[param] = self.sim.n_cells; continue 
                # check if param is instance of sim (eg. field)
                elif hasattr(self.sim, param): param_dict[param] = getattr(self.sim, param) 

                if self.const_param is not None:
                    if param in self.const_param.keys(): 
                        # constant parameters may be passed (eg. order of moment)
                        param_dict[param] = self.const_param[param]

            current_params[kernel_func] = param_dict
        return current_params