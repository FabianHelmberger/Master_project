import math
import numpy as np
from typing import TYPE_CHECKING

from src.numba_target import myjit
import src.scal as scal

if TYPE_CHECKING:
    # Import only for type checking
    from simulation.langevin_dynamics import LangevinDynamics
    from simulation.cl_simulation import ComplexLangevinSimulation


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
def update_noise_kernel(idx, eta, dt):
    eta[idx] = scal.SCAL_TYPE_REAL(math.sqrt(dt) * np.random.normal())


@myjit
def evolve_kernel(idx, phi0, phi1, dS, eta, dt, **kwargs):
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


# @myjit
# def mexican_hat_kernel_real(idx, obj: "LangevinDynamics"):
#     phi_idx = obj.field.phi0[idx]
#     out = 0
#     out += obj.config.mass_real * phi_idx
#     out += obj.config.interaction/6 * phi_idx*phi_idx*phi_idx
#     obj.field.dS[idx] = out

@myjit
def mexican_hat_kernel_complex(idx, field, dS_out, mass_real, interaction):
    phi_idx = field[idx]
    out = 0
    out += mass_real * phi_idx
    out += interaction/6 * phi_idx*phi_idx*phi_idx
    dS_out[idx] = out

# def arg_map(sim: "ComplexLangevinSimulation"):
#     return {
#         'iter_max': sim.lattice.n_cells, 
#         'field': sim.field.phi0, 
#         'dS_out': sim.field.dS, 
#         'mass_real' : sim.config.mass_real, 
#         'mass_imag' : sim.config.mass_imag, 
#         'interaction' : sim.config.interaction
#         }

# kernel_args = {}
# kernel_args[mexican_hat_kernel_real] = arg_map()['iter_max',]