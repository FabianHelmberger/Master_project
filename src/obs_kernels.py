from src.numba_target import myjit
from typing import TYPE_CHECKING

import src.scal as scal

if TYPE_CHECKING:
    # Import only for type checking
    from simulation.langevin_dynamics import LangevinDynamics
    from simulation.cl_simulation import ComplexLangevinSimulation

@myjit
def langevin_time(sim: 'ComplexLangevinSimulation', result):
    result = sim.dt_ada

@myjit
def n_moment_kernel(idx, phi0, result, order, langevin_time, adims, meas_time):
    phi_idx = phi0[idx]
    res = 1
    for _ in range(order):
        res *= phi_idx
    result[idx] = res
    traj_idx = idx // adims[1]
    meas_time[traj_idx] = langevin_time[traj_idx]

@myjit
def abs_drift(idx, result, dS_norm, langevin_time, meas_time):
    result[idx] = dS_norm[idx]
    meas_time[idx] = langevin_time[idx]




@myjit
def skew_action(idx, phi0, result, mass_real, interaction, langevin_time, adims, meas_time):
    """
    dyson-schwinger eq. predict, that the exp. value wrt. the unmodified theory of this skew action is zero
    calculates sigma*phi+lambda*phi**3
    """
    phi_idx = phi0[idx]
    res = mass_real * phi_idx*phi_idx + interaction * phi_idx*phi_idx*phi_idx*phi_idx - 1

    result[idx] = res
    traj_idx = idx // adims[1]
    meas_time[traj_idx] = langevin_time[traj_idx]

@myjit
def dse_n_moment_kernel(idx, phi0, result, order, mass_real, interaction, langevin_time, adims, meas_time):
    """
    dyson-schwinger eq. predict, that the exp. value wrt. the unmodified theory of this skew action is one
    calculates sigma*phi+lambda*phi**3
    """
    phi_idx = phi0[idx]
    # phi_n = 1
    # phi_n_1 = 1
    # for k in range(order):
    #     phi_n *= phi_idx
    #     if k > 1 : phi_n_1 *= phi_idx
    
    res = order*phi_idx**(order-1)-phi_idx**order * (mass_real * phi_idx + interaction * phi_idx*phi_idx*phi_idx)

    result[idx] = res
    traj_idx = idx // adims[1]
    meas_time[traj_idx] = langevin_time[traj_idx]





@myjit
def test_kernel(idx, result, constant_param_1):
    result[idx] = constant_param_1

@myjit
def langevin_time(idx, result, langevin_time):
    result[idx] = langevin_time