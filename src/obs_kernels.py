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
def n_moment_kernel(idx, phi0, result, order):
    phi_idx = phi0[idx]
    res = 1
    for _ in range(order):
        res *= phi_idx
    result[idx] = res

@myjit
def test_kernel(idx, result, constant_param_1):
    result[idx] = constant_param_1

@myjit
def langevin_time(idx, result, langevin_time):
    result[idx] = langevin_time