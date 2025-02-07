import numpy as np

# Set environment variables
import os

os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "single"
os.environ["MY_NUMBA_TARGET"] = "numba"
 
# Add cle_fun to PYTHON_PATH
import sys
sys.path.append("../../clonscal")

# create parameters
start_angle = 0
stop_angle = 2 * np.pi
angles_cl = np.linspace(start_angle, stop_angle, 32)

pullback_phase = np.linspace(start_angle, stop_angle, 1)

parameters = {
    "steps": [int(1e5)],
    "trajs": [int(1e5)],
    "dt": [1e-4],
    "sigma_abs": [1],
    "sigma_phase": angles_cl,
    "lambda_abs": [1],
    "pullback_phase": pullback_phase,
    "pullback_abs": [5],
    "mass_modification": [2],
}

# Generate all parameter combinations
import itertools
param_combinations = [dict(zip(parameters.keys(), values)) for values in itertools.product(*parameters.values())]


# Running the simulation
from simulation.config import Config
from simulation.cl_simulation import ComplexLangevinSimulation
from src.obs_kernels import n_moment_kernel, dse_n_moment_kernel
from src.utils import gaussian_modified_density_drift_kernel
from tqdm import tqdm

def run_sim(params):
    param_key = "_".join(map(str, params.values()))
    sigma = params["sigma_abs"] * np.exp(1j*params["sigma_phase"])
    pullback = params["pullback_abs"] * np.exp(1j*params["pullback_phase"])
    interaction = params["lambda_abs"]

    config = Config(dt = params["dt"], 
                    trajs = params["trajs"], 
                    dims = [1], 
                    mass_real = sigma, 
                    interaction = interaction, 
                    ada_step = True, 
                    drift_kernel=gaussian_modified_density_drift_kernel,
                    mass_modification=params["mass_modification"], 
                    pullback = pullback
                    )
    
    sim = ComplexLangevinSimulation(config)
    sim.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={"order" : 1},  langevin_history=False, thermal_time=1, auto_corr=1)
    sim.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={"order" : 3},  langevin_history=False, thermal_time=1, auto_corr=1)
    sim.register_observable('dse_5_moment', obs_kernel=dse_n_moment_kernel, const_param={"order" : 5},  langevin_history=False, thermal_time=1, auto_corr=1)

    for _ in tqdm(range(params["steps"])):
        sim.step()
        for name, tr in sim.trackers.items():
            tr.mark_equilibrated_trajs()
            tr.compute()
    sim.finish()

    results = {'params': params}
    for name, tr in sim.trackers.items():
        mean, sem = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean, tr.counter)
        results[name] = {}
        results[name]["mean_real"] = mean.real
        results[name]["mean_imag"] = mean.imag
        results[name]["sem"] = sem
        
    return results, param_key


for param in param_combinations[:2]:
    results, param_key = run_sim(param)

    import json
    file_path = os.path.join('$HOME/gitrepos/clonscal/parameter_study', f"{param_key}.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
