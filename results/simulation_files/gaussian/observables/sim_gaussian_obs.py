import numpy as np
from tqdm import tqdm
from  datetime import datetime

# Add cle_fun to PYTHON_PATH
import sys
sys.path.append("../../../../../clonscal")

# Set environment variables
import os, json
os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "single"
os.environ["MY_NUMBA_TARGET"] = "cuda"
 
from simulation.config import Config
from simulation.cl_simulation import ComplexLangevinSimulation
from simulation.gpu_handler import GPU_handler

from src.obs_kernels import (
    n_moment_kernel, 
    dse_n_moment_kernel,
)

import src.scal as scal
from src.numba_target import (
    use_cuda,
    my_act_parallel_loop
)
from src.utils import (
    update_histogram_complex, 
    calculate_stats_complex,
    gaussian_modified_density_drift_kernel
)

if use_cuda: from numba import cuda
from time import time

# define simulation run
def run_sim(params):
    sigma = params["sigma_abs"] * np.exp(1j*params["sigma_phase"])
    interaction = params["lambda_abs"]
    mass_modification = params["mass_modification"]
    pullback = params["pullback"]
    dt = parameters["dt"]
    trajs = parameters["trajs"]
    stop_lt = parameters["stop_lt"]
    steps = int(50*stop_lt/dt)

    config = Config(
                    steps = steps,
                    dt = params["dt"], 
                    trajs = trajs, 
                    dims = [1], 
                    sigma = sigma, 
                    interaction = interaction, 
                    mass_modification = mass_modification, 
                    pullback = pullback, 
                    ada_step = True,
                    mean_dS_max = params["mean_dS_max"],
                    drift_kernel = gaussian_modified_density_drift_kernel,
                    )
    
    sim = ComplexLangevinSimulation(config)

    bins_p = params["bins_p"]  # Number of p bins
    hist_p = np.zeros((bins_p, bins_p), dtype=scal.SCAL_TYPE_REAL)
    
    if use_cuda:
        hist_p = cuda.to_device(hist_p)
        gpu_handler = GPU_handler(sim)
        gpu_handler.to_device()

    # reigster observables
    sim.register_observable('1_moment', obs_kernel=n_moment_kernel, const_param={"order" : 1},  langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    sim.register_observable('2_moment', obs_kernel=n_moment_kernel, const_param={"order" : 2},  langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    sim.register_observable('3_moment', obs_kernel=n_moment_kernel, const_param={"order" : 3},  langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    sim.register_observable('4_moment', obs_kernel=n_moment_kernel, const_param={"order" : 4},  langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    
    sim.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 1}, langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    sim.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 3}, langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    # sim.register_observable('abs_drift', obs_kernel=abs_drift, langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    
    # run the sim,
    import cupy as cp
    min_lt = 0
    with tqdm(total=stop_lt+1) as pbar:
        while cp.min(cp.asarray(sim.langevin_time)[cp.asarray(sim.alive)])<stop_lt:
            pbar.update(float(cp.min(cp.asarray(sim.langevin_time)[cp.asarray(sim.alive)]))-pbar.n)
            sim.step()
            sim.kill_trajs()
            for name, tr in sim.trackers.items():
                tr.mark_equilibrated_trajs()
                tr.compute()

            # update histograms
            p_tracker = sim.trackers["1_moment"]
            my_act_parallel_loop(update_histogram_complex, p_tracker.equilibrated_trajs, params["trajs"], p_tracker.result, hist_p, params["bins_p"], params["p_min_real"], params["p_max_real"], params["p_min_imag"], params["p_max_imag"])
            if use_cuda: cuda.synchronize()
    
    sim.finish()
    if use_cuda:
        hist_p = hist_p.copy_to_host()

    # write results in dict
    results = {'params': params}
    for name, tr in sim.trackers.items():

        mean, variance_real, variance_imag, covariance_reim, counter = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean_real, tr.rolling_sqr_mean_imag, tr.rolling_sqr_mean_cross, tr.counter)
        
        results[name] = {}
        results[name]["mean_real"] = mean.real
        results[name]["mean_imag"] = mean.imag
        results[name]["variance_real"] = variance_real
        results[name]["variance_imag"] = variance_imag
        results[name]["covariance_reim"] = covariance_reim
        results[name]["counter"] = float(counter)
        # results[name]["angle"] = angle
        results[name]["thermal_time"] = tr.thermal_time
        results[name]["auto_corr"] = tr.auto_corr

    print(f"Steps: {sim.langevin_steps}")
    import random, string


    base_path = os.path.join(os.getenv("HOME"), "gitrepos", "clonscal", "results", "sim_data", "gaussian_data", "observables", )
    # Define base directory
    os.makedirs(base_path, exist_ok=True)  # Create directory if it doesn't exist

    # Generate a random param_key
    def generate_param_key():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        return f"{timestamp}_{rand_str}"

    param_key = generate_param_key()

    # Save JSON file
    json_path = os.path.join(base_path, f"{param_key}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved JSON: {json_path}")

    # save langevin history (mean over trajs)
    for name, tr in sim.trackers.items():
        if tr.langevin_history:
            res = tr.history_result  # Retrieve result array
            lt = tr.history_meas_times  # Retrieve measurement times array

            # Save corresponding npz file
            npz_filename = f"{param_key}_{name}.npz"
            npz_path = os.path.join(base_path, npz_filename)
            np.savez(npz_path, result=res, meas_times=lt)

            print(f"Saved NPZ: {npz_path}")
    


    npy_filename = f"{param_key}_hist_p.npy"
    npy_path = os.path.join(base_path, npy_filename)
    np.save(npy_path, hist_p)

################################################################################################
################################################################################################
# unbroken symmetry
# study different step sizes (1e-3 - 1e-6)
lamb = 2


parameters = {
        "trajs": int(1e6),
        "dt": None,
        "sigma_abs": None,
        "sigma_phase": None,
        "lambda_abs": lamb,
        "mass_modification": None,
        "pullback": None,
        "bins_p": int(1e3),
        "p_min_real": -10,
        "p_max_real": 10,
        "p_min_imag": -10,
        "p_max_imag": 10,
        "mean_dS_max": 50,
        "stop_lt": 100
    }


SIGMAS = [-1+4j, 1+1j, -1+2j, 1+2j, -1+3j, 1+3j, -1+4j, 1+4j]
MAX_LTS = [50, 60, 70]
LAMBDAS = [1,2]

import argparse
import itertools
# Generate all possible parameter combinations
param_permutations = list(itertools.product(SIGMAS, LAMBDAS, MAX_LTS))




parser = argparse.ArgumentParser()
parser.add_argument("--slurm_idx", type=int, required=True)
args = parser.parse_args()


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
sigma = -1+1j
parameters["sigma_abs"] = np.abs(sigma)
parameters["sigma_phase"] = np.angle(sigma)
parameters["mass_modification"] = 2.88888
crit_r = 3.906250

pullbacks = [crit_r/2, crit_r-1, crit_r+1, crit_r*2]
parameters["pullback"] =  pullbacks[args.slurm_idx]

for dt in [5e-3, 1e-3, 5e-4, 1e-4]:
    parameters["dt"] = dt
    for key, value in parameters.items():
        print(key, "->", value)
    run_sim(parameters)
    
###############################################################################################
# sigma = -1+2j
# parameters["sigma_abs"] = np.abs(sigma)
# parameters["sigma_phase"] = np.angle(sigma)
# parameters["mass_modification"] = 1.425
# crit_r = 6.4375

# # for pullback in [crit_r/2, crit_r-1, crit_r+1, crit_r*2]:
# for pullback in [crit_r/2]:
#     parameters["pullback"] =  pullback

#     for dt in [5e-3, 1e-3, 5e-4, 1e-4]:
#         parameters["dt"] = dt
#         for key, value in parameters.items():
#             print(key, "->", value)
#         run_sim(parameters)
# ###############################################################################################
# sigma = -1+3j
# parameters["sigma_abs"] = np.abs(sigma)
# parameters["sigma_phase"] = np.angle(sigma)
# parameters["mass_modification"] = 1.111111
# crit_r = 12.265625 

# for pullback in [crit_r/2, crit_r-1, crit_r+1, crit_r*2]:
#     parameters["pullback"] =  pullback

#     for dt in [5e-3, 1e-3, 5e-4, 1e-4]:
#         parameters["dt"] = dt
#         for key, value in parameters.items():
#             print(key, "->", value)
#         run_sim(parameters)
# ###############################################################################################
# sigma = -1+4j
# parameters["sigma_abs"] = np.abs(sigma)
# parameters["sigma_phase"] = np.angle(sigma)
# parameters["mass_modification"] = 1.222222
# crit_r = 19.140625

# for pullback in [crit_r/2, crit_r-1, crit_r+1, crit_r*2]:
#     parameters["pullback"] =  pullback

#     for dt in [5e-3, 1e-3, 5e-4, 1e-4]:
#         parameters["dt"] = dt
#         for key, value in parameters.items():
#             print(key, "->", value)
#         run_sim(parameters)
# ###############################################################################################
