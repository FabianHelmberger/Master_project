import numpy as np
from tqdm import tqdm
from  datetime import datetime

# Add cle_fun to PYTHON_PATH
import sys
sys.path.append("../../clonscal")

# Set environment variables
import os, json
os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "single"
os.environ["MY_NUMBA_TARGET"] = "numba"
 
from simulation.config import Config
from simulation.cl_simulation import ComplexLangevinSimulation
from simulation.gpu_handler import GPU_handler

from src.obs_kernels import (
    n_moment_kernel, 
    dse_n_moment_kernel,
    abs_drift
)

import src.scal as scal
from src.numba_target import (
    use_cuda,
    my_act_parallel_loop
)
from src.utils import (
    update_histogram_complex, 
    update_histogram_real,
    calculate_stats_complex,
)

if use_cuda: from numba import cuda

# define simulation run
def run_sim(params):
    sigma = params["sigma_abs"] * np.exp(1j*params["sigma_phase"])
    interaction = params["lambda_abs"]

    config = Config(
                    steps = params["steps"],
                    dt = params["dt"], 
                    trajs = params["trajs"], 
                    dims = [1], 
                    sigma = sigma, 
                    interaction = interaction, 
                    ada_step = True,
                    mean_dS_max = params["mean_dS_max"],
                    )
    
    sim = ComplexLangevinSimulation(config)

    bins_p = params["bins_p"]  # Number of p bins
    bins_u = params["bins_u"]  # Number of u bins
    hist_p = np.zeros((bins_p, bins_p), dtype=scal.SCAL_TYPE_REAL)
    hist_u = np.zeros(bins_u, dtype=scal.SCAL_TYPE_REAL)
    
    if use_cuda:
        hist_u = cuda.to_device(hist_u)
        hist_p = cuda.to_device(hist_p)
        gpu_handler = GPU_handler(sim)
        gpu_handler.to_device()

    # reigster observables
    sim.register_observable('1_moment', obs_kernel=n_moment_kernel, const_param={"order" : 1},  langevin_history=True, thermal_time=0, auto_corr=0)
    sim.register_observable('2_moment', obs_kernel=n_moment_kernel, const_param={"order" : 2},  langevin_history=True, thermal_time=0, auto_corr=0)
    sim.register_observable('3_moment', obs_kernel=n_moment_kernel, const_param={"order" : 3},  langevin_history=True, thermal_time=0, auto_corr=0)
    
    sim.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 1}, langevin_history=True, thermal_time=0, auto_corr=0)
    sim.register_observable('dse_2_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 2}, langevin_history=True, thermal_time=0, auto_corr=0)
    sim.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 3}, langevin_history=True, thermal_time=0, auto_corr=0)
    sim.register_observable('abs_drift', obs_kernel=abs_drift, langevin_history=False, thermal_time=0, auto_corr=0)
    
    # run the sim
    for idx in tqdm(range(params["steps"])):
        sim.step()
        for name, tr in sim.trackers.items():
            tr.mark_equilibrated_trajs()
            tr.compute()

        # update histograms
        p_tracker = sim.trackers["1_moment"]
        u_tracker = sim.trackers["abs_drift"]
        my_act_parallel_loop(update_histogram_complex, p_tracker.equilibrated_trajs, params["trajs"], p_tracker.result, hist_p, params["bins_p"], params["p_min_real"], params["p_max_real"], params["p_min_imag"], params["p_max_imag"])
        my_act_parallel_loop(update_histogram_real, u_tracker.equilibrated_trajs, params["trajs"], u_tracker.result, hist_u, params["bins_u"], params["u_min"], params["u_max"])
        if use_cuda: cuda.synchronize()
    
    sim.finish()
    if use_cuda:
        hist_u = hist_u.copy_to_host()
        hist_p = hist_p.copy_to_host()

    # write results in dict
    results = {'params': params}
    for name, tr in sim.trackers.items():
        mean, sem_real, sem_imag = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean_real, tr.rolling_sqr_mean_imag, tr.counter)
        results[name] = {}
        results[name]["mean_real"] = mean.real
        results[name]["mean_imag"] = mean.imag
        results[name]["sem_real"] = sem_real
        results[name]["sem_imag"] = sem_imag
        results[name]["thermal_time"] = tr.thermal_time
        results[name]["auto_corr"] = tr.auto_corr
        
    
    import random, string

    # Define base directory
    base_path = os.path.join(os.getenv("HOME"), "gitrepos", "clonscal", "thermalization", "unmod_sim_data", "2500301")
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
    

    npy_filename = f"{param_key}_hist_u.npy"
    npy_path = os.path.join(base_path, npy_filename)
    np.save(npy_path, hist_p)

    npy_filename = f"{param_key}_hist_p.npy"
    npy_path = os.path.join(base_path, npy_filename)
    np.save(npy_path, hist_p)

################################################################################################
################################################################################################

steps = int(5e4)
trajs = int(1e3)
dt = 1e-3

sigma = 1+1j
parameters = {
        "steps": steps,
        "trajs": trajs,
        "dt": dt,
        "sigma_abs": np.abs(sigma),
        "sigma_phase": np.angle(sigma),
        "lambda_abs": 1,
        "bins_u": int(1e6),
        "bins_p": int(1e3),
        "p_min_real": -5,
        "p_max_real": 5,
        "p_min_imag": -5,
        "p_max_imag": 5,
        "u_min": 1e-3,
        "u_max": int(1e3),
        "mean_dS_max": 50,
    }

for key, value in parameters.items():
    print(key, "->", value)

run_sim(parameters)

################################################################################################
################################################################################################

steps = int(1e5)
trajs = int(1e3)
dt = 5e-4

sigma = 1+1j
parameters = {
        "steps": steps,
        "trajs": trajs,
        "dt": dt,
        "sigma_abs": np.abs(sigma),
        "sigma_phase": np.angle(sigma),
        "lambda_abs": 1,
        "bins_u": int(1e6),
        "bins_p": int(1e3),
        "p_min_real": -5,
        "p_max_real": 5,
        "p_min_imag": -5,
        "p_max_imag": 5,
        "u_min": 1e-3,
        "u_max": int(1e3),
        "mean_dS_max": 5,
    }

for key, value in parameters.items():
    print(key, "->", value)

run_sim(parameters)

################################################################################################
################################################################################################

steps = int(5e5)
trajs = int(1e3)
dt = 1e-4

sigma = 1+1j
parameters = {
        "steps": steps,
        "trajs": trajs,
        "dt": dt,
        "sigma_abs": np.abs(sigma),
        "sigma_phase": np.angle(sigma),
        "lambda_abs": 1,
        "bins_u": int(1e6),
        "bins_p": int(1e3),
        "p_min_real": -5,
        "p_max_real": 5,
        "p_min_imag": -5,
        "p_max_imag": 5,
        "u_min": 1e-3,
        "u_max": int(1e3),
        "mean_dS_max": 50,
    }

for key, value in parameters.items():
    print(key, "->", value)

run_sim(parameters)


################################################################################################
################################################################################################

steps = int(1e6)
trajs = int(1e3)
dt = 5e-5

sigma = 1+1j
parameters = {
        "steps": steps,
        "trajs": trajs,
        "dt": dt,
        "sigma_abs": np.abs(sigma),
        "sigma_phase": np.angle(sigma),
        "lambda_abs": 1,
        "bins_u": int(1e6),
        "bins_p": int(1e3),
        "p_min_real": -5,
        "p_max_real": 5,
        "p_min_imag": -5,
        "p_max_imag": 5,
        "u_min": 1e-3,
        "u_max": int(1e3),
        "mean_dS_max": 50,
    }

for key, value in parameters.items():
    print(key, "->", value)

run_sim(parameters)
################################################################################################
################################################################################################

steps = int(5e6)
trajs = int(1e3)
dt = 1e-5

sigma = 1+1j
parameters = {
        "steps": steps,
        "trajs": trajs,
        "dt": dt,
        "sigma_abs": np.abs(sigma),
        "sigma_phase": np.angle(sigma),
        "lambda_abs": 1,
        "bins_u": int(1e6),
        "bins_p": int(1e3),
        "p_min_real": -5,
        "p_max_real": 5,
        "p_min_imag": -5,
        "p_max_imag": 5,
        "u_min": 1e-3,
        "u_max": int(1e3),
        "mean_dS_max": 50,
    }

for key, value in parameters.items():
    print(key, "->", value)

run_sim(parameters)

################################################################################################
################################################################################################

steps = int(1e7)
trajs = int(1e3)
dt = 1e-6

sigma = 1+1j
parameters = {
        "steps": steps,
        "trajs": trajs,
        "dt": dt,
        "sigma_abs": np.abs(sigma),
        "sigma_phase": np.angle(sigma),
        "lambda_abs": 1,
        "bins_u": int(1e6),
        "bins_p": int(1e3),
        "p_min_real": -5,
        "p_max_real": 5,
        "p_min_imag": -5,
        "p_max_imag": 5,
        "u_min": 1e-3,
        "u_max": int(1e3),
        "mean_dS_max": 50,
    }

for key, value in parameters.items():
    print(key, "->", value)

run_sim(parameters)