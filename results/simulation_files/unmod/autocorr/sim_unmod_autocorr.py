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
os.environ["MY_NUMBA_TARGET"] = "numba"
 
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
    update_histogram_real, 
    calculate_stats_complex,
)

if use_cuda: from numba import cuda
from time import time

# define simulation run
def run_sim(params):
    sigma = params["sigma_abs"] * np.exp(1j*params["sigma_phase"])
    interaction = params["lambda_abs"]
    dt = parameters["dt"]
    trajs = parameters["trajs"]
    max_langevin_time = parameters["max_langevin_time"]
    mean_dS_max = parameters["mean_dS_max"]
    auto_corr = parameters["auto_corr"]

    config = Config(
                    dt = dt, 
                    trajs = trajs, 
                    sigma = sigma, 
                    interaction = interaction, 
                    mean_dS_max = mean_dS_max,
                    max_langevin_time = max_langevin_time
                    )
    
    sim = ComplexLangevinSimulation(config)

    # bins_p = params["bins_p"]  # Number of p bins
    # bins_u = params["bins_u"]  # Number of u bins
    # hist_p = np.zeros((bins_p, bins_p), dtype=scal.SCAL_TYPE_REAL)
    # hist_p = np.zeros(bins_u, dtype=scal.SCAL_TYPE_REAL)
    
    if use_cuda:
        # hist_p = cuda.to_device(hist_p)
        # hist_u = cuda.to_device(hist_u)
        gpu_handler = GPU_handler(sim)
        gpu_handler.to_device()

    # reigster observables
    sim.register_observable('1_moment', obs_kernel=n_moment_kernel, const_param={"order" : 1},  langevin_history_full=True, thermal_time=0, auto_corr=auto_corr, maximal_lt=max_langevin_time)
    sim.register_observable('2_moment', obs_kernel=n_moment_kernel, const_param={"order" : 2},  langevin_history_full=True, thermal_time=0, auto_corr=auto_corr, maximal_lt=max_langevin_time)
    
    sim.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 1}, langevin_history_full=True, thermal_time=0, auto_corr=auto_corr, maximal_lt=max_langevin_time)
    sim.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 3}, langevin_history_full=True, thermal_time=0, auto_corr=auto_corr, maximal_lt=max_langevin_time)
    # sim.register_observable('abs_drift', obs_kernel=abs_drift, langevin_history=False, thermal_time=10, auto_corr=2, maximal_lt=stop_lt)
    
    # run the sim,
    if use_cuda: 
        import cupy as cp
        def min_lt():
            return cp.min(cp.asarray(sim.langevin_time)[cp.asarray(sim.alive)])
    else: 
        def min_lt():
            return np.min(sim.langevin_time[sim.alive])
        
    with tqdm(total=max_langevin_time) as pbar:
        while min_lt() < max_langevin_time:
            pbar.update(float(min_lt())-pbar.n)
            sim.step()
            sim.kill_trajs()
            for name, tr in sim.trackers.items():
                tr.mark_equilibrated_trajs()
                tr.compute()

            # update histograms
            # p_tracker = sim.trackers["1_moment"]
            # u_tracker = sim.trackers["1_moment"]
            # my_act_parallel_loop(update_histogram_complex, p_tracker.equilibrated_trajs, trajs, p_tracker.result, hist_p, bins_p, params["p_min_real"], params["p_max_real"], params["p_min_imag"], params["p_max_imag"])
            # my_act_parallel_loop(update_histogram_real, u_tracker.equilibrated_trajs, trajs, u_tracker.result, hist_u, bins_u, params["u_min"], params["u_max"])
            
            if use_cuda: cuda.synchronize()
    
    sim.finish()
    # if use_cuda:
    #     hist_p = hist_p.copy_to_host()
    #     hist_u = hist_u.copy_to_host()

    # write results in dict
    if use_cuda: alive = float(cp.sum(cp.asarray(sim.alive)))
    else: alive = np.sum(sim.alive)
    results = {'params': params}
    print(type(alive))
    results["params"]["alive"] = int(alive)

    for name, tr in sim.trackers.items():
        mean, variance_real, variance_imag, covariance_reim, counter = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean_real, tr.rolling_sqr_mean_imag, tr.rolling_sqr_mean_cross, tr.counter)
        
        results[name] = {}
        results[name]["mean_real"] = mean.real
        results[name]["mean_imag"] = mean.imag
        results[name]["variance_real"] = variance_real
        results[name]["variance_imag"] = variance_imag
        results[name]["covariance_reim"] = covariance_reim
        results[name]["counter"] = float(counter)
        results[name]["thermal_time"] = tr.thermal_time
        results[name]["auto_corr"] = tr.auto_corr

    print(f"Steps: {sim.langevin_steps}")
    import random, string


    base_path = os.path.join(os.getenv("HOME"), "gitrepos", "clonscal", "results", "sim_data", "unmod", "autocorr", )
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

    # save langevin history
    for name, tr in sim.trackers.items():
        if tr.langevin_history_full:
            res = tr.history_result_full  # Retrieve result array
            lt = tr.history_meas_times_full  # Retrieve measurement times array

            # Save corresponding npz file
            npz_filename = f"{param_key}_{name}.npz"
            npz_path = os.path.join(base_path, npz_filename)
            np.savez(npz_path, result=res, meas_times=lt)

            print(f"Saved NPZ: {npz_path}")
    


    # npy_filename = f"{param_key}_hist_p.npy"
    # npy_path = os.path.join(base_path, npy_filename)
    # np.save(npy_path, hist_p)

################################################################################################
################################################################################################
parameters = {
        "trajs": int(1e3),
        "mean_dS_max": 50,
        "max_langevin_time": 1,
        "lambda_abs": 1,
    }

sigma = -1+1j
parameters["sigma_abs"] = np.abs(sigma)
parameters["sigma_phase"] = np.angle(sigma)

for dt in [5e-2, 1e-3, 5e-4, 1e-4]:
    parameters["dt"] = dt
    parameters["auto_corr"] = dt/2
    for key, value in parameters.items():
        print(key, "->", value)
    run_sim(parameters)
