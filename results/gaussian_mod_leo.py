# import numpy as np
# import matplotlib.pyplot as plt

# # Set environment variables
# import os


# os.environ["SCAL_TYPE"] = "complex"
# os.environ["PRECISION"] = "single"
# os.environ["MY_NUMBA_TARGET"] = "cuda"
 
 
# # Add cle_fun to PYTHON_PATH
# import sys
# sys.path.append("../clonscal")

# ##########################################################
# ##########################################################

# import numpy as np

# def calculate_stats_complex(rolling_mean, rolling_sqr_mean_real, rolling_sqr_mean_imag, counter):
#     """
#     Calculate mean and SEM for complex numbers with separate variance tracking.

#     Parameters:
#     rolling_mean (np.array): Single-element array containing the rolling sum of complex values.
#     rolling_sqr_mean_real (np.array): Single-element array containing the rolling sum of squared real parts.
#     rolling_sqr_mean_imag (np.array): Single-element array containing the rolling sum of squared imaginary parts.
#     counter (np.array): Single-element array containing the number of values.

#     Returns:
#     tuple: (mean, sem_real, sem_imag), where:
#         - mean is the complex mean,
#         - sem_real is the SEM for the real part,
#         - sem_imag is the SEM for the imaginary part.
#     """
#     if counter == 0:
#         raise ValueError("Counter cannot be zero to avoid division by zero.")

#     # Extract real and imaginary parts
#     rolling_mean_real = rolling_mean.real
#     rolling_mean_imag = rolling_mean.imag

#     # Compute means
#     mean_real = rolling_mean_real / counter
#     mean_imag = rolling_mean_imag / counter

#     # Compute variances separately for real and imaginary parts
#     variance_real = (rolling_sqr_mean_real / counter) - mean_real**2
#     variance_imag = (rolling_sqr_mean_imag / counter) - mean_imag**2

#     # Ensure variances are non-negative
#     variance_real = max(variance_real, 0)
#     variance_imag = max(variance_imag, 0)

#     # Compute standard errors separately
#     sem_real = np.sqrt(variance_real / counter)
#     sem_imag = np.sqrt(variance_imag / counter)

#     # Return complex mean and separate SEMs
#     mean = mean_real + 1j * mean_imag
#     return mean, sem_real, sem_imag



# ##########################################################
# ##########################################################

# import numpy as np
# from simulation.config import Config
# from simulation.cl_simulation import ComplexLangevinSimulation
# from src.obs_kernels import (
#     n_moment_kernel, 
#     dse_n_moment_kernel,
#     abs_drift
# )
# from tqdm import tqdm
# from simulation.gpu_handler import GPU_handler
# from src.numba_target import use_cuda
# import src.scal as scal
# from src.utils import (
#     gaussian_modified_density_drift_kernel, 
#     update_histogram_complex, 
#     update_histogram_real,
#     mexican_hat_kernel_real, 
#     noise_kernel_rotated
# )

# from src.numba_target import my_act_parallel_loop
# from numba import cuda


# def run_sim(params):

#     param_key = "_".join(map(str, params.values()))
#     sigma = params["sigma_abs"] * np.exp(1j*params["sigma_phase"])
#     interaction = params["lambda_abs"]

#     config = Config(dt = params["dt"], 
#                     trajs = params["trajs"], 
#                     dims = [1], 
#                     mass_real = sigma, 
#                     interaction = interaction, 
#                     ada_step = True, 
#                     drift_kernel=gaussian_modified_density_drift_kernel,
#                     mass_modification=params["mass_modification"], 
#                     pullback=params["pullback"]
#                     )
    
#     sim_dse = ComplexLangevinSimulation(config)

#     bins_p = params["bins_p"]  # Number of p bins
#     bins_u = params["bins_u"]  # Number of u bins
#     hist_p = np.zeros((bins_p, bins_p), dtype=scal.SCAL_TYPE_REAL)
#     hist_u = np.zeros(bins_u, dtype=scal.SCAL_TYPE_REAL)
    
#     if use_cuda:
#         hist_u = cuda.to_device(hist_u)
#         hist_p = cuda.to_device(hist_p)
#         gpu_handler = GPU_handler(sim_dse)
#         gpu_handler.to_device()

#     # reigster observables
#     sim_dse.register_observable('1_moment', obs_kernel=n_moment_kernel, const_param={"order" : 1},  langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('2_moment', obs_kernel=n_moment_kernel, const_param={"order" : 2},  langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('3_moment', obs_kernel=n_moment_kernel, const_param={"order" : 3},  langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('4_moment', obs_kernel=n_moment_kernel, const_param={"order" : 4},  langevin_history=False, thermal_time=1, auto_corr=1)
    
#     sim_dse.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 1}, langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('dse_2_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 2}, langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 3}, langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('dse_4_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 4}, langevin_history=False, thermal_time=1, auto_corr=1)
#     sim_dse.register_observable('abs_drift', obs_kernel=abs_drift, langevin_history=False, thermal_time=1, auto_corr=1)

#     # run the sim
#     for _ in tqdm(range(params["steps"])):
#         sim_dse.step()
#         for name, tr in sim_dse.trackers.items():
#             tr.mark_equilibrated_trajs()
#             tr.compute()

#         # update histograms
#         p_tracker = sim_dse.trackers["1_moment"]
#         u_tracker = sim_dse.trackers["abs_drift"]
#         my_act_parallel_loop(update_histogram_complex, p_tracker.equilibrated_trajs, params["trajs"], p_tracker.result, hist_p, params["bins_p"], params["p_min_real"], params["p_max_real"], params["p_min_imag"], params["p_max_imag"])
#         my_act_parallel_loop(update_histogram_real, u_tracker.equilibrated_trajs, params["trajs"], u_tracker.result, hist_u, params["bins_u"], params["u_min"], params["u_max"])
#         if use_cuda: cuda.synchronize()
    
#     sim_dse.finish()
#     if use_cuda:
#         hist_u = hist_u.copy_to_host()
#         hist_p = hist_p.copy_to_host()

#     results = {'params': params}

#     for name, tr in sim_dse.trackers.items():
#         mean, sem_real, sem_imag = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean_real, tr.rolling_sqr_mean_imag, tr.counter)
#         results[name] = {}
#         results[name]["mean_real"] = mean.real
#         results[name]["mean_imag"] = mean.imag
#         results[name]["sem_real"] = sem_real
#         results[name]["sem_imag"] = sem_imag

#     # results["drift_hist"] = hist_u
#     # results["p_hist"] = hist_p

#     return results, param_key, hist_p, hist_u

# sigma = -1+4j
# parameters = {
#     "steps": [int(5e5)],
#     "trajs": [int(5e7)],
#     "dt": [5e-5],
#     "sigma_abs": [np.abs(sigma)],
#     "sigma_phase": [np.angle(sigma)],
#     "lambda_abs": [2],
#     "pullback": [200, 100, 50],
#     "mass_modification": [2.5, 5, 10],
#     "bins_u": [10000],
#     "bins_p": [1000],
#     "p_min_real": [-5],
#     "p_max_real": [5],
#     "p_min_imag": [-1],
#     "p_max_imag": [1],
#     "u_min": [1e-3],
#     "u_max": [1e3],
# }

# # Generate all parameter combinations
# import itertools
# param_combinations = [dict(zip(parameters.keys(), values)) for values in itertools.product(*parameters.values())]

# # Get index from Slurm
# slurm_index = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))  # Default to 0 if not running in Slurm

# if slurm_index >= len(param_combinations):
#     print(f"Invalid SLURM_ARRAY_TASK_ID={slurm_index}, max is {len(param_combinations)-1}")
#     sys.exit(1)

# # Select parameters for this job
# para = param_combinations[slurm_index]

# results, param_key, hist_p, hist_u = run_sim(para)
# # Save results
# import json
# file_path = os.path.join(os.getenv("HOME"), "gitrepos", "clonscal", "results", "gaussian_sim_data")
# with open(os.path.join(file_path, f"{param_key}.json"), "w") as f:
#     json.dump(results, f, indent=4)
# with open(os.path.join(file_path, f"{param_key}_hist_p.npy"), "wb") as f:
#     np.save(f, hist_p)
# with open(os.path.join(file_path, f"{param_key}_hist_u.npy"), "wb") as f:
#     np.save(f, hist_u)


import numpy as np
import matplotlib.pyplot as plt

# Set environment variables
import os


os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "single"
os.environ["MY_NUMBA_TARGET"] = "numba"
 
 
# Add cle_fun to PYTHON_PATH
import sys
sys.path.append("../clonscal")

##########################################################
##########################################################

import numpy as np



##########################################################
##########################################################

import numpy as np
from simulation.config import Config
from simulation.cl_simulation import ComplexLangevinSimulation
from src.obs_kernels import (
    n_moment_kernel, 
    dse_n_moment_kernel,
    abs_drift
)
from tqdm import tqdm
from simulation.gpu_handler import GPU_handler
from src.numba_target import use_cuda
import src.scal as scal
from src.utils import (
    update_histogram_complex, 
    update_histogram_real,
    gaussian_modified_density_drift_kernel, 
    noise_kernel_rotated,
    calculate_stats_complex
)

from src.numba_target import my_act_parallel_loop
from numba import cuda


def run_sim(params):

    param_key = "_".join(map(str, params.values()))
    sigma = params["sigma_abs"] * np.exp(1j*params["sigma_phase"])
    interaction = params["lambda_abs"]

    config = Config(dt = params["dt"], 
                    trajs = params["trajs"], 
                    dims = [1], 
                    sigma = sigma, 
                    interaction = interaction, 
                    ada_step = True, 
                    drift_kernel=gaussian_modified_density_drift_kernel,
                    mass_modification=params["mass_modification"], 
                    pullback=params["pullback"]
                    )
    
    sim_dse = ComplexLangevinSimulation(config)

    bins_p = params["bins_p"]  # Number of p bins
    bins_u = params["bins_u"]  # Number of u bins
    hist_p = np.zeros((bins_p, bins_p), dtype=scal.SCAL_TYPE_REAL)
    hist_u = np.zeros(bins_u, dtype=scal.SCAL_TYPE_REAL)
    
    if use_cuda:
        hist_u = cuda.to_device(hist_u)
        hist_p = cuda.to_device(hist_p)
        gpu_handler = GPU_handler(sim_dse)
        gpu_handler.to_device()

    # reigster observables
    sim_dse.register_observable('1_moment', obs_kernel=n_moment_kernel, const_param={"order" : 1},  langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('2_moment', obs_kernel=n_moment_kernel, const_param={"order" : 2},  langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('3_moment', obs_kernel=n_moment_kernel, const_param={"order" : 3},  langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('4_moment', obs_kernel=n_moment_kernel, const_param={"order" : 4},  langevin_history=False, thermal_time=10, auto_corr=5)
    
    sim_dse.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 1}, langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('dse_2_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 2}, langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 3}, langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('dse_4_moment', obs_kernel=dse_n_moment_kernel, const_param={'order': 4}, langevin_history=False, thermal_time=10, auto_corr=5)
    sim_dse.register_observable('abs_drift', obs_kernel=abs_drift, langevin_history=False, thermal_time=1, auto_corr=1)

    # run the sim
    for _ in tqdm(range(params["steps"])):
        sim_dse.step()
        for name, tr in sim_dse.trackers.items():
            tr.mark_equilibrated_trajs()
            tr.compute()

        # update histograms
        p_tracker = sim_dse.trackers["1_moment"]
        u_tracker = sim_dse.trackers["abs_drift"]
        my_act_parallel_loop(update_histogram_complex, p_tracker.equilibrated_trajs, params["trajs"], p_tracker.result, hist_p, params["bins_p"], params["p_min_real"], params["p_max_real"], params["p_min_imag"], params["p_max_imag"])
        my_act_parallel_loop(update_histogram_real, u_tracker.equilibrated_trajs, params["trajs"], u_tracker.result, hist_u, params["bins_u"], params["u_min"], params["u_max"])
        if use_cuda: cuda.synchronize()
    
    sim_dse.finish()
    if use_cuda:
        hist_u = hist_u.copy_to_host()
        hist_p = hist_p.copy_to_host()

    results = {'params': params}

    for name, tr in sim_dse.trackers.items():
        mean, sem_real, sem_imag = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean_real, tr.rolling_sqr_mean_imag, tr.counter)
        results[name] = {}
        results[name]["mean_real"] = mean.real
        results[name]["mean_imag"] = mean.imag
        results[name]["sem_real"] = sem_real
        results[name]["sem_imag"] = sem_imag


    return results, param_key, hist_p, hist_u

import json
if __name__ == "__main__":
    print("############################################################\n")
    sigma = -1+1j
    for slurm_index in range(1):
        print("############################################################")
        parameters = {
            "steps": [int(1e5)],
            "trajs": [int(1e3)],
            "dt": [1e-3],
            "sigma_abs": [np.abs(sigma)],
            "sigma_phase": [np.angle(sigma)],
            "lambda_abs": [2],
            "pullback": [100],
            "mass_modification": [0.5],
            "bins_u": [int(1e6)],
            "bins_p": [int(1e3)],
            "p_min_real": [-5],
            "p_max_real": [5],
            "p_min_imag": [-5],
            "p_max_imag": [5],
            "u_min": [1e-3],
            "u_max": [int(1e3)],
        }
        
        # Generate all parameter combinations
        import itertools
        param_combinations = [dict(zip(parameters.keys(), values)) for values in itertools.product(*parameters.values())]

        # Get index from Slurm
        if slurm_index >= len(param_combinations):
            print(f"Invalid SLURM_ARRAY_TASK_ID={slurm_index}, max is {len(param_combinations)-1}")
            sys.exit(1)

        # Select parameters for this job
        para = param_combinations[slurm_index]
        print(f"idx {slurm_index}")

        for key, value in para.items():
            print(key, "->", value)

        results, param_key, hist_p, hist_u = run_sim(para)
        # Save results
        import json
        file_path = os.path.join(os.getenv("HOME"), "gitrepos", "clonscal", "results", "gaussian_sim_data_local", "250225")
        with open(os.path.join(file_path, f"{param_key}.json"), "w") as f:
            json.dump(results, f, indent=4)
        with open(os.path.join(file_path, f"{param_key}_hist_p.npy"), "wb") as f:
            np.save(f, hist_p)
        with open(os.path.join(file_path, f"{param_key}_hist_u.npy"), "wb") as f:
            np.save(f, hist_u)
        print(f"saved data to {file_path}/{param_key}")
        print("############################################################\n")