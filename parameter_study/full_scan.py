def calculate_stats_complex(rolling_mean, rolling_sqr_mean, counter):
    """
    Calculate mean and SEM for complex numbers.
    
    Parameters:
    rolling_mean (np.array): Single-element array containing the rolling sum of complex values.
    rolling_sqr_mean (np.array): Single-element array containing the rolling sum of squared magnitudes of complex values.
    counter (np.array): Single-element array containing the number of values.
    
    Returns:
    tuple: (mean, sem_real, sem_imag), where:
        - mean is the complex mean,
        - sem_real is the SEM for the real part,
        - sem_imag is the SEM for the imaginary part.
    """
    if counter[0] == 0:
        raise ValueError("Counter cannot be zero to avoid division by zero.")
    
    # Extract real and imaginary parts
    rolling_mean_real = rolling_mean[0].real
    rolling_mean_imag = rolling_mean[0].imag
    
    # Mean calculation
    mean_real = rolling_mean_real / counter[0]
    mean_imag = rolling_mean_imag / counter[0]
    
    # Variance for real part
    rolling_sqr_mean_real = rolling_sqr_mean[0]
    variance = (rolling_sqr_mean_real / counter[0]) - mean_real**2 - mean_imag**2
    variance = max(variance, 0)
        
    # Standard error of the mean (SEM)
    sem = np.sqrt(variance / counter[0])
    
    # Return complex mean and SEM for real and imaginary parts
    mean = mean_real + 1j * mean_imag
    return mean, sem



import numpy as np

# Set environment variables
import os
os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "single"
os.environ["MY_NUMBA_TARGET"] = "cuda"
 
# Add cle_fun to PYTHON_PATH
import sys
sys.path.append("../../clonscal")

# defining the simulation
from simulation.config import Config
from simulation.cl_simulation import ComplexLangevinSimulation
from simulation.gpu_handler import GPU_handler
from src.obs_kernels import n_moment_kernel, dse_n_moment_kernel
from src.utils import gaussian_modified_density_drift_kernel
from src.numba_target import use_cuda
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

    sim_dse = ComplexLangevinSimulation(config)
    
    if use_cuda: 
        gpu_handler = GPU_handler(sim_dse)
        gpu_handler.to_device()

    sim_dse.register_observable('dse_1_moment', obs_kernel=dse_n_moment_kernel, const_param={"order" : 1},  langevin_history=False, thermal_time=1, auto_corr=1)
    sim_dse.register_observable('dse_3_moment', obs_kernel=dse_n_moment_kernel, const_param={"order" : 3},  langevin_history=False, thermal_time=1, auto_corr=1)
    sim_dse.register_observable('dse_5_moment', obs_kernel=dse_n_moment_kernel, const_param={"order" : 5},  langevin_history=False, thermal_time=1, auto_corr=1)

    for _ in tqdm(range(params["steps"])):
        sim_dse.step()
        for name, tr in sim_dse.trackers.items():
            tr.mark_equilibrated_trajs()
            tr.compute()
    sim_dse.finish()

    results = {'params': params}
    for name, tr in sim_dse.trackers.items():
        mean, sem = calculate_stats_complex(tr.rolling_mean, tr.rolling_sqr_mean, tr.counter)
        results[name] = {}
        results[name]["mean_real"] = mean.real
        results[name]["mean_imag"] = mean.imag
        results[name]["sem"] = sem
        
    return results, param_key


# create parameters
start_angle = 0
stop_angle = 2 * np.pi
angles_cl = np.linspace(start_angle, stop_angle, 16)
pullback_phase = np.linspace(start_angle, stop_angle)
parameters = {
    "steps": [int(5e4)],
    "trajs": [int(1e4)],
    "dt": [5e-4],
    "sigma_abs": [1],
    "sigma_phase": angles_cl,
    "lambda_abs": [1],
    "pullback_phase": [0],
    "pullback_abs": [5],
    "mass_modification": [2],
}

# Generate all parameter combinations
import itertools
param_combinations = [dict(zip(parameters.keys(), values)) for values in itertools.product(*parameters.values())]

# Get index from Slurm
slurm_index = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))  # Default to 0 if not running in Slurm

if slurm_index >= len(param_combinations):
    print(f"Invalid SLURM_ARRAY_TASK_ID={slurm_index}, max is {len(param_combinations)-1}")
    sys.exit(1)

# Select parameters for this job
param = param_combinations[slurm_index]


# Run the simulation
results, param_key = run_sim(param)

# Save results
import json
file_path = os.path.join(os.getenv("HOME"), "gitrepos", "clonscal", "parameter_study", "sim_data", f"{param_key}.json")
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"{slurm_index}: Saved results to {file_path}")