# imports
# Add cle_fun to PYTHON_PATH
import sys, os, json
sys.path.append("../../../../../clonscal_bak")


from analytic.formulas import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "double"
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
    gaussian_modified_density_drift_kernel
)

if use_cuda: from numba import cuda

parameters = {
        "trajs": int(5e2),
        "mean_dS_max": 50,
        "max_langevin_time": 10,
        "lambda_abs": 1,
    }

# sigma = -1+1j
# parameters["sigma_abs"] = np.abs(sigma)
# parameters["sigma_phase"] = np.angle(sigma)

from scipy.optimize import curve_fit

def fit_exp(x, tau):
    return np.exp(-x / tau)

import numba
@numba.njit(parallel=True)
def autocorrelate(dat, tmax=10000):
    # correlate
    if tmax is None or tmax > len(dat):
        tmax = len(dat)

    # subtract mean
    dat_norm = dat - np.mean(dat)
    
    acf = np.zeros(tmax, dtype=dat.dtype)
    for dt_ in numba.prange(tmax):
        corr = 0.0
        for t in range(len(dat)-dt_):
            update = dat_norm[t]*dat_norm[t+dt_]
            corr += update

        corr /= len(dat)-dt_
        acf[dt_] = corr

    # normalize
    acf /= acf[0]
    return acf



import tqdm
def run_sim(dt, maximal_lt, sigma, interaction, auto_corr=None, thermal_time=None):
    config = Config(trajs=int(1e4), dt = dt, ada_step = True, sigma = sigma, interaction = interaction)
    sim = ComplexLangevinSimulation(config)
    if auto_corr == None: auto_corr = dt
    if thermal_time == None: thermal_time = 0


    sim.register_observable('1_moment', obs_kernel=n_moment_kernel, const_param={"order" : 1},  
                            langevin_history_full=True, langevin_history=True, thermal_time=thermal_time, 
                            auto_corr=auto_corr, maximal_lt=maximal_lt, history_grid_size=auto_corr*2)
    
    sim.register_observable('2_moment', obs_kernel=n_moment_kernel, const_param={"order" : 2},  
                        langevin_history_full=True, langevin_history=True, thermal_time=thermal_time, 
                        auto_corr=auto_corr, maximal_lt=maximal_lt, history_grid_size=auto_corr*2)

    if use_cuda:
        gpu_handler = GPU_handler(sim)
        gpu_handler.to_device()

    max_lt_global = np.max([obj.maximal_lt for obj in sim.trackers.values()])
    with tqdm.tqdm(total=max_lt_global) as pbar:
        while np.min(sim.langevin_time[sim.alive]) < max_lt_global:
            lt = np.min(sim.langevin_time[sim.alive])
            pbar.update(lt- pbar.n)
            sim.step()
            sim.kill_trajs()

            for _, tr in sim.trackers.items():
                tr.mark_equilibrated_trajs()
                tr.compute()
        sim.finish()
    return sim  

# Define parameter values
SIGMAS = [-1+1j, 1+1j, -1+2j, 1+2j, -1+3j, 1+3j, -1+4j, 1+4j]
MAX_LTS = [50, 60, 70]
LAMBDAS = [1,2]

import argparse
import itertools
# Generate all possible parameter combinations
param_permutations = list(itertools.product(SIGMAS, LAMBDAS, MAX_LTS))

import copy
def main(slurm_idx):
    dt = 5e-4
    sigma, interaction, max_lt = param_permutations[slurm_idx]
    # run simulation

    sim = run_sim(dt, max_lt, sigma = sigma, interaction=interaction, auto_corr=dt, thermal_time = 0)

    tr = sim.trackers["2_moment"]
    for name, tr in sim.trackers.items():
        for part in [np.real, np.imag]:
            fig, ax = plt.subplots(1,figsize=(4.5, 3.2))

            res = copy.deepcopy(tr.history_result_full)
            lt = copy.deepcopy(tr.history_meas_times_full)
            # calculate acf for every traj individually and store it
            acfs = []
            lts = []
            dats = []
            for traj_idx in tqdm.tqdm(range(sim.trajs)):
                # fetch traj data and langevin measure times
                this_res = part(res)[:, traj_idx]
                this_lt = lt[:, traj_idx]

                # remove nans
                this_res = this_res[~np.isnan(this_res)]
                this_lt = this_lt[~np.isnan(this_lt)]

                # calculate acf
                this_acf = autocorrelate(this_res, tmax=len(this_res))
                acfs.append(this_acf)
                lts.append(this_lt)
                dats.append(this_res)

            # draw acfs individually
            for acf, lt in zip(acfs, lts):
                ax.plot(lt, acf, color = "black", alpha = 0.01)

            # fill removed datapoints with nans
            max_len = max(len(arr) for arr in acfs)
            padded_acfs = np.full((len(acfs), max_len), np.nan)
            for i, arr in enumerate(acfs):
                padded_acfs[i, :len(arr)] = arr

            max_len = max(len(arr) for arr in lts)
            padded_lts = np.full((len(lts), max_len), np.nan)
            for i, arr in enumerate(lts):
                padded_lts[i, :len(arr)] = arr


            # get mean result over trajs
            res = part(tr.history_result_full)
            lt = tr.history_meas_times_full
            res_mean = np.nanmean(res, axis = 1)
            lt_mean = np.nanmean(lt, axis = 1)

            res_mean = res_mean[~np.isnan(res_mean)]
            lt_mean = lt_mean[~np.isnan(lt_mean)]

            # plot acf of the mean
            ax.plot(lt_mean, autocorrelate(res_mean, tmax = len(lt_mean)), color = "red", label = "acf of mean")

            # plot mean acf and its fit
            ax.plot(np.mean(padded_lts, axis = 0), np.mean(padded_acfs, axis = 0), color = "blue", label = "mean of acf")
            p, _ = curve_fit(fit_exp, np.nanmean(padded_lts, axis = 0), np.nanmean(padded_acfs, axis = 0))
            ax.plot(np.mean(padded_lts, axis = 0), fit_exp(np.mean(padded_lts, axis = 0), p), color = "magenta", label = "fit")

            ax.set_xlim(0, int(10*p))
            ax.set_ylim(-1, 1)
            ax.set_title(f"dt = {dt}, tmax = {max_lt}, p ={p}")
            # plt.show()
            ax.legend()
            fig.savefig(f"../../autocorr/unmod_{name}_{part.__name__}_s{np.round(sim.sigma)}_l{np.round(sim.interaction)}_dt{dt}_ltmax{max_lt}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm_idx", type=int, required=True)
    args = parser.parse_args()

    main(args.slurm_idx)