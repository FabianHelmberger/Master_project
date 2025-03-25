# simulation/observables.py
import copy

from typing import Dict, Callable, Tuple
from .langevin_dynamics import *
from .config import *
from src.utils import (
                    KernelBridge, 
                    update_histogram_real,
                    mark_equilibrated_trajs_kernel,
                    fill_history_kernel,
                    RollingStats,
                    update_rolling_stats_scal_kernel,
                    update_history_kernel,
                    update_history_full_kernel
                    )

import src.scal as scal
from src.numba_target import use_cuda, my_act_parallel_loop, my_act_loop


if use_cuda:
    from numba import cuda # type: ignore
    from simulation.gpu_handler import GPU_handler


class Observables(LangevinDynamics):
    """
    Extends LangevinDynamics to calculate and track observables over Langevin time.
    """
    def __init__(self, config):
        super().__init__(config)
        self.trackers: Dict[str, ObservableTracker] = {}  # Stores trackers for different observables
        # self.result: Dict = {}
        self.kernel_bridges: Dict[str, KernelBridge] = {}  # Stores trackers for different observables
        # self.meas_time = {} # np.full(shape=self.trajs, fill_value=-1, dtype=scal.SCAL_TYPE_REAL)


    def register_observable(self, obs_name: str, obs_kernel: Callable, shape = None, const_param={}, 
                            langevin_history=False, langevin_history_full=False, thermal_time=5, auto_corr=0.1,maximal_lt=100, 
                            history_grid_size=None, dtype=scal.SCAL_TYPE):
        """
        Register a new observable with the option to track Langevin time.
        """

        if obs_name in self.trackers.keys():
            raise ValueError(f"Observable '{obs_name}' is already registered.")
        if shape is None: shape = (self.n_cells,)

        self.trackers[obs_name] = ObservableTracker(sim_instance=self, obs_name=obs_name, shape=shape, obs_kernel=obs_kernel,
                                                    const_param=const_param, langevin_history=langevin_history, langevin_history_full = langevin_history_full,
                                                    thermal_time=thermal_time, auto_corr=auto_corr, maximal_lt=maximal_lt,history_grid_size=history_grid_size,
                                                      dtype=dtype)
        # assert not use_cuda or not langevin_history, print("Langevin history currently only in Python/Numba Mode!")

        # self.result[obs_name] = self.trackers[obs_name].result

    # def compute_all(self):
    #     """
    #     Compute an observable using the provided kernel and track its results.
    #     """
    #     for name in obs_name in self.trackers.keys():
    #         try:
    #             tracker = self.trackers[obs_name]
    #             bridge = self.kernel_bridges[obs_name]
    #         except:
    #             raise ValueError(f"'{obs_name}' is not registered.")

    #     tracker.mark_equilibrated_trajs()

    #     obs_kernel = tracker.obs_kernel
    #     print("\n")
    #     print(obs_kernel, tracker.trajs, tracker.equilibrated_trajs, tracker.phi0,
    #                          tracker.result, 2, tracker.langevin_time, tracker.adims, tracker.meas_time)
        
    #     # my_act_parallel_loop(obs_kernel, self.trajs, self.trackers[obs_name].equilibrated_trajs, self.phi0,
    #     #                      self.result[obs_name], 2, self.langevin_time, self.adims, self.meas_time[obs_name])
        
    #     # kernel_args = bridge.get_current_params()[obs_kernel].values()
    #     # my_act_parallel_loop(obs_kernel, *kernel_args)

    #     # tracker.update(self.equilibrated_traj, self.adims)
    #     tracker.update()

    def finish(self):
        for tr in self.trackers.values():
            # merge trajectory data 
            tr.rolling_mean = np.sum(tr.rolling_mean, axis = 0)
            tr.rolling_sqr_mean_real= np.sum(tr.rolling_sqr_mean_real, axis = 0)
            tr.rolling_sqr_mean_imag = np.sum(tr.rolling_sqr_mean_imag, axis = 0)
            tr.rolling_sqr_mean_cross = np.sum(tr.rolling_sqr_mean_cross, axis = 0)
            
            tr.counter = np.sum(tr.counter, axis = 0)

            if tr.langevin_history:
                # bin and average over trajs
                mask = tr.history_counter>0
                tr.history_result  = np.where(mask, tr.history_result / tr.history_counter, np.nan)
                tr.history_meas_times = np.where(mask, tr.history_meas_times / tr.history_counter, np.nan)

            if tr.langevin_history_full:
                # bin the data 
                
                mask = tr.history_counter_full>0
                tr.history_result_full  = np.where(mask, tr.history_result_full / tr.history_counter_full, np.nan)
                tr.history_meas_times_full = np.where(mask, tr.history_meas_times_full / tr.history_counter_full, np.nan)

                # mask = sim.trackers["2_moment"].history_counter_full>0
                # dat = np.where(mask, sim.trackers["2_moment"].history_result_full / sim.trackers["2_moment"].history_counter_full, np.nan )
                # lt = np.where(mask, sim.trackers["2_moment"].history_meas_times_full / sim.trackers["2_moment"].history_counter_full, np.nan )

                # mask = tr.history_counter_ > 0
                # tr.history_result_ = tr.history_result_[mask]/tr.history_counter_[mask]
                # tr.history_meas_times_ = tr.history_meas_times_[mask]/tr.history_counter_[mask]

                # tr.history_result[~mask] = np.nan
                # tr.history_meas_times[~mask] = np.nan
                # tr.history_counter[~mask] = np.nan
                # tr.history_result[~mask] = np.nan  # Mark invalid bins
                # tr.history_meas_times[~mask] = np.nan

class ObservableTracker:
    def __init__(self, sim_instance: LangevinDynamics, obs_name, shape: tuple, 
                 obs_kernel: Callable, langevin_history=False, langevin_history_full=False,
                 const_param={}, thermal_time=5, auto_corr=0.1,maximal_lt=None, history_grid_size=None,
                 dtype=scal.SCAL_TYPE):
        
        self.obs_name = obs_name
        self.shape = shape
        self.obs_kernel = obs_kernel
        self.langevin_history = langevin_history
        self.langevin_history_full = langevin_history_full
        self.const_param = const_param
        self.thermal_time = thermal_time
        self.auto_corr = auto_corr
        self.maximal_lt = maximal_lt
        if history_grid_size==None: self.history_grid_size = 2*self.auto_corr 
        else history_grid_size
        self.dtype = dtype
        self.init_history_size = 2*int(maximal_lt / self.history_grid_size)
        for key, val in const_param.items(): self.__setattr__(key, val)
        self.__dict__['_sim_instance'] = sim_instance

        self.equilibrated_trajs = np.zeros(sim_instance.trajs, dtype=bool)
        self.meas_time = np.zeros(shape=sim_instance.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.result = np.zeros(shape=shape, dtype=dtype)

        if langevin_history: 

            self.history_result = np.zeros(shape=self.init_history_size, dtype=dtype) # for every traj and langevin steps store value and meas time
            self.history_counter = np.zeros(shape=self.init_history_size, dtype=np.int32) # count the number of trajectories that participated to a 
            self.history_meas_times = np.zeros(shape=self.init_history_size, dtype=scal.SCAL_TYPE_REAL) # for every traj and langevin steps store value and meas time
        
        if langevin_history_full:
            self.history_result_full = np.zeros(shape=(self.init_history_size, self.trajs), dtype=dtype) # for every traj and langevin steps store value and meas time
            self.history_counter_full = np.zeros(shape=(self.init_history_size, self.trajs), dtype=np.int32) # count the number of trajectories that participated to a 
            self.history_meas_times_full = np.zeros(shape=(self.init_history_size, self.trajs), dtype=scal.SCAL_TYPE_REAL) # for every traj and langevin steps store value and meas time

        self.kernel_bridge = KernelBridge(self, kernel_funcs=[obs_kernel], const_param=const_param, result=self.result)
        
        self.stats = RollingStats()

        self.rolling_mean = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE)
        self.rolling_sqr_mean_abs = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE)
        self.rolling_sqr_mean_real = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.rolling_sqr_mean_cross = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.rolling_sqr_mean_imag = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.counter = np.zeros(sim_instance.trajs, dtype=scal.IDX_TYPE)
       
        if use_cuda:
            gpu_hanlder = GPU_handler(self)
            gpu_hanlder.to_device()

    def __getattr__(self, name):
        return getattr(self._sim_instance, name)
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value


    def update(self):
        """
        Update the tracker with a new observable value at the current Langevin step.
        """
        # else: result = self.result.copy()


        my_act_parallel_loop(update_rolling_stats_scal_kernel, self.equilibrated_trajs, self.trajs, 
                             self.result, self.rolling_mean, self.rolling_sqr_mean_real, self.rolling_sqr_mean_imag, self.rolling_sqr_mean_cross, self.counter)

        if self.langevin_history:
            my_act_loop(update_history_kernel, self.equilibrated_trajs, self.trajs, self.history_counter, 
                                 self.history_result, self.history_meas_times, self.meas_time, self.result, self.history_grid_size)
            
        if self.langevin_history_full:
            my_act_loop(update_history_full_kernel, self.equilibrated_trajs, self.trajs, self.history_counter_full, 
                                 self.history_result_full, self.history_meas_times_full, self.meas_time, self.result, self.history_grid_size)
            
            # EXPERIMENTAL: Manually append all traj data 
            # for traj_idx in range(self.trajs):
            #     if self.equilibrated_trajs[traj_idx]:
            #         bin_mt = int(self.meas_time[traj_idx] / self.history_grid_size)
            #         self.history_counter_[bin_mt, traj_idx] += 1
            #         self.history_result_[bin_mt, traj_idx] += self.result[traj_idx]
            #         self.history_meas_times_[bin_mt, traj_idx] += self.meas_time[traj_idx]
            
	
            

    def get_full_history(self):
        """Return the full history of observables (if enabled)."""

        if not self.langevin_history:
            raise ValueError("Langevin tracking is disabled for this observable.")
        return self.history_result
    
    def mark_equilibrated_trajs(self):

        my_parallel_loop(mark_equilibrated_trajs_kernel, self.trajs, self.meas_time, self.langevin_time, 
                         self.equilibrated_trajs, self.thermal_time, self.auto_corr, self.maximal_lt, self.alive)
        # self.equilibrated_trajs &= self.alive

    def compute(self):
        args = self.kernel_bridge.get_current_params()[self.obs_kernel]
        my_act_parallel_loop(self.obs_kernel, self.equilibrated_trajs, *args.values())

        self.update()