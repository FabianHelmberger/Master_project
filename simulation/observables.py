# simulation/observables.py
import copy

from typing import Dict, Callable, Tuple
from .langevin_dynamics import *
from .config import *
from src.utils import KernelBridge, mark_equilibrated_trajs_kernel, fill_history_kernel, RollingStats, update_rolling_stats_scal_kernel, update_history
import src.scal as scal

from src.numba_target import use_cuda, my_act_parallel_loop


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
        self.meas_time = {} # np.full(shape=self.trajs, fill_value=-1, dtype=scal.SCAL_TYPE_REAL)


    def register_observable(self, obs_name: str, obs_kernel: Callable, shape = None, const_param={}, 
                            langevin_history=False, thermal_time=5, auto_corr=0.1, dtype=scal.SCAL_TYPE):
        """
        Register a new observable with the option to track Langevin time.
        """

        if obs_name in self.trackers.keys():
            raise ValueError(f"Observable '{obs_name}' is already registered.")
        if shape is None: shape = (self.n_cells,)

        self.trackers[obs_name] = ObservableTracker(sim_instance=self, obs_name=obs_name, shape=shape, obs_kernel=obs_kernel,
                                                    const_param=const_param, langevin_history=langevin_history,
                                                    thermal_time=thermal_time, auto_corr=auto_corr, dtype=dtype)
        
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
    #     if use_cuda: cuda.synchronize()

    #     # tracker.update(self.equilibrated_traj, self.adims)
    #     tracker.update()

    def finish(self):
        for tr in self.trackers.values():
            # merge trajectory data 
            tr.rolling_mean = np.sum(tr.rolling_mean, axis = 0)
            tr.rolling_sqr_mean_real= np.sum(tr.rolling_sqr_mean_real, axis = 0)
            tr.rolling_sqr_mean_imag = np.sum(tr.rolling_sqr_mean_imag, axis = 0)
            tr.counter = np.sum(tr.counter, axis = 0)

            # if tr.langevin_history:
            #     mask = tr.history_result!=0
            #     tr.history_result = tr.history_result[mask]
            #     tr.history_meas_times = tr.history_meas_times[mask]


class ObservableTracker:
    def __init__(self, sim_instance: LangevinDynamics, obs_name, shape: tuple, 
                 obs_kernel: Callable, langevin_history=False, const_param={}
                 , thermal_time=5, auto_corr=0.1,dtype=scal.SCAL_TYPE):
        
        self.obs_name = obs_name
        self.shape = shape
        self.obs_kernel = obs_kernel
        self.langevin_history = langevin_history
        self.const_param = const_param
        self.init_history_size = int(sim_instance.steps+1)
        self.thermal_time = thermal_time
        self.auto_corr = auto_corr
        for key, val in const_param.items(): self.__setattr__(key, val)
        
        self.__dict__['_sim_instance'] = sim_instance

        self.equilibrated_trajs = np.zeros(sim_instance.trajs, dtype=bool)
        self.meas_time = np.full(shape=sim_instance.trajs, fill_value=-1, dtype=scal.SCAL_TYPE_REAL)
        self.result = np.zeros(shape=shape, dtype=dtype)

        if langevin_history: 
            # self.history = np.zeros(shape=(sim_instance.trajs, init_history_size, *self.shape), dtype=scal.SCAL_TYPE)
            # self.history_result = np.zeros(shape=(sim_instance.trajs, init_history_size, 1), dtype=dtype) # for every traj and langevin steps store value and meas time
            # self.history_meas_times = np.zeros(shape=(sim_instance.trajs, init_history_size, 1), dtype=scal.SCAL_TYPE_REAL) # for every traj and langevin steps store value and meas time
            
            self.history_result = np.zeros(shape=self.init_history_size, dtype=dtype) # for every traj and langevin steps store value and meas time
            self.history_meas_times = np.zeros(shape=self.init_history_size, dtype=scal.SCAL_TYPE_REAL) # for every traj and langevin steps store value and meas time

        self.kernel_bridge = KernelBridge(self, kernel_funcs=[obs_kernel], const_param=const_param, result=self.result)
        
        self.stats = RollingStats()
        # self.rolling_mean = np.array([0], dtype=scal.SCAL_TYPE)
        # self.rolling_sqr_mean = np.array([0], dtype=scal.SCAL_TYPE_REAL)
        # self.counter = np.array([0], dtype=scal.LATT_TYPE)

        self.rolling_mean = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE)
        self.rolling_sqr_mean_real = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.rolling_sqr_mean_imag = np.zeros(sim_instance.trajs, dtype=scal.SCAL_TYPE_REAL)
        self.counter = np.zeros(sim_instance.trajs, dtype=scal.LATT_TYPE)

        if use_cuda: 
            gpu_hanlder = GPU_handler(self)
            gpu_hanlder.to_device()

    def __getattr__(self, name):
        return getattr(self._sim_instance, name)
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value

#         # TODO: make history size dynamic
#         # if langevin_history: self.history = np.full((init_history_size, *self.shape), np.nan, dtype=scal.SCAL_TYPE)
#         # self.result = np.full(shape=self.shape, fill_value=np.nan, dtype=scal.SCAL_TYPE)


    def update(self):
        """
        Update the tracker with a new observable value at the current Langevin step.
        """
        # if use_cuda: result = self.result.copy_to_host().copy()
        # else: result = self.result.copy()


        my_act_parallel_loop(update_rolling_stats_scal_kernel, self.equilibrated_trajs, self.trajs, self.result, self.rolling_mean, self.rolling_sqr_mean_real, self.rolling_sqr_mean_imag, self.counter)
        if use_cuda: cuda.synchronize()

        if self.langevin_history:

            self.history_result[self.langevin_steps] = np.mean(self.result)
            self.history_meas_times[self.langevin_steps] = np.mean(self.meas_time)

            # my_act_parallel_loop(update_history, self.equilibrated_trajs,
            #                     self.trajs, self.langevin_steps, self.meas_time, self.history_result,
            #                     self.history_meas_times, self.result)

            # self.history[self.langevin_steps] = self.result
            # self.history_meas_times[self.langevin_steps] = self.meas_time
	
            if use_cuda: cuda.synchronize()
            
        # self.stats.update(result)
        # print(result)
        # self.result[:] = np.NaN
        # self.result[:] = 0.0

    def get_full_history(self):
        """Return the full history of observables (if enabled)."""

        if not self.langevin_history:
            raise ValueError("Langevin tracking is disabled for this observable.")
        return self.history_result
    
    def mark_equilibrated_trajs(self):

        my_parallel_loop(mark_equilibrated_trajs_kernel, self.trajs, self.meas_time, self.langevin_time, 
                         self.equilibrated_trajs, self.thermal_time, self.auto_corr)
    def compute(self):
        args = self.kernel_bridge.get_current_params()[self.obs_kernel]
        my_act_parallel_loop(self.obs_kernel, self.equilibrated_trajs, *args.values())
        if use_cuda: cuda.synchronize()
        self.update()
