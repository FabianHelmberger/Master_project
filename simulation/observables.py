# simulation/observables.py
import copy

from typing import Dict, Callable, Tuple
from .langevin_dynamics import *
from .config import *
from src.utils import KernelBridge
import src.scal as scal

from src.numba_target import use_cuda

if use_cuda:
    from numba import cuda

class ObservableTracker:
    """
    Tracks an observable's Langevin time trajectory and/or rolling statistics.
    """
    def __init__(self, shape: tuple, obs_kernel: Callable, langevin_history=False, init_history_size = int(1e5)):
        self.obs_kernel = obs_kernel
        self.shape = shape
        self.langevin_history = langevin_history
        self.history = np.empty(shape=(init_history_size, *self.shape), dtype=scal.SCAL_TYPE)
        self.langevin_steps = 0
        self.result = np.empty(shape=self.shape, dtype=scal.SCAL_TYPE)

        if use_cuda: self.result  = cuda.to_device(self.result)

    def update(self):
        """
        Update the tracker with a new observable value at the current Langevin step.
        """
        if use_cuda: result = self.result.copy_to_host().copy()
        else: result = self.result.copy()
        if self.langevin_history:
            self.history[self.langevin_steps] = result
        self.langevin_steps += 1

    def get_full_history(self):
        """Return the full history of observables (if enabled)."""

        if not self.langevin_history:
            raise ValueError("Langevin tracking is disabled for this observable.")
        return self.history
    


class Observables(LangevinDynamics):
    """
    Extends LangevinDynamics to calculate and track observables over Langevin time.
    """
    def __init__(self, config):
        super().__init__(config)
        self.trackers: Dict[str, ObservableTracker] = {}  # Stores trackers for different observables
        self.result: Dict = {}
        self.const_param: Dict = {}
        self.kernel_bridges: Dict[str, KernelBridge] = {}  # Stores trackers for different observables


    def register_observable(self, name: str, obs_kernel: Callable, shape = None, const_params={}, 
                            langevin_history=False):
        """
        Register a new observable with the option to track Langevin time.
        """

        if name in self.trackers.keys():
            raise ValueError(f"Observable '{name}' is already registered.")
        if shape is None: shape = (self.n_cells,)

        self.trackers[name] = ObservableTracker(shape=shape, obs_kernel=obs_kernel, 
                                                langevin_history=langevin_history)
        self.result[name] = self.trackers[name].result

        self.kernel_bridges[name] = KernelBridge(self, kernel_funcs=[obs_kernel], 
                                                 const_param=const_params, result=self.result[name])


    def compute(self, obs_name: str):
        """
        Compute an observable using the provided kernel and track its results.
        """
        try:
            tracker = self.trackers[obs_name]
            bridge = self.kernel_bridges[obs_name]
        except:
            raise ValueError(f"'{obs_name}' is not registered.")

        obs_kernel = tracker.obs_kernel
        kernel_args = bridge.get_current_params()[obs_kernel].values()
        my_parallel_loop(obs_kernel, *kernel_args)
        if use_cuda: cuda.synchronize()
        tracker.update()

    def finish(self):
        for tr in self.trackers.values():
            if tr.langevin_history:
                tr.history = tr.history[:tr.langevin_steps]