# config.py
import math
import src.scal as scal
from src.utils import noise_kernel, mexican_hat_kernel_real, evolve_kernel
from src.numba_target import use_cuda

if use_cuda:
    from src.utils import cuda_noise_kernel as noise_kernel

class Config:
    def __init__(self, **kwargs):
        self.dims = kwargs.get('dims', [1])
        self.trajs = kwargs.get('trajs', 1)
        self.dt: scal.SCAL_TYPE_REAL = kwargs.get('dt', 1e-5)
        self.sigma: scal.SCAL_TYPE_REAL = kwargs.get('sigma', 1.0)
        self.interaction: scal.SCAL_TYPE_REAL = kwargs.get('interaction', 0.4)
        self.noise_seed = 0
        self.ada_step: bool =  kwargs.get('ada_step', True)
        self.noise_kernel: callable = kwargs.get('noise_kernel', noise_kernel)
        self.evolve_kernel: callable = kwargs.get('evolve_kernel', evolve_kernel)
        self.drift_kernel: callable = kwargs.get('drift_kernel', mexican_hat_kernel_real)
        self.max_langevin_time: scal.SCAL_TYPE_REAL = kwargs.get('max_langevin_time', 10)
        self.show_bars: bool =  kwargs.get('show_bars', True)
        self.phi_singular=kwargs.get('phi_singular', 10)
        self.pullback=kwargs.get('pullback', 1.5)
        self.mass_modification=kwargs.get('mass_modification', 1)
        self.mean_dS_max=kwargs.get("mean_dS_max", 50)
        self.ada_min = kwargs.get("ada_min", 1e-3)
        self.history_grid_size = kwargs.get("history_grid_size", self.dt*10)
        self.steps: scal.IDX_TYPE = kwargs.get('steps', int(self.max_langevin_time/(self.dt*self.ada_min)))

        
        super().__init__()  # Ensure compatibility with multiple inheritance