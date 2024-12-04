# config.py
import math
import src.scal as scal
from src.utils import noise_kernel, mexican_hat_kernel_real, evolve_kernel
from src.numba_target import use_cuda

if use_cuda:
    from src.utils import cuda_noise_kernel as noise_kernel

class Config:
    def __init__(self, **kwargs):
        self.steps: scal.LATT_TYPE = kwargs.get('steps', 1e3)
        self.dims = kwargs.get('dims', [10, 10])
        self.trajs = kwargs.get('trajs', 1)
        self.dt: scal.SCAL_TYPE_REAL = kwargs.get('dt', 1e-5)
        self.mass_real: scal.SCAL_TYPE_REAL = kwargs.get('mass_real', 1.0)
        self.mass_imag: scal.SCAL_TYPE_REAL = kwargs.get('mass_real', 0.0)
        self.interaction: scal.SCAL_TYPE_REAL = kwargs.get('interaction', 0.4)
        self.noise_seed = 0
        self.ada_step: bool =  kwargs.get('ada_step', True)
        self.noise_kernel: callable = kwargs.get('noise_kernel', noise_kernel)
        self.evolve_kernel: callable = kwargs.get('evolve_kernel', evolve_kernel)
        self.drift_kernel: callable = kwargs.get('drift_kernel', mexican_hat_kernel_real)
        self.max_langevin_time: scal.SCAL_TYPE_REAL = kwargs.get('max_langevin_time', 10)
        self.show_bars: bool =  kwargs.get('show_bars', True)
        
        super().__init__()  # Ensure compatibility with multiple inheritance
