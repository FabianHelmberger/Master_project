# config.py
import math
import src.scal as scal

class Config:
    def __init__(self, **kwargs):
        self.steps: scal.LATT_TYPE = kwargs.get('steps', 1e3)
        self.dims = kwargs.get('dims', [10, 10])
        self.dt: scal.SCAL_TYPE_REAL = kwargs.get('dt', 1e-5)
        self.mass_real: scal.SCAL_TYPE_REAL = kwargs.get('mass_real', 1.0)
        self.mass_imag: scal.SCAL_TYPE_REAL = kwargs.get('mass_real', 0.0)
        self.interaction: scal.SCAL_TYPE_REAL = kwargs.get('interaction', 0.4)
        self.noise_seed = 0
        super().__init__()  # Ensure compatibility with multiple inheritance
