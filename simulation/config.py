# config.py
import math

class Config:
    def __init__(self, **kwargs):
        self.steps = kwargs.get('steps', 1e3)
        self.dims = kwargs.get('dims', [10, 10])
        self.noise_factor = kwargs.get('noise_factor', 1.0)
        self.dt = kwargs.get('dt', 1e-5)
        self.sqrt2 = kwargs.get('noise_factor', math.sqrt(2.0))
        self.mass_real = kwargs.get('mass_real', 1.0)
        self.mass_imag = kwargs.get('mass_real', 0.0)
        self.interaction = kwargs.get('interaction', 0.4)