# config.py
import math
import src.scal as scal

class Constants:
    def __init__(self):
        self.sqrt2 = scal.SCAL_TYPE_REAL(math.sqrt(2.0))
        self.noise_factor = scal.SCAL_TYPE_REAL(1.0)
        super().__init__()