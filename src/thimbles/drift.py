# src/thimbles/drift.py
from typing import Callable, Optional
import numpy as np

class DriftModel:
    def __init__(self, 
                 compiled_drift: Callable[[complex], complex], 
                 critical_points_func: Optional[Callable[[], np.ndarray]] = None):
        """
        Expects a Numba-compiled drift function of signature (z: complex) -> complex.
        """
        self._compiled_drift = compiled_drift
        self.critical_points_func = critical_points_func

    def evaluate(self, z: complex) -> complex:
        return self._compiled_drift(z)

    def critical_points(self) -> Optional[np.ndarray]:
        if self.critical_points_func:
            return self.critical_points_func()
        return None