# src/thimbles/drift.py
from typing import Callable, Optional
import numpy as np
import numba

class DriftFunction:
    def __init__(self, 
                 drift_func: Callable[[complex, dict], complex], 
                 params: dict, 
                 action_func: Optional[Callable[[complex, dict], complex]] = None,
                 critical_points_func: Optional[Callable[[], np.ndarray]] = None):
        self.drift_func = drift_func
        self.params = params
        self.action_func = action_func
        self.critical_points_func = critical_points_func
        self._compiled_drift = numba.njit(lambda z: drift_func(z, params))

    def evaluate(self, z: complex) -> complex:
        return self._compiled_drift(z)

    def action(self, z: complex) -> Optional[complex]:
        if self.action_func:
            return self.action_func(z, self.params)
        return None

    def critical_points(self) -> Optional[np.ndarray]:
        if self.critical_points_func:
            return self.critical_points_func()
        return None