
# src/thimbles/sdflow.py
import numpy as np
from .drift import DriftFunction

class SDFlow:
    def __init__(self, drift_fn: DriftFunction, steps=100, eps=1e-3, max_abs_r=10., bound=1e-3, dual=False):
        self.drift_fn = drift_fn
        self.steps = steps
        self.eps = eps
        self.max_abs_r = max_abs_r
        self.bound = bound
        self.dual = dual

    def run(self, r0: complex) -> np.ndarray:
        eps = -self.eps if self.dual else self.eps
        rlist, r = [], r0

        for _ in range(self.steps):
            dS = self.drift_fn.evaluate(r)
            dS_abs = np.abs(dS)

            if eps * dS_abs > self.bound:
                r += self.bound / dS_abs * dS.conjugate()
            else:
                r += eps * dS.conjugate()

            if np.abs(r) >= self.max_abs_r:
                break

            rlist.append(r)

        return np.array(rlist)
