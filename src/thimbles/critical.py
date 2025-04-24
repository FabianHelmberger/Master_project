

# src/thimbles/critical.py
import numpy as np
from scipy.optimize import newton
from .drift import DriftFunction

class CriticalPointFinder:
    def __init__(self, drift_fn: DriftFunction, x_range=[-5, 5], y_range=[-5, 5], x_n=50, y_n=50):
        self.drift_fn = drift_fn
        self.x_range = x_range
        self.y_range = y_range
        self.x_n = x_n
        self.y_n = y_n

    def find(self) -> np.ndarray:
        analytic = self.drift_fn.critical_points()
        if analytic is not None:
            return analytic

        if not self.drift_fn.action_func:
            raise ValueError("DriftFunction must include an action function to find critical points.")

        action = lambda z: self.drift_fn.action(z)
        x_vals = np.linspace(*self.x_range, self.x_n)
        y_vals = np.linspace(*self.y_range, self.y_n)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = X + 1j * Y

        points = []
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                try:
                    res = newton(action, Z[i, j])
                    points.append(res)
                except:
                    pass

        points = np.array(np.unique(np.round(points, 8)))
        return points
