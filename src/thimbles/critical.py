# src/thimbles/critical.py
import numpy as np
from scipy.optimize import newton
from .drift import DriftModel

class CriticalPointFinder:
    def __init__(self, dm: DriftModel, x_range=[-5, 5], y_range=[-5, 5], x_n=200, y_n=200):
        self.dm = dm
        self.x_range = x_range
        self.y_range = y_range
        self.x_n = x_n
        self.y_n = y_n
        self.critical_points = None

    def find(self) -> np.ndarray:
        analytic = self.dm.critical_points()
        if analytic is not None:
            return analytic

        drift = lambda z: self.dm.evaluate(z)
        x_vals = np.linspace(*self.x_range, self.x_n)
        y_vals = np.linspace(*self.y_range, self.y_n)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = X + 1j * Y

        points = []
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                try:
                    res = newton(drift, Z[i, j])
                    points.append(res)
                except:
                    pass

        points = np.array(np.unique(np.round(points, 8)))
        self.critical_points = points
        return points