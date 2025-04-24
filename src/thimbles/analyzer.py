# src/thimbles/analyzer.py
import numpy as np
from .sdflow import SDFlow
from .drift import DriftModel

class ThimbleAnalyzer:
    def __init__(self, drift_fn: DriftModel, steps=200000, nudge_amplitude=0.01, num_nudges=16):
        self.drift_fn = drift_fn
        self.steps = steps
        self.nudge_amplitude = nudge_amplitude
        self.num_nudges = num_nudges

    def count_relevant_thimbles(self, critical_points: np.ndarray) -> int:
        relevant = []

        for cp in critical_points:
            angles = np.linspace(0.01, 2*np.pi + 0.01, self.num_nudges, endpoint=False)
            perturbations = self.nudge_amplitude * np.exp(1j * angles)
            combined_sol = []

            for pert in perturbations:
                sol = SDFlow(self.drift_fn, steps=self.steps, dual=True).run(cp + pert)
                sol = sol[~np.isnan(sol)]
                combined_sol.append(sol)

            combined_sol = np.concatenate(combined_sol)
            if np.any(combined_sol.imag > 0) and np.any(combined_sol.imag < 0):
                relevant.append(cp)

        return len(relevant)