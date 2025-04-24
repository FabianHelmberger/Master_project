
# src/thimbles/sweep.py
import pandas as pd
import numpy as np
from .drift import DriftFunction
from .critical import CriticalPointFinder
from .analyzer import ThimbleAnalyzer

class ParameterSweep:
    def __init__(self, base_params: dict, sweep_var: str, sweep_values: list, 
                 lambda_abs: float, sigma: complex, tol=0.1, pullback_bounds=(0, 500)):
        self.base_params = base_params
        self.sweep_var = sweep_var
        self.sweep_values = sweep_values
        self.lambda_abs = lambda_abs
        self.sigma = sigma
        self.tol = tol
        self.lower_bound, self.upper_bound = pullback_bounds
        self.results = []

    def run(self) -> pd.DataFrame:
        for value in self.sweep_values:
            params = self.base_params.copy()
            params[self.sweep_var] = value

            lower, upper = self.lower_bound, self.upper_bound

            while abs(upper - lower) > self.tol:
                pullback = (upper + lower) / 2
                params['pullback'] = pullback
                params['lambda'] = self.lambda_abs
                params['sigma'] = self.sigma

                drift_fn = DriftFunction(
                    drift_func=params['drift_func'],
                    params=params,
                    action_func=params.get('action_func'),
                    critical_points_func=params.get('critical_points_func')
                )

                critical_points = CriticalPointFinder(drift_fn).find()
                num_relevant = ThimbleAnalyzer(drift_fn).count_relevant_thimbles(critical_points)

                if num_relevant == 1:
                    upper = pullback
                else:
                    lower = pullback

            result = {
                self.sweep_var: value,
                'pullback_upper': upper,
                'pullback_lower': lower,
                'sigma_abs': abs(self.sigma),
                'sigma_phase': np.angle(self.sigma),
                'lambda_abs': self.lambda_abs,
                'mass_modification': params.get('mass_modification')
            }
            self.results.append(result)

        return pd.DataFrame(self.results)
