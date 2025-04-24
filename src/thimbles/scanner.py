import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product

from thimbles.drift import DriftModel
from thimbles.critical import CriticalPointFinder
from thimbles.analyzer import ThimbleAnalyzer

class ThimbleScanner:
    def __init__(self, make_drift_fn, sigma, lamb, 
                 mass_mod_vals=None, pullback_vals=None, 
                 steps=200000, n_jobs=-1):
        self.make_drift_fn = make_drift_fn
        self.sigma = sigma
        self.lamb = lamb
        self.steps = steps
        self.n_jobs = n_jobs

        self.mass_mod_vals = mass_mod_vals if mass_mod_vals is not None else np.logspace(np.log10(0.6), np.log10(4.0), 50)
        self.pullback_vals = pullback_vals if pullback_vals is not None else np.logspace(np.log10(4.0), np.log10(80.0), 50)

        self.param_grid = list(product(self.mass_mod_vals, self.pullback_vals))
        self.results = None

    def _compute_num(self, pullback, mass_mod):
        drift_fn = self.make_drift_fn(self.sigma, self.lamb, pullback, mass_mod)
        df = DriftModel(drift_fn)
        cpf = CriticalPointFinder(df)
        critical_points = cpf.find()
        ana = ThimbleAnalyzer(df, steps=self.steps)
        return ana.count_relevant_thimbles(critical_points)

    def run_scan(self):
        print(f"Scanning {len(self.param_grid)} parameter pairs...")
        self.results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_num)(pb, mm) for (mm, pb) in tqdm(self.param_grid, desc="Sweeping parameter space")
        )
        self.results = np.array(self.results).reshape((len(self.mass_mod_vals), len(self.pullback_vals)))

    def plot_results(self):
        if self.results is None:
            raise RuntimeError("Scan not yet run. Call `run_scan()` first.")

        plt.figure(figsize=(8, 6))
        plt.imshow(self.results.T, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')

        plt.xlabel('Mass Modification Values')
        plt.ylabel('Pullback Values')
        plt.title('Heatmap of Relevant Thimbles Count')

        plt.xticks(np.arange(len(self.mass_mod_vals)), [f'{m:.2f}' for m in self.mass_mod_vals], rotation=90)
        plt.yticks(np.arange(len(self.pullback_vals)), [f'{p:.2f}' for p in self.pullback_vals])
        plt.colorbar(label='Relevant Thimbles Count')
        plt.tight_layout()
        plt.show()