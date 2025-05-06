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
        self.results = Parallel(n_jobs=self.n_jobs, batch_size="auto")(
            delayed(self._compute_num)(pb, mm) for (mm, pb) in tqdm(self.param_grid, desc="Sweeping parameter space")
        )
        self.results = np.array(self.results).reshape((len(self.mass_mod_vals), len(self.pullback_vals)))

    def plot_results(self):
        if self.results is None:
            raise RuntimeError("Scan not yet run. Call `run_scan()` first.")

        plt.figure(figsize=(8, 6))

        X, Y = np.meshgrid(self.mass_mod_vals, self.pullback_vals)
        plt.pcolormesh(X, Y, self.results.T, shading='auto', cmap='viridis')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel('Mass Modification Values')
        plt.ylabel('Pullback Values')
        plt.title('Heatmap of Relevant Thimbles Count')

        # plt.xticks(np.arange(len(self.mass_mod_vals)), [f'{m:.2f}' for m in self.mass_mod_vals], rotation=90)
        # plt.yticks(np.arange(len(self.pullback_vals)), [f'{p:.2f}' for p in self.pullback_vals])
        plt.colorbar(label='Relevant Thimbles Count')
        plt.tight_layout()
        plt.savefig("thimble_scan.png")
        plt.show()

    def plot_thimble_flows(self, pullback, mass_mod, x_range=(-5, 5), y_range=(-5, 5), grid_res=400):
        import matplotlib.pyplot as plt
        import numpy as np
        import scienceplots
        from scipy.integrate import solve_ivp
        from thimbles.drift import DriftModel
        from thimbles.critical import CriticalPointFinder

        plt.style.use(['science'])

        # Create drift model
        drift_fn = self.make_drift_fn(self.sigma, self.lamb, pullback, mass_mod)
        df = DriftModel(drift_fn)

        # Find critical points
        cpf = CriticalPointFinder(df)
        critical_points = cpf.find()

        # Define the flow equation
        def flow_equation(t, z):
            dz = df.eval(z[0] + 1j * z[1])
            return [np.real(dz), np.imag(dz)]

        # Generate perturbations around critical points
        epsilon = 1e-3
        angles = np.linspace(0.01, 2 * np.pi + 0.01, 2, endpoint=False)
        perturbations = epsilon * np.exp(1j * angles)

        flow_solutions = []
        for crit in critical_points:
            for perturb in perturbations:
                z0 = crit + perturb
                init = [np.real(z0), np.imag(z0)]
                sol = solve_ivp(flow_equation, t_span=(0, 100), y0=init, t_eval=np.linspace(0, 100, 500))
                flow_solutions.append(sol.y[0] + 1j * sol.y[1])
        print(f"Generated {len(flow_solutions)} flow lines.")
        # Evaluate drift magnitude on a grid
        x = np.linspace(*x_range, grid_res)
        y = np.linspace(*y_range, grid_res)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        mags = np.abs(df.eval(Z))

        # Plotting
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle(
            rf"$\tilde{{\rho}} = \exp(-\sigma/2z^2-\lambda/4z^4) \quad \sigma={self.sigma:.2g}, \lambda={self.lamb:.2g}$"
        )

        im = ax.imshow(-np.log10(mags), extent=[*x_range, *y_range], origin='lower',
                    cmap='hot', alpha=0.75, vmin=-3, vmax=3)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r'$\log(|\partial_z \tilde{S}|)$', rotation=90)

        for sol in flow_solutions:
            ax.plot(sol.real, sol.imag, color='gray', ls=':', alpha=0.6, lw=0.6)
            ax.plot(sol.real, sol.imag, color='blue', ls='solid')

        ax.scatter(critical_points.real, critical_points.imag,
                s=10, facecolors='none', edgecolors='black', lw=0.6, label="zero drift")

        ax.plot([], [], color='black', ls=':', label="unstable thimble")
        ax.plot([], [], color='black', ls='solid', label="stable thimble")

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel(r"$Re(z)$")
        ax.set_ylabel(r"$Im(z)$")
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=5,
                loc="lower left", mode="expand", borderaxespad=0, ncol=4, handletextpad=1)

        filename = f"thimble_frame_sig_{self.sigma}_lam_{self.lamb}_pb_{pullback}_mm_{mass_mod}.png"
        fig.savefig(filename)
        plt.show()