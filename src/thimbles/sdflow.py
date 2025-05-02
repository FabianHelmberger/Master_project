
# src/thimbles/sdflow.py
import numpy as np
from .drift import DriftModel

from scipy.integrate import solve_ivp

class SDFlow:
    def __init__(self, dm: DriftModel, steps=20000, eps=1e-3, max_abs_r=10., bound=1e-3,  nudge_amplitude = 0.001, num_nudges = 2):
        self.dm = dm
        self.steps = steps
        self.eps = eps
        self.max_abs_r = max_abs_r
        self.bound = bound
        self.thimbles = None
        self.thimbles_dual = None
        self.nudge_amplitude = nudge_amplitude
        self.num_nudges = num_nudges

    def flow_all(self, critical_points: np.ndarray) -> np.ndarray:
        all_thimbles = []
        all_dual_thimbles = []

        import tqdm
        for cp in tqdm.tqdm(critical_points):
            angles = np.linspace(0, 2 * np.pi, self.num_nudges, endpoint=False)
            perturbations = self.nudge_amplitude * np.exp(1j * angles)

            combined_sol = []
            for pert in perturbations:
                sol = self.flow(cp + pert, dual=False)
                sol = sol[~np.isnan(sol)]
                combined_sol.append(sol)
            # combined_sol = np.concatenate(combined_sol)
            all_thimbles.append(combined_sol)

            combined_sol = []
            for pert in perturbations:
                sol = self.flow(cp + pert, dual=True)
                sol = sol[~np.isnan(sol)]
                combined_sol.append(sol)
            # combined_sol = np.concatenate(combined_sol)
            all_dual_thimbles.append(combined_sol)

        self.thimbles = all_thimbles
        self.thimbles_dual = all_dual_thimbles
    

    def solve_floweq(flow_equation, crit_point, epsilon = 0.001, t_span = (0, 100), t_points=10, events = None, **kwargs):
        
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        sol = []
        directions_num = 2
        angles = np.linspace(0+0.01, 2*np.pi+0.01, directions_num, endpoint=False)
        perturbations = epsilon * np.exp(1j * angles)

        sigma = kwargs.get("sigma", 5*np.exp(1j*np.pi*3/4)) 
        lamb = kwargs.get("lamb", 1) 
        pullback = kwargs.get("pullback", 1) 
        mass_modification = kwargs.get("mass_modification", 5)

        def stop_event(t, z, *args, **kwargs):
            x, y = z
            return max(x, y) - 10  # Triggers when either x or y reaches 10

        # Ensure the event is terminal
        stop_event.terminal = True

        for perturb in perturbations:
            z0 = crit_point + perturb
            z0 = [np.real(crit_point + perturb), np.imag(crit_point + perturb)]

            solution = solve_ivp(flow_equation, t_span, y0 = z0, args = (sigma, lamb, pullback, mass_modification), t_eval=t_eval, method="BDF", events=stop_event)
            sol.append(solution.y[0] + 1j * solution.y[1])
        return sol


    def flow(self, r0: complex, dual = False) -> np.ndarray:
        eps = -self.eps if dual else self.eps
        rlist, r = [], r0

        for _ in range(self.steps):
            dS = self.dm.evaluate(r)
            dS_abs = np.abs(dS)

            if eps * dS_abs > self.bound:
                r += self.bound / dS_abs * dS.conjugate()
            else:
                r += eps * dS.conjugate()

            if np.abs(r) >= self.max_abs_r:
                break

            rlist.append(r)

        return np.array(rlist)