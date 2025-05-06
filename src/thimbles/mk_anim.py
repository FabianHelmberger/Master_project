from scipy.integrate import solve_ivp
import numpy as np




import numpy as np
import matplotlib.pyplot as plt

# Set environment variables
import os
os.environ["SCAL_TYPE"] = "complex"
os.environ["PRECISION"] = "single"
os.environ["MY_NUMBA_TARGET"] = "numba"

# Add cle_fun to PYTHON_PATH
import sys
sys.path.append("../../src")

from thimbles import ThimbleScanner
from thimbles.sdflow import SDFlow
from thimbles.drift import 

def make_mod_drift(sigma, lamb, pullback, mass_modification):
    @numba.njit
    def drift(z):
        action = sigma/2*z**2+lamb/4*z**4
        drift = sigma*z+lamb*z**3

        # to prevent overflow
        if np.real(action-mass_modification/2*z**2)<0:
            return drift - pullback*np.exp(action-mass_modification*z**2/2)*(drift - mass_modification*z) / (1+pullback*np.exp(action-mass_modification/2*z**2))
        else: 
            return drift - pullback*(drift - mass_modification*z) / (np.exp(-action+mass_modification/2*z**2)+pullback)

    return drift

df = DriftModel(drift_fn)
sol = SDFlow(drift_fn, steps=self.steps, dual=True).run(cp + pert)


scanner.plot_thimble_flows(pullback=20.0, mass_mod=1.0)

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


# @jit
def drift_func(z, sigma, lamb, pullback, mass_modification):
    action = sigma/2*z**2+lamb/4*z**4
    mod = -mass_modification*z**2/2
    action_mod = action + mod
    drift = sigma*z+lamb*z**3

    if np.real(action_mod) < 0:
        return drift-pullback*(drift-mass_modification*z)*np.exp(action_mod) / (1+pullback*np.exp(action_mod))
    else: 
        return drift-pullback*(drift-mass_modification*z) / (np.exp(-action_mod)+pullback)

drift_func = np.vectorize(drift_func)

# @jit
def flow_equation_stable(t, z, *kwargs):
    z_real, z_imag = z
    z_complex = z_real+1j*z_imag
    flow = np.conj(drift_func(z_complex, *kwargs))
    return [np.real(flow), np.imag(flow)]

# @jit
def flow_equation_unstable(t, z, *kwargs):
    z_real, z_imag = z
    z_complex = z_real+1j*z_imag
    flow = -np.conj(drift_func(z_complex, *kwargs))
    return [np.real(flow), np.imag(flow)]


lamb = 1
sigma_abs = 3

start_angle = 0
stop_angle = 2*np.pi

num = 6
angles = np.linspace(start_angle, stop_angle, num)
sigma_vals = sigma_abs*(np.cos(angles) + 1j*np.sin(angles))

mass_modification = 0.5
pullback = 120

import matplotlib.pyplot as plt
from scipy.optimize import root
import scienceplots

plt.style.use(['science', 'ieee'])


def F_real(z_real):
    z = z_real[0] + 1j * z_real[1]
    Fz = F(z)
    return np.array([Fz.real, Fz.imag])

import time

for idx_sig, sigma in enumerate(sigma_vals):
    print(idx_sig)
    t_final = time.time()
    x_range = (-3, 3)
    y_range = (-3, 3)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust spacing between subplots
    deg = np.degrees(np.angle(sigma))
    fig.suptitle(rf"$\tilde{{\rho}} = \exp(-\sigma/2z^2-\lambda/4z^4) + r \exp(-\alpha/2z^2)$"+"\n"\
                 rf"$\text{{arg}}(\sigma) = {{{np.round((deg+360)%360, 3)}}}$°, $|\sigma| = {{{np.round(np.abs(sigma), 3)}}}$, $r={{{np.round(pullback, 3)}}}$"+\
                 rf", $\alpha={{{mass_modification}}}$")

    # mark zeros
    # Define the function whose zeros we want to find
    @np.vectorize
    def F(z):
        action_mod = ((sigma - mass_modification) / 2) * z**2 + (lamb / 4) * np.power(z, 4)
        if np.real(action_mod+np.log(pullback+0j)) > 100:
            return mass_modification * z + ((sigma - mass_modification) * z + lamb * z**3)/pullback * np.exp(-action_mod)
        return mass_modification * z + ((sigma - mass_modification) * z + lamb * z**3) / (1 + pullback * np.exp(action_mod))
    
    x_vals = np.linspace(-5, 5, 50)  # Real part of z
    y_vals = np.linspace(-5, 5, 50)  # Imaginary part of z
    initial_guesses = [(x, y) for x in x_vals for y in y_vals]

    # Store found roots
    roots = []

    # Solve for each initial guess
    find_roots = 0
    solve_flow = 0
    im_show = 0
    for z0 in initial_guesses:
        t0 = time.time()
        sol = root(F_real, z0, method='hybr')
        find_roots += time.time() - t0

        if sol.success:
            root_found = sol.x[0] + 1j * sol.x[1]

            # Avoid duplicates (consider numerical precision)
            if not any(np.isclose(root_found, r, atol=1e-6) for r in roots):

                roots.append(root_found)
                # stable
                t0 = time.time()
                solutions = solve_floweq(flow_equation_unstable, crit_point=root_found, t_span = (0, 50), t_points=100000, sigma = sigma, lamb=lamb, pullback=pullback, mass_modification=mass_modification)
                solve_flow += time.time() - t0
                
                relevant = False
                conc_sol = np.concatenate(solutions, axis = 0)
                if any(conc_sol.imag>0) and any(conc_sol.imag<0): relevant = True

                for sol in solutions:
                    if relevant:
                        ax.plot(sol.real, sol.imag, color = 'blue', ls = ':')  # Set transparency for better visibility
                    else: 
                        ax.plot(sol.real, sol.imag, color = 'gray', ls = ':', alpha=0.6, lw = 0.6)  # Set transparency for better visibility
    
                # stable
                t0 = time.time()                
                solutions = solve_floweq(flow_equation_stable, crit_point=root_found, t_span = (0, 50), t_points=100000, sigma = sigma, lamb=lamb, pullback=pullback, mass_modification=mass_modification)
                solve_flow += time.time() - t0
                for sol in solutions:
                    if relevant: 
                        ax.plot(sol.real, sol.imag, color = 'blue', ls = "solid")  # Set transparency for better visibility
                    else: 
                        ax.plot(sol.real, sol.imag, color = 'gray', ls = "solid", alpha=0.6, lw = 0.6)  # Set transparency for better visibility
        

    roots = np.array(roots)

    x = np.linspace(*x_range, 400)
    y = np.linspace(*y_range, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    drift_values = drift_func(Z, sigma=sigma, lamb=lamb, pullback=pullback, mass_modification=mass_modification)
    mags = np.abs(drift_values)
    # threshold = 1e-1
    # mags[mags < threshold] = 0
    t0 = time.time() 
    im = ax.imshow(-np.log10(mags), extent=[*x_range, *y_range], origin='lower', cmap='hot', alpha=0.75, vmin=-3, vmax=3)
    cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.805)
    cbar.ax.set_ylabel(rf'$\log(|\partial_z \tilde{{S}}|)$', rotation=90)
    im_show += time.time() - t0

    # x = np.linspace(*x_range, 35)
    # y = np.linspace(*y_range, 35)
    # X, Y = np.meshgrid(x, y)
    # Z = X + 1j * Y
    # drift_values = np.conj(drift_func(Z, sigma=sigma, lamb=lamb, pullback=pullback, mass_modification=mass_modification))
    
    # U = np.real(drift_values)
    # V = np.imag(drift_values)
    # mags = np.abs(drift_values)
    # threshold = 1e-1
    # mags[mags<threshold] = np.inf

    # ax.quiver(X, Y, U/mags, V/mags, color="blue", angles="xy", scale=45, alpha=0.9)
    ax.set_ylim(*y_range)
    ax.set_xlim(*x_range)
    ax.scatter(roots.real, roots.imag, s=10, facecolors='none', edgecolors='gray', lw = 0.6, alpha = 0.6)

    # show singularities in drift
    roots = []
    for z0 in initial_guesses:
        sol = root(lambda x: 1/F_real(x), z0, method='hybr')
        if sol.success:
            root_found = sol.x[0] + 1j * sol.x[1]
            if not any(np.isclose(root_found, r, atol=1e-6) for r in roots):
                roots.append(root_found)
    roots = np.array(roots)
    ax.scatter(roots.real, roots.imag, s=10, facecolors='none', edgecolors='gray', marker="^", lw = 0.6, alpha = 0.6)

    ax.set_xlabel(rf"$Re(z)$")
    ax.set_ylabel(rf"$Im(z)$")

    ax.scatter([], [], s=10, facecolors='none', edgecolors='black', marker="^", lw = 0.6, label = "singular drift")
    ax.scatter([], [], s=10, facecolors='none', edgecolors='black', lw = 0.6, label = "zero drift")
    ax.plot([], [], color = 'black', ls = ':', label = "unstable thimble")
    ax.plot([], [], color = 'black', ls = 'solid', label = "stable thimble")

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize = 5, loc="lower left",mode="expand", borderaxespad=0, ncol=4, handletextpad=1)
    fig.savefig(f"animated_thimbles/gaussian_mod/{idx_sig}")