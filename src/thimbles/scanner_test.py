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
import numba

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

mass_mod_vals = np.logspace(np.log10(0.6), np.log10(4.0), 20)
pullback_vals = np.logspace(np.log10(4.0), np.log10(80.0), 20)

scanner = ThimbleScanner(
    make_drift_fn=make_mod_drift,
    mass_mod_vals = mass_mod_vals,
    pullback_vals = pullback_vals,
    sigma=-1 + 4j,
    lamb=2.0,
    steps=20000  # or whatever you prefer
)
scanner.run_scan()
scanner.plot_results()