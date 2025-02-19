
# numerical integration
# fast implementation
import numpy as np
from scipy.integrate import quad
from numba import jit


@jit
def density_unmod(x, sigma, lamb):
    return np.exp(-sigma/2*x**2-lamb/4*x**4)

@jit
def density_gaussianmod(x, sigma, lamb, pullback, mass_mod):
    return np.exp(-sigma/2*x**2-lamb/4*x**4) + pullback*np.exp(-mass_mod/2*x**2)

@jit
def n_power(x, order):
    return x**order

@jit
def dse_n_power(x, order, sigma, lamb):
    return order*x**(order-1) - x**order*(sigma*x+lamb*x**3)

@np.vectorize
def n_moment_expval_unmod(order, sigma, lamb):
    real_part = quad(lambda x: np.real(density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
    partition_sum= real_part + 1j * imag_part

    real_part = quad(lambda x: np.real(n_power(x, order)*density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(n_power(x, order)*density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
    results = real_part + 1j * imag_part
    return results / partition_sum


@np.vectorize
def dse_n_moment_expval_gaussianmod(order, sigma, lamb, pullback, mass_mod):
    real_part, real_error = quad(lambda x: np.real(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)
    imag_part, imag_error = quad(lambda x: np.imag(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)
    partition_sum= real_part + 1j * imag_part

    real_part = quad(lambda x: np.real(dse_n_power(x, order, sigma, lamb)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(dse_n_power(x, order, sigma, lamb)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
    results = real_part + 1j * imag_part

    return results / partition_sum

@np.vectorize
def n_moment_expval_gaussianmod(order, sigma, lamb, pullback, mass_mod):
    real_part = quad(lambda x: np.real(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
    partition_sum= real_part + 1j * imag_part

    real_part = quad(lambda x: np.real(n_power(x, order)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(n_power(x, order)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
    results = real_part + 1j * imag_part

    return results / partition_sum