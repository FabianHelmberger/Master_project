
# # numerical integration
# # fast implementation
# import numpy as np
# from scipy.integrate import quad
# from numba import jit


# @jit
# def density_unmod(x, sigma, lamb):
#     return np.exp(-sigma/2*x**2-lamb/4*x**4)

# @jit
# def density_gaussianmod(x, sigma, lamb, pullback, mass_mod):
#     return np.exp(-sigma/2*x**2-lamb/4*x**4) + pullback*np.exp(-mass_mod/2*x**2)

# @jit
# def n_power(x, order):
#     return x**order

# @jit
# def dse_n_power(x, order, sigma, lamb):
#     return order*x**(order-1) - x**order*(sigma*x+lamb*x**3)

# @np.vectorize
# def n_moment_expval_unmod(order, sigma, lamb):
#     real_part = quad(lambda x: np.real(density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     partition_sum= real_part + 1j * imag_part

#     real_part = quad(lambda x: np.real(n_power(x, order)*density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(n_power(x, order)*density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     results = real_part + 1j * imag_part
#     return results / partition_sum

# @np.vectorize
# def n_moment_expval_unmod(order, sigma, lamb):
#     real_part = quad(lambda x: np.real(density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     partition_sum= real_part + 1j * imag_part

#     real_part = quad(lambda x: np.real(n_power(x, order)*density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(n_power(x, order)*density_unmod(x, sigma, lamb)), -np.inf, np.inf)[0]
#     results = real_part + 1j * imag_part
#     return results / partition_sum

# @np.vectorize
# def dse_n_moment_expval_gaussianmod(order, sigma, lamb, pullback, mass_mod):
#     real_part, real_error = quad(lambda x: np.real(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)
#     imag_part, imag_error = quad(lambda x: np.imag(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)
#     partition_sum= real_part + 1j * imag_part

#     real_part = quad(lambda x: np.real(dse_n_power(x, order, sigma, lamb)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(dse_n_power(x, order, sigma, lamb)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     results = real_part + 1j * imag_part

#     return results / partition_sum

# @np.vectorize
# def n_moment_expval_R(order, sigma, lamb, pullback, mass_mod):
#     real_part = quad(lambda x: np.real(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     partition_sum= real_part + 1j * imag_part

#     real_part = quad(lambda x: np.real(n_power(x, order)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(n_power(x, order)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     results = real_part + 1j * imag_part

#     return results / partition_sum

# @np.vectorize
# def n_moment_expval_gaussianmod(order, sigma, lamb, pullback, mass_mod):
#     real_part = quad(lambda x: np.real(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     partition_sum= real_part + 1j * imag_part

#     real_part = quad(lambda x: np.real(n_power(x, order)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     imag_part = quad(lambda x: np.imag(n_power(x, order)*density_gaussianmod(x, sigma, lamb, pullback, mass_mod)), -np.inf, np.inf)[0]
#     results = real_part + 1j * imag_part

#     return results / partition_sum

# import scipy
# # define analytic resul for n_moment_dse wrt R
# @np.vectorize
# def n_moment_dse_R_analytic(order, r, a, sigma, lamb):
#     if order > 1:
#         out = order*scipy.special.factorial2(order-2)/np.power(a, (order-1)//2)
#     else: 
#         out = 1
#     out -= sigma*scipy.special.factorial2(order)/ np.power(a, (order+1)//2) 
#     out -= lamb*scipy.special.factorial2(order+2)/np.power(a, (order+3)//2)
#     return out

import numpy as np
from scipy.integrate import quad
from numba import jit

# define obs that vanishes wrt rho
def n_moment_dse(x, order, sigma, lamb):
    return order*x**(order-1)-x**order * (sigma * x + lamb * x*x*x)

def n_moment(x, order):
    return x**order

# define rho and R
@jit
def action(x, sigma, lamb): return sigma/2*x**2+lamb/4*x**4

@jit
def rho(x, sigma, lamb): return np.exp(-action(x, sigma, lamb))

@jit
def R(x, r, a): return r*np.exp(-a*x**2/2)

@jit
def rho_tilde(x, r, a, sigma, lamb): return rho(x, sigma, lamb)+R(x, r, a)

# define partition sums
def z_rho(sigma, lamb): 
    real_part = quad(lambda x: np.real(rho(x, sigma, lamb)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(rho(x, sigma, lamb)), -np.inf, np.inf)[0]
    return real_part+1j*imag_part

def z_rho_tilde(r, a, sigma, lamb): 
    real_part = quad(lambda x: np.real(rho_tilde(x, r, a, sigma, lamb)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(rho_tilde(x, r, a, sigma, lamb)), -np.inf, np.inf)[0]
    return real_part+1j*imag_part

def z_R(r, a): 
    real_part = quad(lambda x: np.real(R(x, r, a)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(R(x, r, a)), -np.inf, np.inf)[0]
    return real_part+1j*imag_part

# define exp val of n_moment_dse wrt rho_tilde
@np.vectorize
def n_moment_dse_rho_tilde(order, r, a, sigma, lamb):
    real_part = quad(lambda x: np.real(rho_tilde(x, r, a, sigma, lamb)*n_moment_dse(x, order, sigma, lamb)), 
                     -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(rho_tilde(x, r, a, sigma, lamb)*n_moment_dse(x, order, sigma, lamb)), 
                     -np.inf, np.inf)[0]
    out = real_part + 1j*imag_part
    return out / z_rho_tilde(r, a, sigma, lamb)
    
# define exp val of n_moment_dse wrt rho
@np.vectorize
def n_moment_dse_rho(order, sigma, lamb):
    real_part = quad(lambda x: np.real(rho(x, sigma, lamb)*n_moment_dse(x, order, sigma, lamb)), 
                     -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(rho(x, sigma, lamb)*n_moment_dse(x, order, sigma, lamb)), 
                     -np.inf, np.inf)[0]
    out = real_part + 1j*imag_part
    return out / z_rho(sigma, lamb)

# define exp val of n_moment_dse wrt R
@np.vectorize
def n_moment_dse_R(order, r, a, sigma, lamb):
    real_part = quad(lambda x: np.real(R(x, r, a)*n_moment_dse(x, order, sigma, lamb)), 
                     -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(R(x, r, a)*n_moment_dse(x, order, sigma, lamb)), 
                     -np.inf, np.inf)[0]
    out = real_part + 1j*imag_part
    return out / z_R(r, a)

     
import scipy
# define analytic resul for n_moment_dse wrt R
@np.vectorize
def n_moment_dse_R_analytic(order, r, a, sigma, lamb):
    if  order > 1:
        out = order/np.power(a,(order-1)//2) * scipy.special.factorial2(order-2)
    else: out = 1
    out -= sigma/np.power(a, (order+1)//2)*scipy.special.factorial2(order)
    out -= lamb/np.power(a, (order+2)//2)*scipy.special.factorial2(order+2)
    return r*out

@np.vectorize
def n_moment_R(order, r, a):
    real_part = quad(lambda x: np.real(R(x, r, a)*n_moment(x, order)), -np.inf, np.inf)[0]
    imag_part = quad(lambda x: np.imag(R(x, r, a)*n_moment(x, order)), -np.inf, np.inf)[0]
    out = real_part + 1j*imag_part
    return out / z_R(r, a)