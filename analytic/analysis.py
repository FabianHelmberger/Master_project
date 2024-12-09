from sympy import symbols, exp, sqrt, diff
from sympy.functions.special.bessel import besselk

class Analysis:
    def __init__(self):
        self.funcs = {}

    def set_func(self, func_name, func):
        self.funcs[func_name] = func

    def get_func(self, func_name):
        if func_name not in self.funcs:
            raise KeyError(f"{func_name} is not set")
        return self.funcs[func_name]
    
    def set_derivative(self, func_name, diff_var, diff_order: int):
        func = self.get_func(func_name)

        diff_func_name = f'{func_name}_diff({diff_var}_{diff_order})'
        diff_func = diff(func, diff_var, diff_order)
        self.funcs[diff_func_name] = diff_func

    def n_moment_phi4(self, func_name, mass, n):
        moment_func_name = f"{n}_moment_phi4"
        if n % 2 == 0:
            order = n // 2
            deriv_name = f"{func_name}_diff({mass}_{order})"
            func = self.get_func(func_name)
            if not deriv_name in self.funcs:
                self.set_derivative(func_name, mass, order)
            deriv_func = self.get_func(deriv_name)
            moment_func = (-2)**order * deriv_func / func
            self.set_func(moment_func_name, moment_func)
        else: 
            self.set_func(moment_func_name, 0*mass)
    
    def eval(self, func_name, **kwargs):
        func = self.get_func(func_name)
        return complex(func.subs(kwargs).evalf())