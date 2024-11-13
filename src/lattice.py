"""
    general scalar functions
    Grid functions
"""
from .numba_target import myjit


"""
    Shift on d-dimensional grid
"""

@myjit
def shift(xi, i, o, dims, adims):
    res = xi
    di = xi // adims[i + 1]
    wdi = di % dims[i]
    if o > 0:
        if wdi == dims[i] - 1:
            res = res - adims[i]
        res = res + adims[i + 1]
    else:
        if wdi == 0:
            res = res + adims[i]
        res = res - adims[i + 1]
    return int(res)  # needs explicit cast, otherwise 'res' promoted to a float otherwise. this is new behaviour.

@myjit
def get_index(pos, dims, adims):
    index = pos[0]
    for d in range(1, len(dims)):
        index = index * dims[d] + pos[d]
    return index