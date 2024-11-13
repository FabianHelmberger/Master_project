"""
    General group and algebra functions
    Grid functions
"""
from .numba_target import myjit
from . import su

import math

"""
    SU(2) group & algebra functions
"""


# product of 4 matrices
@myjit
def mul4(a, b, c, d):
    ab = su.mul(a, b)
    cd = su.mul(c, d)
    abcd = su.mul(ab, cd)
    return abcd


# group add: g0 = g0 + f * g1
@myjit
def add_mul(g0, g1, f):  # TODO: inline explicitly everywhere and remove this function
    return su.add(g0, su.mul_s(g1, f))


# commutator of two su(2) elements
@myjit
def comm(a, b):
    buffer1 = su.mul(a, b)
    buffer2 = su.mul(b, a)
    result = add_mul(buffer1, buffer2, -1)
    return result


"""
    Plaquette functions
"""


# compute general plaquette U_{x, oi*i, oj*j}
@myjit
def plaq(u, idx, i, j, oi, oj, dims, adims):
    idx0 = idx
    idx1 = shift(idx0, i, oi, dims, adims)
    idx2 = shift(idx1, j, oj, dims, adims)
    idx3 = shift(idx2, i, -oi, dims, adims)

    u0 = get_link(u, idx0, i, oi, dims, adims)
    u1 = get_link(u, idx1, j, oj, dims, adims)
    u2 = get_link(u, idx2, i, -oi, dims, adims)
    u3 = get_link(u, idx3, j, -oj, dims, adims)

    # U_{x, i} * U_{x+i, j} * U_{x+i+j, -i} * U_{x+j, -j}
    return mul4(u0, u1, u2, u3)


@myjit
def get_link(u, x, i, oi, dims, adims):
    if oi > 0:
        return su.load(u[x, i])
    else:
        xs = shift(x, i, oi, dims, adims)
        return su.inv(u[xs, i])


# compute a staple
@myjit
def staple(u, x, i, j, oj, dims, adims):
    x0 = x
    x1 = shift(x0, i, +1, dims, adims)
    x2 = shift(x1, j, oj, dims, adims)
    x3 = shift(x2, i, -1, dims, adims)

    u1 = get_link(u, x1, j, +oj, dims, adims)
    u2 = get_link(u, x2, i, -1, dims, adims)
    u3 = get_link(u, x3, j, -oj, dims, adims)

    return su.mul(su.mul(u1, u2), u3)


# compute sum over staples
@myjit
def staple_sum(x, d, u, dims, acc):
    result = su.zero()
    for j in range(len(dims)):
        if j != d:
            for oj in [-1, 1]:
                s = staple(u, x, d, j, oj, dims, acc)
                result = su.add(result, s)
    return result


# compute plaquette sum
@myjit
def plaquettes(x, d, u, dims, adims):
    res = su.zero()
    for j in range(len(dims)):
        if j != d:
            p1 = plaq(u, x, d, j, 1, +1, dims, adims)
            p2 = plaq(u, x, d, j, 1, -1, dims, adims)
            res = su.add(res, p1)
            res = su.add(res, p2)
    return res


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
