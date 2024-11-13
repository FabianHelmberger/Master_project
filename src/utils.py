from src.numba_target import myjit

@myjit
def shift(index, dir, amount, dims, adims):
    res = index
    di = index // adims[dir + 1]
    wdi = di % dims[dir]
    
    if amount > 0:
        if wdi == dims[dir] - 1:
            res = res - adims[dir]
        res = res + adims[dir + 1]
    else:
        if wdi == 0:
            res = res + adims[dir]
        res = res - adims[dir + 1]
    return int(res)  # needs explicit cast, otherwise 'res' promoted to a float otherwise. this is new behaviour.



@myjit
def get_index(pos, dims):
    index = pos[0]
    for d in range(1, len(dims)):
        index = index * dims[d] + pos[d]
    return index
