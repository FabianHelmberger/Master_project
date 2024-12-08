import numpy as np

import src.scal as scal
import src.utils as utils
from .config import Config
from .constants import Constants

class Lattice(Config, Constants):
    def __init__(self, config: Config):
        super().__init__(**config.__dict__)  # Use super() to handle both parent initializations
        self.config = config
        self.dims = np.array(self.dims, dtype=scal.LATT_TYPE)
        self.sim_dims = np.append(self.trajs, self.dims) # artificial lattice including direction for different trajs
        self.n_dims = len(self.sim_dims)
        self.n_cells = int(np.prod(self.sim_dims))

        cumprod_dims = np.cumprod(self.sim_dims[::-1], dtype=scal.LATT_TYPE)[::-1]
        self.adims = np.zeros(self.n_dims + 1, dtype=scal.LATT_TYPE)
        self.adims[:-1] = cumprod_dims
        self.adims[-1] = 1

    def shift(self, index, dir, amount):
        return utils.shift(index, dir, amount, self.sim_dims[:1], self.adims[:1])

    def get_index(self, pos, traj):
        return utils.get_index(pos, self.dims[:1], traj)
