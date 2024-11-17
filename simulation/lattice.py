import numpy as np

import src.scal as scal
import src.utils as utils
from .config import Config
from .constants import Constants

class Lattice(Config, Constants):
    def __init__(self, config: Config):
        super().__init__(**config.__dict__)  # Use super() to handle both parent initializations

        self.dims = np.array(self.dims, dtype=scal.LATT_TYPE)
        self.n_dims = len(self.dims)
        self.n_cells = int(np.prod(self.dims))

        cumprod_dims = np.cumprod(self.dims[::-1], dtype=scal.LATT_TYPE)[::-1]
        self.adims = np.zeros(self.n_dims + 1, dtype=scal.LATT_TYPE)
        self.adims[:-1] = cumprod_dims
        self.adims[-1] = 1

    def shift(self, index, dir, amount):
        return utils.shift(index, dir, amount, self.dims, self.adims)

    def get_index(self, pos):
        return utils.get_index(pos, self.dims)
