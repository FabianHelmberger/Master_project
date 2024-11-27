"""
    Complex Langevin simulation module
"""

import numpy as np

# from src.numba_target import use_cuda, myjit, my_pa                                                              rallel_loop, prange, threadSperblock

from .langevin_dynamics import *
from .observables import *
from src.numba_target import use_cuda

class ComplexLangevinSimulation(Observables):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tensor_names = []
        self.define_tensor_data()

        if use_cuda: self.to_device()

    def define_tensor_data(self):
        """
        Collect np.ndarray attributes into tensor_data.
        tensor_data stores only the attribute names (not the tensors themselves).
        """
        self.tensor_names = []
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, np.ndarray):
                self.tensor_names.append(attr_name)
            elif hasattr(attr, 'define_tensor_data') and callable(attr.define_tensor_data):
                attr.define_tensor_data()

    def to_device(self):
        """
        Transfer all tensors corresponding to tensor_data to the GPU and update the corresponding attributes.
        """
        for attr_name in self.tensor_names:
            tensor = getattr(self, attr_name)  # Get the tensor from the attribute
            device_tensor = cuda.to_device(tensor)  # Transfer to GPU
            setattr(self, attr_name, device_tensor)  # Update the original attribute

    def to_host(self):
        """
        Transfer all tensors corresponding to tensor_data from the GPU to the host and update the corresponding attributes.
        """
        for attr_name in self.tensor_names:
            device_tensor = getattr(self, attr_name)  # Get the device tensor
            host_tensor = device_tensor.copy_to_host()  # Copy tensor from GPU to host
            setattr(self, attr_name, host_tensor)  # Update the original attribute
