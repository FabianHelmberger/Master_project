import numpy as np

from .cl_simulation import ComplexLangevinSimulation
from src.numba_target import use_cuda

if use_cuda: 
    from numba import cuda # type: ignore 

class GPU_handler:
    def __init__(self, sim: ComplexLangevinSimulation, exception = [None]):
        self.sim = sim
        self.exception = exception
        self.define_tensor_data()
        if use_cuda: self.to_device()

    def define_tensor_data(self):
        """
        Collect np.ndarray attributes into tensor_data.
        tensor_data stores only the attribute names (not the tensors themselves).
        """
        self.sim.tensor_names = []
        for attr_name, attr in self.sim.__dict__.items():
            if isinstance(attr, np.ndarray):
                self.sim.tensor_names.append(attr_name)
            elif hasattr(attr, 'define_tensor_data') and callable(attr.define_tensor_data):
                attr.define_tensor_data()

    def to_device(self):
        """
        Transfer all tensors corresponding to tensor_data to the GPU and update the corresponding attributes.
        """
        for attr_name in self.sim.tensor_names:
            if attr_name not in self.exception:
                tensor = getattr(self.sim, attr_name)  # Get the tensor from the attribute
                device_tensor = cuda.to_device(tensor)  # Transfer to GPU
                setattr(self.sim, attr_name, device_tensor)  # Update the original attribute

    def to_host(self):
        """
        Transfer all tensors corresponding to tensor_data from the GPU to the host and update the corresponding attributes.
        """
        for attr_name in self.sim.tensor_names:
            if attr_name not in self.exception:
                device_tensor = getattr(self.sim, attr_name)  # Get the device tensor
                host_tensor = device_tensor.copy_to_host()  # Copy tensor from GPU to host
                setattr(self.sim, attr_name, host_tensor)  # Update the original attribute