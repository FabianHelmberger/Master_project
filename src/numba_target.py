"""
Switch between CPU Numba version and CUDA version.
"""
import os
import math

# from packaging.version import Version
# import numba
# _required_numba_version = '0.42.0'
# if Version(numba.__version__) < Version(_required_numba_version):
#     print("Warning: This program requires Numba version {} or higher.".format(_required_numba_version))
#     print("Current numba version: {}".format(numba.__version__))

target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()
try:
    import numba
except ModuleNotFoundError:
    print("Numba not installed! Please install Numba: http://numba.pydata.org/")
    target = 'python'

if target == 'cuda':
    from numba import cuda
    myjit = numba.jit(nogil=True, fastmath=True)  # cuda.jit(device=True)
    mycudajit = cuda.jit
    prange = range
    use_cuda = True
    use_python = False
    use_numba = False
    print("Using CUDA")

elif target == 'python':
    myjit = lambda a : a
    mycudajit = lambda a : a
    prange = range
    use_cuda = False
    use_python = True
    use_numba = False
    print("Using Python")

elif target == 'numba':
    import numba
    myjit = numba.jit(nogil=True, fastmath=True)
    mycudajit = lambda a : a
    prange = numba.prange
    use_cuda = False
    use_python = False
    use_numba = True
    print("Using Numba")
    
else:
    print("Unknown Numba target device: " + target)
    exit()

##############################################################################

threadsperblock = 256

##############################################################################

# Unique counter for compiling python or CUDA functions
_unique_counter = 0
##############################################################################
# dict with compiled kernels, indexed with original function
compiled_kernels = {}
compiled_act_kernels = {}
##############################################################################

#TODO: create decorator version of functionality? i.e. @my_parallel_jit



def my_parallel_loop(kernel_function, iter_max, *args, stream=None):
    """Perform parallel loop over a kernel function either on CPU
    (using Numba's prange) or on GPU (using a compiled cuda kernel).

    :param kernel_function: kernel function with arguments (i, *args)
    :param iter_max: maximum index for iteration
    :param args: optional arguments
    """
    global _unique_counter

    if use_python:
        # loop over the function directly:
        for xi in range(iter_max):
            kernel_function(xi, *args)

    elif use_cuda:
        # if not hasattr(kernel_function, 'compiled_cuda_kernel'):
        if kernel_function not in compiled_kernels:
            # Create compiled cuda kernel for given kernel function

            # Create string of arguments 'c1, c2, c3, [...]' for len(args)
            args_string = ', '.join('c' + str(i) for i in range(len(args)))
            _unique_counter += 1
            try:
                original_name = kernel_function.py_func.__name__
            except AttributeError:
                # Probably running the CUDA simulator which does not provide the name
                original_name = "no_name"
            code = """def {}_cuda_kernel(iter_max, {}):
                          xi = cuda.grid(1)
                          if xi < iter_max:
                              _kernel_function_{}(xi, {})""".format(original_name, args_string, _unique_counter, args_string)
            locals_copy = locals()
            globals()['_kernel_function_{}'.format(_unique_counter)] = kernel_function
            exec(code, globals(), locals_copy)
            cuda_kernel = locals_copy[original_name + '_cuda_kernel']
            # kernel_function.compiled_cuda_kernel = cuda.jit(cuda_kernel)  # alternative: cuda.jit(fastmath=True)(cuda_kernel)
            compiled_kernels[kernel_function] = cuda.jit(cuda_kernel)
            
        kernel_function = compiled_kernels[kernel_function]
        # Call the compiled cuda kernel function:
        blockspergrid = math.ceil(iter_max / threadsperblock)
        kernel_function[blockspergrid, threadsperblock,stream](iter_max, *args)

    # else: # use_numba
    #     if not hasattr(kernel_function, 'compiled_numba_prange'):
    #         # Create compiled prange loop for given kernel function

    #         # Create string of arguments 'c1, c2, c3, [...]' for len(args)
    #         args_string = ', '.join('c' + str(i) for i in range(len(args)))
    #         _unique_counter += 1
    #         original_name = kernel_function.py_func.__name__
    #         code = """def {}_numba_prange(iter_max, {}):
    #                       for xi in prange(iter_max):
    #                           _kernel_function_{}(xi, {})""".format(original_name, args_string, _unique_counter, args_string)
    #         locals_copy = locals()
    #         globals()['_kernel_function_{}'.format(_unique_counter)] = kernel_function
    #         exec(code, globals(), locals_copy)
    #         numba_prange = locals_copy[original_name + '_numba_prange']
    #         kernel_function.compiled_numba_prange = numba.jit(numba_prange, nogil=True, fastmath=True, parallel=True)

    #     # Call the compiled numba prange function:
    #     kernel_function.compiled_numba_prange(iter_max, *args)

    else: # use_numba
            # Cache or compile the Numba-prange function
        if kernel_function not in compiled_kernels:
            # Define the function dynamically
            @numba.njit(parallel=True, nogil=True, fastmath=True)
            def numba_prange_func(iter_max, *args):
                for idx in prange(iter_max):
                    kernel_function(idx, *args)
            
            # Store in the dictionary
            compiled_kernels[kernel_function] = numba_prange_func
        
        # Retrieve the compiled version
        compiled_function = compiled_kernels[kernel_function]
        
        # Execute the compiled function
        compiled_function(iter_max, *args)

def my_act_parallel_loop(kernel_function, iter_max, act_matrix, *args, stream=None):
    """Perform parallel loop over a kernel function either on CPU
    (using Numba's prange) or on GPU (using a compiled cuda kernel).

    :param kernel_function: kernel function with arguments (i, *args)
    :param iter_max: maximum index for iteration
    :param act_matrix: actovation matrix
    :param args: optional arguments
    """
    global _unique_counter


    if use_python:
        # loop over the function directly:
        for xi in range(iter_max):
            if act_matrix[xi]: kernel_function(xi, act_matrix, *args)

    elif use_cuda:
        # if not hasattr(kernel_function, 'compiled_cuda_kernel'):
        if kernel_function not in compiled_act_kernels:
            # check if actiavtion is valid
            assert act_matrix.size == iter_max, "Activation matrix in valid"
            # Create string of arguments 'c1, c2, c3, [...]' for len(args)
            args_string = ', '.join('c' + str(i) for i in range(len(args)))
            _unique_counter += 1
            try:
                original_name = kernel_function.py_func.__name__
            except AttributeError:
                # Probably running the CUDA simulator which does not provide the name
                original_name = "no_name"
                
            code = """def {}_cuda_kernel(iter_max, act_matrix, {}):
                          xi = cuda.grid(1)
                          if xi < iter_max and act_matrix[xi]:
                              _kernel_function_{}(xi, act_matrix, {})""".format(original_name, args_string, _unique_counter, args_string)
            locals_copy = locals()
            globals()['_kernel_function_{}'.format(_unique_counter)] = kernel_function
            exec(code, globals(), locals_copy)
            cuda_kernel = locals_copy[original_name + '_cuda_kernel']
            compiled_act_kernels[kernel_function] = cuda.jit(cuda_kernel)
            
        kernel_function = compiled_act_kernels[kernel_function]
        # Call the compiled cuda kernel function:
        blockspergrid = math.ceil(iter_max / threadsperblock)
        kernel_function[blockspergrid, threadsperblock,stream](iter_max, act_matrix, *args)

    else: # use_numba
            # Cache or compile the Numba-prange function
        if kernel_function not in compiled_act_kernels:
            # Define the function dynamically
            @numba.njit(parallel=True, nogil=True, fastmath=True)
            def numba_prange_func(iter_max, act_matrix, *args):
                for idx in prange(iter_max):
                    if act_matrix[idx]: kernel_function(idx, act_matrix, *args)
            
            # Store in the dictionary
            compiled_act_kernels[kernel_function] = numba_prange_func
        
        # Retrieve the compiled version
        compiled_function = compiled_act_kernels[kernel_function]
        
        # Execute the compiled function
        compiled_function(iter_max, act_matrix, *args)