"""
    Complex Langevin evolution module
"""

import cmath, math
import numba
import numpy as np
from numpy import abs as arr_abs
from numpy import around, amax, mean, imag, real, reshape

import src.lattice as l
import src.scal as scal
from src.numba_target import use_cuda, myjit, my_parallel_loop, prange, threadsperblock


# global parameters
DEBUG = False
SQRT2 = math.sqrt(2.0)

# upper and lower limits for the norm of the drift term
DS_MAX_UPPER = 1e12
DS_MAX_LOWER = 1e-12

if use_cuda:
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float32,
    )
    import cupy as cp
    from cupy import abs as arr_abs
    from cupy import around, imag, real, reshape

    def amax(arr):
        return cp.amax(cp.asarray(arr)).get().item()
    
    def mean(arr):
        return cp.mean(cp.asarray(arr)).get().item()
    
    def arr_mean(arr, **kwargs):
        to_cpu = kwargs.get("to_cpu", False)
        del kwargs["to_cpu"]
        ret = cp.mean(cp.asarray(arr), **kwargs)
        if to_cpu:
            ret = cp.asnumpy(ret)
        return ret
    
    def my_round(arr, digits=10):
        return cp.around(arr.real, digits) + cp.around(arr.imag, digits) * 1.0j
else: 
    @myjit
    def my_round(arr, digits=10):
        return np.around(arr.real, digits) + np.around(arr.imag, digits) * 1.0j
    
    def arr_mean(arr, **kwargs):
        ax = kwargs.get("axis", None)
        return mean(arr, axis = ax)
    
class ComplexLangevinSimulation_scalar:
    def __init__(
        self,
        dims = [8, 8],
        dt   = 5e-5,
        phi0    = None, 
        phi1    = None,
        a_0     = None,
        count   = 0,
        mass_real   = 1,
        mass_imag   = 0,
        interaction = 1,
        start_type  = 1,
        init_seed   = 19700101,
        noise_seed  = 31337,
        converged   = False,
        drift_term_type = 1,
        boundary_type   = 0,
        **kwargs
    ):
        # print(f"mean ds max: {kwargs.get('mean_dS_max', None)}")        
        """
        Creates a new simulation object. All required arrays are allocated and initialized in this step.

        :param dims:            lattice dimensions (grid size)
        :param dt:              langevin time step
        :param gamma:           global lattice anisotropy
        :param mass_real:       bare mass_real (lattice dims)
        :param mass_imag:       bare mass_imag (lattice dims)
        :param drift_term_type:     type of action used 
        :param noise_seed:      seed for noise term
        :param converged:       boolean stating if the process has converged
        :param min_drift:       minimal drift to evolve
        :return:
        """

        # basic parameters
        self.dims   = np.array(dims, dtype=np.int32)
        self.n_dims = len(dims)
        self.adims  = np.append(np.cumprod(dims[::-1])[::-1], 1) 
        self.a_0  = a_0
        self.count = count
        self.dt     = dt
        self.l_xpos = np.array(list(np.ndindex(*dims[1:])), dtype=np.int32)
        self.mass_real  = mass_real
        self.mass_imag  = mass_imag
        self.n_cells    = int(np.prod(self.dims))
        self.interaction    = interaction
        self.buffer_field_big = np.empty(self.n_cells*self.dims[0])
        # types
        self.init_seed   = init_seed
        self.start_type     = start_type
        self.boundary_type  = boundary_type
        self.drift_term_type    = drift_term_type
        self.noise_seed     = noise_seed
        self.converged      = converged
        self.sig_digs = int(-np.log10(DS_MAX_LOWER))
        ## arrays

        self.phi0 = np.zeros(
            self.n_cells, 
            dtype=scal.SCAL_FIELD_TYPE
            )
        
        self.phi1 = np.zeros(
            self.n_cells, 
            dtype=scal.SCAL_FIELD_TYPE
            )
        # noise 


        self.eta = np.zeros(
            self.n_cells,
            dtype=scal.SCAL_FIELD_TYPE_REAL
        )

        # drift
        self.dS = np.zeros(
            self.n_cells,
            dtype=scal.SCAL_FIELD_TYPE
        )

        # drift norm
        self.dS_norm = np.zeros(
            self.n_cells,
            dtype=scal.SCAL_FIELD_TYPE_REAL
        )

        # hamilton density
        self.hamilton_dens = np.zeros(
            self.n_cells, 
            dtype=scal.SCAL_FIELD_TYPE
        )
        # hamilton
        self.hamilton = np.zeros(
            self.dims[0], 
            dtype=scal.SCAL_FIELD_TYPE
        )

        # variables for adaptive stepsize
        self.l_dt = [0.0]
        self.l_th = [0.0]
        self.l_dS_mean = []
        self.l_dS_max = []


        # set drift action Kernel
        if self.drift_term_type   == 0: pass
        elif self.drift_term_type == 1: self.drift_term_kernel = sk_drift_kernel
        elif self.drift_term_type == 2: self.drift_term_kernel = euclidean_drift_kernel
        elif self.drift_term_type == 3: self.drift_term_kernel = minkowskian_drift_kernel
        else:
            raise NotImplementedError("Chosen action type not available!")

        # set starting kernel
        if self.start_type      == 0: pass
        elif self.start_type    == 1: self.start_kernel = start_zero_kernel
        elif self.start_type    == 2: self.start_kernel = start_ones_kernel
        elif self.start_type    == 3: self.start_kernel = start_random_kernel
        elif self.start_type    == 4: self.start_kernel = start_classical_free_kernel
        else:
            raise NotImplementedError("Chosen start type not available!")

        # set boundary kernel
        if self.boundary_type   == 0: pass
        elif self.boundary_type == 1: self.boundary_kernel = boundary_temporal_clamp_kernel
        elif self.boundary_type == 2: self.boundary_kernel = boundary_classical_kernel
        else:
            raise NotImplementedError("Chosen boundary type not available!")
        
        if self.a_0 is None:
            self.a_0 = np.ones(self.dims[0], dtype=scal.SCAL_FIELD_TYPE) / 10 # courant cond satisf. (gamma 10 default)
        self.gamma = 1 / self.a_0
        self.gamma_abs = abs(self.gamma)

        # set starting configuration


        # types
        self.init_seed   = init_seed
        self.start_type     = start_type
        self.boundary_type  = boundary_type
        self.drift_term_type    = drift_term_type
        self.noise_seed     = noise_seed
        self.converged      = converged
        self.sig_digs = int(-np.log10(dt))

        # device arrays
        self.d_phi0 = self.phi0
        self.d_phi1 = self.phi1
        self.d_eta  = self.eta
        self.d_a_0  = self.a_0
        self.d_gamma  = self.gamma
        self.d_gamma_abs  = self.gamma_abs
        self.d_dS   = self.dS
        self.d_dS_norm  = self.dS_norm
        self.d_dims     = self.dims
        self.d_adims    = self.adims
        self.d_l_xpos   = self.l_xpos
        self.d_buffer_field_big = self.buffer_field_big

        if self.start_type > 0: 
            my_parallel_loop(self.start_kernel, 
                             self.n_cells, 
                             self.d_phi0, 
                             self.d_dims,
                             self.d_adims, 
                             self.d_gamma,
                             self.mass_real, 
                            )

        if self.boundary_type > 0:
            my_parallel_loop(self.boundary_kernel, 
                             self.n_cells, 
                             self.d_phi0, 
                             self.d_dims, 
                             self.d_adims,
                            )
        
        if self.start_type > 0:             
            my_parallel_loop(self.start_kernel, 
                             self.n_cells, 
                             self.d_phi1, 
                             self.d_dims, 
                             self.d_adims, 
                             self.d_gamma,
                             self.mass_real, 
                            )
        if self.boundary_type > 0:
            my_parallel_loop(self.boundary_kernel, 
                             self.n_cells, 
                             self.d_phi1, 
                             self.d_dims, 
                             self.d_adims,
                            )
        
                
        # random number generation
        if use_cuda:
            n_blocks = math.ceil(self.n_cells / threadsperblock)
            self.rng = create_xoroshiro128p_states(
                threadsperblock * n_blocks, seed=self.noise_seed
            )
            self.copy_to_device()
        else:
            np.random.seed(self.noise_seed)
            self.rng = None


        
    def swap(self):
        """
        Swaps the references (pointers) of phi0 and phi1.

        :return:
        """
        self.d_phi0, self.d_phi1 = self.d_phi1, self.d_phi0

    def copy_to_device(self):
        """
        Copies all fields in CPU memory to GPU memory (arrays starting with `d_` prefix).

        :return:
        """
        if use_cuda:
            self.d_phi0 = cuda.to_device(self.phi0)
            self.d_phi1 = cuda.to_device(self.phi1)
            self.d_eta  = cuda.to_device(self.eta)
            self.d_a_0 = cuda.to_device(self.a_0)
            self.d_gamma = cuda.to_device(self.gamma)
            self.d_gamma_abs = cuda.to_device(self.gamma_abs)
            self.d_dS   = cuda.to_device(self.dS)
            self.d_dS_norm  = cuda.to_device(self.dS_norm)
            self.d_dims = cuda.to_device(self.dims)
            self.d_adims = cuda.to_device(self.adims)
            self.d_l_xpos   = cuda.to_device(self.l_xpos)
            self.d_buffer_field_big = cuda.to_device(self.buffer_field_big)

    def copy_to_host(self):
        """
        Copies all fields in GPU memory (arrays starting with `d_` prefix) to CPU memory.

        :return:
        """
        if use_cuda:
            self.d_phi0.copy_to_host(self.phi0)
            self.d_phi1.copy_to_host(self.phi1)
            self.d_eta.copy_to_host(self.eta)
            self.d_a_0.copy_to_host(self.a_0)
            self.d_gamma.copy_to_host(self.gamma)
            self.d_gamma_abs.copy_to_host(self.gamma_abs)
            self.d_dS.copy_to_host(self.dS)
            self.d_dS_norm.copy_to_host(self.dS_norm)
            self.d_dims.copy_to_host(self.dims)
            self.d_adims.copy_to_host(self.adims)
            self.d_l_xpos.copy_to_host(self.l_xpos)
            self.d_buffer_field_big.copy_to_host(self.buffer_field_big)

    def steps(self, n_steps=100, **kwargs):
        """
        Performs multiple CLE steps.

        kwargs:
            :param noise_factor:    a pre-factor for the noise term (for debugging)
            :param n_steps:         number of CLE steps to perform
            :param mean_dS_max:     mean of maximal drift terms (needed for adaptive step size)
            :evolve:                boolean which decides if fields are evolved (for debugging)
        :return:
        """
        for _ in range(int(n_steps)):
            self.step(**kwargs)


    def step(self, **kwargs):
        """
        Performs a single CLE step.

        kwargs:
            :param noise_factor:    a pre-factor for the noise term (for debugging)
            :param mean_dS_max:     mean of maximal drift terms (needed for adaptive step size)
            :evolve:                boolean which decides if fields are evolved (for debugging)
            :ds:                    boolean which decides if dynamical stabilization is used
        """
        # swap arrays
        if not self.converged:
            self.swap()
            noise_factor = kwargs.get("noise_factor", 1.0)
            # generate noise
            if use_cuda:
                my_parallel_loop(
                    generate_noise_cuda_kernel,
                    self.n_cells,
                    self.d_eta,
                    noise_factor,
                    self.d_dims,
                    self.rng,
                )
                cuda.synchronize()
            else:
                sqrt2 = math.sqrt(2.0)
                self.d_eta[:] = noise_factor * sqrt2 * np.random.normal(
                    size=self.d_eta.shape
                )

            # calc drift term
            if self.drift_term_type > 0:
                
                my_parallel_loop(
                    self.drift_term_kernel,
                    self.n_cells,
                    self.d_phi0,
                    self.d_gamma,
                    self.interaction, 
                    self.d_dims,
                    self.d_adims,
                    self.d_dS,
                    self.d_dS_norm,
                    self.mass_real,
                    self.mass_imag,
                    self.boundary_type,
                    )
                if use_cuda: cuda.synchronize()

            # calc maximum drift term
            mean_dS_max = kwargs.get("mean_dS_max", None)
            dS_max = amax(self.d_dS_norm)
            self.l_dS_max.append(dS_max)


            # adaptive stepsize
            ad_dt = self.dt
            if mean_dS_max is not None and dS_max > DS_MAX_LOWER:
                ad_dt = (
                    (ad_dt * mean_dS_max / dS_max)
                    if mean_dS_max < dS_max
                    else ad_dt
                )
            # print(f"ad_dt: {ad_dt}")
            # self.l_dt.append(real(math.round(ad_dt,self.sig_digs+1)))
            # self.l_dt.append(round(ad_dt, 12))
            self.l_dt.append(ad_dt)
            new_th = self.l_th[-1] + ad_dt# , self.sig_digs+1))
            self.l_th.append(new_th)

            # evolve
            evolve = kwargs.get("evolve", True)
            if evolve:
                my_parallel_loop(
                    evolve_kernel,
                    self.n_cells,
                    self.d_phi0,
                    self.d_phi1,
                    # self.d_gamma,
                    # self.d_gamma_abs,
                    self.d_dS,
                    self.d_eta,
                    self.dt,
                    self.gamma, 
                    self.dims, 
                    self.adims
                    # self.d_dims,
                    # self.d_adims,
                )
                
                if use_cuda:
                    cuda.synchronize()

            # set boundary condition    
            if self.boundary_type > 0:
                my_parallel_loop(self.boundary_kernel, 
                                self.n_cells, 
                                self.d_phi1, 
                                self.d_dims,
                                self.d_adims,
                )
                if use_cuda:
                    cuda.synchronize()

            self.count += 1



    def phi_n_moment_scalar(self, order = 1, use_cupy = True):
        """
        returns the n-th field moment (averaged over full lattice)
        """
        moment = self.phi_n_moment(order)
        return mean(moment)

    def phi_n_moment(self, order = 1, use_cupy = False):
        """
        returns the n-th field moment (averaged over spacial lattice)
        """

        nt = self.dims[0]

        out = np.zeros(self.n_cells, dtype=scal.SCAL_FIELD_TYPE)
        
        if use_cuda: d_out = cuda.to_device(out)
        else: d_out = out
        
        my_parallel_loop(phi_n_moment_kernel,
                         self.n_cells,
                         d_out,
                         order,
                         self.d_phi1,
        )
        
        if use_cuda: cuda.synchronize()
        d_out = reshape(d_out, (nt, self.n_cells // nt))
        d_out = arr_mean(d_out, axis = 1, to_cpu = not (use_cuda))

        # # loop over positions
        # for pos_x in self.d_l_xpos:
        #     # loop over real times
        #     my_parallel_loop(phi_n_moment_kernel,
        #                      nt,
        #                      pos_x,
        #                      d_out,
        #                      order,
        #                      self.d_phi1,
        #                      self.dims,
        #     )
        #     if use_cuda: cuda.synchronize()

        
        return (
            cp.asarray(d_out)
            if (use_cupy and use_cuda)
            else d_out
        )

    def uneqtime_field_corr_scal(self, use_cupy = False):
        num = self.dims[0] * self.n_cells
        out = np.zeros(num, dtype=scal.SCAL_FIELD_TYPE)

        if use_cuda: d_out = cuda.to_device(out)
        else: d_out = out

        my_parallel_loop(uneqtime_field_corr_kernel,
                         num,
                         self.d_phi1,
                         d_out,
                         self.d_dims,
                         self.d_adims
        )

        d_out = np.reshape(d_out, (self.dims[0]**2, self.adims[1]))
        if use_cuda: d_out = arr_mean(d_out, axis = 1, to_cpu = not (use_cuda))
        else: d_out = arr_mean(d_out, axis = 1)

        if use_cuda: cuda.synchronize()
    
        return (
            cp.asarray(d_out)
            if (use_cupy and use_cuda)
            else d_out
        )
    
    # def temporal_field_correlation(self, t0 = 0, use_cupy = False):
    #         """
    #         returns the n-th field moment (averaged over spacial lattice)
    #         """

    #         nt = self.dims[0]
    #         out = np.zeros(self.n_cells, dtype=scal.SCAL_FIELD_TYPE)
            
    #         if use_cuda: d_out = cuda.to_device(out)
    #         else: d_out = out
            
    #         my_parallel_loop(temporal_field_correlation_kernel,
    #                         self.n_cells,
    #                         d_out,
    #                         order,
    #                         self.d_phi1,
    #         )
            
    #         if use_cuda: cuda.synchronize()
    #         d_out = reshape(d_out, (nt, self.n_cells // nt))
    #         d_out = arr_mean(d_out, axis = 1, to_cpu = not (use_cuda))

    #         # # loop over positions
    #         # for pos_x in self.d_l_xpos:
    #         #     # loop over real times
    #         #     my_parallel_loop(phi_n_moment_kernel,
    #         #                      nt,
    #         #                      pos_x,
    #         #                      d_out,
    #         #                      order,
    #         #                      self.d_phi1,
    #         #                      self.dims,
    #         #     )
    #         #     if use_cuda: cuda.synchronize()

            
    #         return (
    #             cp.asarray(d_out)
    #             if (use_cupy and use_cuda)
    #             else d_out
    #         )

    # def temporal_field_correlation(self, t0 = 0,  use_cupy = False, **kwargs):
    #     nt = self.dims[0]
    #     out = np.zeros(nt, dtype=scal.SCAL_FIELD_TYPE)

    #     if use_cuda: d_out = cuda.to_device(out)
    #     else: d_out = out
    #     # loop over positions
    #     for d_pos_x in self.d_l_xpos:
    #         # loop over real times
    #         for idx in range(nt):
    #             temporal_field_correlation_kernel(                            
    #                         idx,
    #                          t0,
    #                          d_pos_x,
    #                          self.d_phi1,
    #                          self.d_dims,
    #                          d_out,)
    #         # my_parallel_loop(temporal_field_correlation_kernel,
    #         #                  nt,
    #         #                  t0,
    #         #                  d_pos_x,
    #         #                  self.d_phi1,
    #         #                  self.d_dims,
    #         #                  d_out,
    #         # )
    #         if use_cuda: cuda.synchronize()

    #     return (
    #         cp.asarray(d_out/self.adims[1])
    #         if (use_cupy and use_cuda)
    #         else cp.asnumpy(d_out/self.adims[1])
    #     )
    
    # def temporal_field_correlation_connected(self, t0 = 0, use_cupy = False):
    #     corr = self.temporal_field_correlation(t0 = t0)
    #     connected = corr - self.phi_n_moment(order = 1) * self.phi_n_moment(order = 1)[t0]

    #     return connected
    
    def field_real(self, use_cupy=True):
        """
        :return: real part of configuration
        """

        out = np.empty(self.n_cells, dtype=scal.SCAL_FIELD_TYPE)
        if use_cuda: d_out = cuda.to_device(out)
        else: d_out = out
        
        my_parallel_loop(field_real_kernel, 
                         self.n_cells, 
                         self.d_phi1, 
                         d_out)
        if use_cuda: cuda.synchronize()

        return (
            cp.asarray(d_out/self.adims[1])
            if (use_cupy and use_cuda)
            else cp.asnumpy(d_out/self.adims[1])
        )
    
    def field_imag(self, use_cupy=True):
        """
        :return: imag part of configuration
        """
        out = np.empty(self.n_cells, dtype=scal.SCAL_FIELD_TYPE)
        if use_cuda: d_out = cuda.to_device(out)
        else: d_out = out

        my_parallel_loop(field_imag_kernel, self.n_cells, self.d_phi1, d_out)
        if use_cuda: cuda.synchronize()
        return (
            cp.asarray(d_out/self.adims[1])
            if (use_cupy and use_cuda)
            else d_out/self.adims[1]
        )
    
    def im_msq(self):
        """
        :return: mean square of imaginary (unphysical) part of configuration, averaged over lattice
        """
        # imag_part = arr_mean(imag(self.d_phi1) * imag(self.d_phi1), to_cpu = not (use_cuda))
        imag_part = arr_mean(imag(self.d_phi1) * imag(self.d_phi1), to_cpu = True)
        if DEBUG: print(f"imag msq: {imag_part}")

        return imag_part
        
    def real_msq(self):
        """
        :return: mean square of imaginary (unphysical) part of configuration, averaged over lattice
        """
        real_part = arr_mean(real(self.d_phi1) * real(self.d_phi1), to_cpu = True)
        if DEBUG: print(f"real msq: {real_part}")
        return real_part
    
    def langevin_time(self, use_cupy = True, *args, **kwargs):
        out = self.l_th[-1]
        return (
            cp.asarray(out).get().item()
            if (use_cupy and use_cuda)
            else out
        )
    
    def ds_max_mean(self, *args, **kwargs):
        return mean(self.l_dS_max)
    
    def ds_max(self, *args, **kwargs):
        # print(np.max(self.d_dS_norm))
        return amax(self.d_dS_norm)
    
# @myjit
# def temporal_field_correlation_kernel(idx, t_0, out, field):
#     """
#     Kernel to calc n-th field moment
#     adds the field value to out
#     :param pos_t:     real time position
#     :param pos_x:     spacial position
#     :param out:       target array (manipulated)
#     :param order:     moment order
#     :param phi:       field array
#     :param dims:      lattice dims
#     """

#     add = 1
#     add *= field[idx]*field[idx_ref]
#     out[idx] += add

# @myjit
# def temporal_field_correlation_kernel(pos1_t, pos2_t, pos_x, field, dims, out):
#     """
#     Kernel to calculate temporal correlation 
#         :pos1_t:    variable real time position (correlation is calculated at t = pos1_t) 
#         :pos2_t:    fixed real time position
#         :pos_x:     spacial position
#         :phi:       field array
#         :dims:      lattice dimensions
#         :out:       target array (manipulated)
#     """

#     idx1 = l.get_index(pos1_t, pos_x, dims)
#     idx2 = l.get_index(pos2_t,pos_x, dims)
#     update = field[idx1] * field[idx2]
#     out[pos1_t] += update

# @myjit
# def temporal_field_correlation_connected_kernel(pos_t, out, pos_x, phi, dims):
#     ### TO BE CHANGED
#     idx = l.get_index(pos_t, pos_x, dims)
#     out[pos_t] += (1 * phi[idx])

@myjit
def phi_n_moment_kernel(idx, out, order, field):
    """
    Kernel to calc n-th field moment
    adds the field value to out
    :param pos_t:     real time position
    :param pos_x:     spacial position
    :param out:       target array (manipulated)
    :param order:     moment order
    :param phi:       field array
    :param dims:      lattice dims
    """

    add = 1
    for _ in range(order):
        add *= field[idx]
    out[idx] += (add)


# @myjit
# def phi_n_moment_kernel(pos_t, pos_x, out, order, field, dims):
#     """
#     Kernel to calc n-th field moment
#     adds the field value to out
#     :param pos_t:     real time position
#     :param pos_x:     spacial position
#     :param out:       target array (manipulated)
#     :param order:     moment order
#     :param phi:       field array
#     :param dims:      lattice dims
#     """

#     # idx = l.get_index(pos_t, pos_x, dims)

#     # # for _ in range(order):
#     # #     update *= field[idx]
#     # out[pos_t] += field[1]**order

#     tmp_pos_t = np.array([pos_t], dtype=np.int32)
#     pos = np.concatenate((tmp_pos_t, pos_x))
#     idx = l.get_index(pos, dims)
#     add = 1
#     for i in range(order):
#         add *= field[idx]
#     out[pos_t] += (add)

@myjit
def boundary_temporal_clamp_kernel(idx, field, dims, adims):
    t_slice = math.floor(idx / adims[1])
    if t_slice == 0: field[idx] = 1
    elif t_slice == dims[0]-1: field[idx] = 1

@myjit
def boundary_classical_kernel(idx, field, dims, adims):
    t_slice = idx // adims[1]
    if t_slice == 0 or t_slice ==1:
        field[idx] = 1
    
@myjit
def start_zero_kernel(idx, field, dims, adims, gamma, mass_real):
    """
    Set the value of field at index `idx` to zero. [kernel]

    :param idx:         lattice site index
    :param field:       scalar field array
    """
    field[idx] = 0
    # t_slice = idx // adims[1]
    # if t_slice == 0 or t_slice == 1:
    #     field[idx] = 1
    # else: field[idx] = 0

@myjit
def start_ones_kernel(idx, field, dims, adims, gamma, mass_real):
    """
    Set the value of field at index `idx` to one. [kernel]

    :param idx:         lattice site index
    :param field:         scalar field array
    """
    field[idx] = 1

@myjit
def start_random_kernel(idx, field, dims, adims, gamma, mass_real):
    field[idx] = np.random.uniform(0, 1)


@myjit
def start_classical_free_kernel(idx, field, dims, adims, gamma, mass_real):
    t_slice = idx // adims[1]
    if np.imag(gamma[t_slice]) != 0: t_slice = 0
    field[idx] = np.cos(np.abs(mass_real) * t_slice / gamma[t_slice])

@myjit
def phi4_drift_kernel(idx, field, gamma, interaction, dims, adims, dS_out, dS_norm, mass_real, mass_imag, boundary_type):
    """
    Computes and returns the action drift term (based on scalar phi4 action) at lattice site `idx`.

    :param idx:         lattice site index
    :param phi:         scalar field array
    :param gamma:       lattice anisotropy
    :param dims:        lattice dimensions
    :param adims:       cumulative product of lattice dimensions
    :param dS_out:      drift term arrayvpn.tuwien.ac.at
    :param mass_real:   bare mass_real
    :param mass_imag:   bare mass_imag
    """
    signature = 1
    t_slice = idx // adims[1]
    dS_out[idx] = -quabla(idx, field, gamma[t_slice], dims, adims, boundary_type)
    
    dS_out[idx] += signature * (mass_real+1j*mass_imag) * (mass_real+1j*mass_imag) * field[idx]
    dS_out[idx] += signature * (field[idx] * field[idx] * field[idx]) * interaction  / 6
    if signature == +1: dS_out[idx] *= 1j
    dS_norm[idx] = arr_abs(dS_out[idx])
    # if np.imag(dS_out[idx]) != 0: print("imag in drift")

@myjit
def euclidean_drift_kernel(idx, field, gamma, interaction, dims, adims, dS_out, dS_norm, mass_real, mass_imag, boundary_type):
    """
    Computes and returns the action drift term on the euclidean branch at lattice site `idx`.
    This has to be completely imaginary (by convention):
    The field stays real, update uses 1j*ds

    :param idx:         lattice site index
    :param phi:         scalar field array
    :param gamma:       lattice anisotropy
    :param dims:        lattice dimensions
    :param adims:       cumulative product of lattice dimensions
    :param dS_out:      drift term arrayvpn.tuwien.ac.at
    :param mass_real:   bare mass_real
    :param mass_imag:   bare mass_imag
    """
    n_dims = len(dims)
    out = 0
    t_slice = idx // adims[1]
    gamma_t = gamma[t_slice]
    a_o_avg = 1/gamma_t

    # temporal
    idx_plus  = l.shift(idx, 0, +1, dims, adims)
    idx_minus = l.shift(idx, 0, -1, dims, adims)
    phi_idx = field[idx]
    out -=  (field[idx_minus]+ field[idx_plus]-2*phi_idx) * abs(gamma_t)**2
    

    # spacial
    for i in range(1, n_dims):
        idx_plus  = l.shift(idx, i, +1, dims, adims)
        idx_minus = l.shift(idx, i, -1, dims, adims) 
        # g = my_round(gamma[t_slice_shift])
        out -= (field[idx_minus]+ field[idx_plus]-2*phi_idx)

    out += (mass_real + 1.0j*mass_imag) * (mass_real + 1.0j*mass_imag) * phi_idx
    out += interaction * phi_idx * phi_idx * phi_idx / 6 
           
    dS_out[idx] = (1j)*out
    dS_norm[idx] = abs(out)


# @myjit
# def quabla_euclidean(idx, phi, gamma, dims, adims):
#         """
#         euclidean lattice d Alembertian 

#         :param phi:         scalar field array
#         :param dims:        lattice dimensions
#         :param gamma:       local Anisotropy Factor
#         :param adims:       cumulative product of lattice dimensions
#         :return:            anisotropic quabla
#         """
#         n_dims = len(dims)
#         q_out = 0
#         # temporal part
#         idx_plus  = l.shift(idx, 0, +1, dims, adims)
#         idx_minus = l.shift(idx, 0, -1, dims, adims)
#         q_out += arr_abs(gamma)*arr_abs(gamma) * (phi[idx_minus] + phi[idx_plus] - 2 * phi[idx])
#         # spacial part
#         for i in range(1, n_dims):
#             idx_plus  = l.shift(idx, i, +1, dims, adims)
#             idx_minus = l.shift(idx, i, -1, dims, adims)
#             q_out += (phi[idx_minus] + phi[idx_plus] - 2*phi[idx])
#         return q_out

@myjit
def minkowskian_drift_kernel(idx, field, gamma, interaction, dims, adims, dS_out, dS_norm, mass_real, mass_imag, boundary_type):
    """
    Computes and returns the action drift term on the real time branch at lattice site `idx`.
    This has to be completely real (by convention):
    The field stays real, update uses 1j*ds

    :param idx:         lattice site index
    :param phi:         scalar field array
    :param gamma:       lattice anisotropy
    :param dims:        lattice dimensions
    :param adims:       cumulative product of lattice dimensions
    :param dS_out:      drift term arrayvpn.tuwien.ac.at
    :param mass_real:   bare mass_real
    :param mass_imag:   bare mass_imag
    """
    n_dims = len(dims)
    out = 0
    t_slice = idx // adims[1]
    gamma_t = gamma[t_slice]
    # assert (gamma_t.real > 0 and gamma_t.imag == 0), f"please use forward real time branch: gamma was {gamma_t}"

    # temporal
    idx_plus  = l.shift(idx, 0, +1, dims, adims)
    idx_minus = l.shift(idx, 0, -1, dims, adims)
    phi_idx = field[idx]
    out -=  (field[idx_minus]+ field[idx_plus]-2*phi_idx) * abs(gamma_t)**2
    
    # spacial
    for i in range(1, n_dims):
        idx_plus  = l.shift(idx, i, +1, dims, adims)
        idx_minus = l.shift(idx, i, -1, dims, adims) 
        out += (field[idx_minus]+ field[idx_plus]-2*phi_idx)

    out -= (mass_real + 1.0j*mass_imag) * (mass_real + 1.0j*mass_imag) * phi_idx 
    out -= interaction * phi_idx * phi_idx * phi_idx / 6
    dS_out[idx] = out
    dS_norm[idx] = abs(out)

@myjit
def quabla_minkowskian(idx, phi, gamma, dims, adims):
        """
        minkowski lattice d Alembertian

        :param phi:         scalar field array
        :param dims:        lattice dimensions
        :param gamma:       local Anisotropy Factor
        :param adims:       cumulative product of lattice dimensions
        :return:            anisotropic quabla
        """
        n_dims = len(dims)
        q_out = 0
        # temporal part
        idx_plus  = l.shift(idx, 0, +1, dims, adims)
        idx_minus = l.shift(idx, 0, -1, dims, adims)
        q_out += arr_abs(gamma)*arr_abs(gamma) * (phi[idx_minus] + phi[idx_plus] - 2 * phi[idx])
        # spacial part
        for i in range(1, n_dims):
            idx_plus  = l.shift(idx, i, +1, dims, adims)
            idx_minus = l.shift(idx, i, -1, dims, adims)
            q_out -= (phi[idx_minus] + phi[idx_plus] - 2*phi[idx])
        return q_out

@myjit
def sk_drift_kernel(idx, phi, gamma, interaction, dims, adims, dS_out, dS_norm, mass_real, mass_imag, boundary_type):
    n_dims = len(dims)
    out = 0
    t_slice = idx // adims[1]
    t_slice_shift = (t_slice - 1) % (dims[0])
    gamma_t = gamma[t_slice]
    gamma_t_shift = gamma[t_slice_shift]
    a_o_avg = (1/gamma_t + 1/gamma_t_shift) / 2

    # temporal
    idx_plus  = l.shift(idx, 0, +1, dims, adims)
    idx_minus = l.shift(idx, 0, -1, dims, adims)
    phi_idx = phi[idx]
    out +=  (phi_idx - phi[idx_minus]) * gamma_t_shift
    out +=  (phi_idx - phi[idx_plus]) * gamma_t
    

    # spacial
    for i in range(1, n_dims):
        idx_plus  = l.shift(idx, i, +1, dims, adims)
        idx_minus = l.shift(idx, i, -1, dims, adims) 
        # g = my_round(gamma[t_slice_shift])
        out -= (phi_idx - phi[idx_minus]) / gamma_t_shift #!!
        out -= (phi_idx - phi[idx_plus]) / gamma_t_shift #!!

    out -= (mass_real + 1.0j*mass_imag) * (mass_real + 1.0j*mass_imag) * phi_idx * \
           a_o_avg
    out -= interaction * phi_idx * phi_idx * phi_idx / 6  * \
           a_o_avg
    dS_out[idx] = out* 1/(1/2*(abs(1/gamma_t)+1/abs(gamma_t_shift)))
    dS_norm[idx] = abs(out)



@myjit
def quabla(idx, phi, gamma, dims, adims, boundary_type):
        """
        Lattice d Alembertian

        :param phi:         scalar field array
        :param dims:        lattice dimensions
        :param gamma:       local Anisotropy Factor
        :param adims:       cumulative product of lattice dimensions
        :return:            anisotropic quabla
        """
        # if np.imag(phi[idx]) != 0: print("given imag to quabla"); print(phi[idx])
        n_dims = len(dims)
        t_slice = idx // adims[1]
        q_out = 0
        # temporal part
        c = 2
        if boundary_type == 2: 
            if idx // adims[1] == dims[0] - 1: c = 1

        idx_plus  = l.shift(idx, 0, +1, dims, adims)
        idx_minus = l.shift(idx, 0, -1, dims, adims)
        q_out += arr_abs(gamma)*arr_abs(gamma) * (phi[idx_minus] + phi[idx_plus] - c * phi[idx])
        # if np.imag(phi[idx_minus]) != 0: print("given imag neighbor to quabla"); print(phi[idx_minus])
        # if np.imag(phi[idx_plus]) != 0: print("given imag neighbor to quabla"); print(phi[idx_plus])

        # if np.imag(q_out) != 0: print("imag in quabla")
        # spacial part
        for i in range(1, n_dims):
            idx_plus  = l.shift(idx, i, +1, dims, adims)
            idx_minus = l.shift(idx, i, -1, dims, adims)
            q_out -= (phi[idx_minus] + phi[idx_plus] - 2*phi[idx])
        return q_out
        

@myjit
def generate_noise_cuda_kernel(idx, eta, factor, dims, rng):
    """
    Generates the noise term (`eta` array) at lattice size `idx`. [kernel]

    :param idx:         lattice site index
    :param eta:         noise term array
    :param factor:      pre-factor (for debugging)
    :param dims:        lattice dimensions
    :param rng:         CUDA random number generator
    :return:
    """
    
    factor2 = SQRT2 * factor
    r = xoroshiro128p_normal_float32(rng, idx)
    eta[idx] = factor2 * r

@myjit
# def evolve_kernel(idx, field0, field1, gamma, gamma_abs, dS, eta, dt, dims, adims):
def evolve_kernel(idx, field0, field1, dS, eta, dt, gamma, dims, adims):

    """
    Performs a single step at lattice site `idx`. [kernel]

    :param idx:         lattice site index
    :param phi0:        scalar field array (old, read only)
    :param phi1:        scalar field (new, written)
    :param gamma:       local anisotropy
    :param dS:          drift term array
    :param eta:         noise term array
    :param dt:          langevin time step
    """
    dt_sqrt = math.sqrt(dt)
    t_slice = idx // adims[1]
    t_slice_shift = (t_slice - 1) % (dims[0])
    gamma_t = gamma[t_slice]
    gamma_t_shift = gamma[t_slice_shift]

    # noise term
    etaterm = eta[idx] * dt_sqrt
    # if np.imag(dS[idx]) != 0: print("imag in evolve")
    update = etaterm + dt * 1.j*dS[idx]*(1/2*(abs(1/gamma_t)+1/abs(gamma_t_shift)))
    field1[idx] = field0[idx] + update


@myjit 
def hamiltonian_density_kernel(idx, phi, gamma, dims, adims, hamilton_dens, mass_real, mass_imag):
    """
    Calculate and set hamilton density at `idx`. [kernel]

    :param idx:             lattice site index
    :param phi:             scalar field array
    :param gamma:           local anisotropy
    :param dims:            lattice dimensions
    :param adims:           cumulative product of lattice dimensions
    :param hamilton_dens:   hamilton density (written)
    :param mass_real:       bare mass_real
    :param mass_imag:       bare mass_imag
    :return:                drift
    """
    n_dims = len(dims)
    
    # temporal part
    idx_plus  = l.shift(idx, 0, +1, dims, adims)
    hamilton_dens[idx] = (gamma*(phi[idx_plus] - phi[idx])) * (gamma*(phi[idx_plus] - phi[idx])) / 2
    
    # spacial part
    for i in range(1, n_dims):
        idx_plus  = l.shift(idx, i, +1, dims, adims)
        hamilton_dens[idx] += (phi[idx_plus] - phi[idx]) * (phi[idx_plus] - phi[idx]) / 2

    # mass term
    hamilton_dens[idx] += mass_real*mass_real * phi[idx] * phi[idx] / 2


@myjit
def field_real_kernel(idx, field, out):
    out[idx] = np.real(field[idx])

@myjit
def field_imag_kernel(idx, field, out):
    out[idx] = imag(field[idx])

@myjit
def add_kernel(idx, field, out):
    out += field[idx]


@myjit
def uneqtime_field_corr_kernel(idx_corr, field, out, dims, adims):    
    idx_t = idx_corr % adims[0]
    idx_tprime = idx_corr // adims[0] * adims[1] + idx_corr % adims[1]

    f1 = field[idx_t]
    f2 = field[idx_tprime]
    out[idx_corr] = f1 * f2