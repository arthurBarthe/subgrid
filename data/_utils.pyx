# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:17:11 2019

@author: Arthur
"""
import h5py
from os.path import join
import numpy as np
cimport numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, gaussian
from grids import RectangularData

dtype = np.double
ctypedef double dtype_t


def load_tom_data(filepath: str, filename: str):
    print('Loading data... This might take a minute or two...')
    with h5py.File(join(filepath, filename), 'r') as file:
        data = file['psi1a'].value
        print('Successfuly loaded the data!')
        print('Shape of data: %s' % str(data.shape))
        print('Size of data: {:.2f} Gb'.format(data.nbytes / 1e9))
        return data

#Taken from Tom Bolton
def calcEddyForcing( psiEddy, psiBar, scale_filter ) :
    """
    Copied from Tom Bolton's code.'
    Given the filtered-streamfunction 'psiBar' and the sub-filter
    streamfunction psiEddy = psi - psiBar, calculate the components
    of the sub-filter eddy momentum forcing Sx and Sy

    (Sx,Sy) = (U.grad)U - filter( (u.grad)u ),

    where U is the velocity from the filtered-streamfunction, and
    u is the full velocity (filtered + sub-filter).    The calculation
    requires more spatial-filtering, which is why the length-scale
    of the filter 'l' (in m) is required as an input variable.

    """

    # spatial-resolution of QG model (7.5km)
    dx = 7.5e3

    # streamfunction for calculating u and v
    psi = psiBar + psiEddy

    # calculate gradients
    [ psi_t, psi_y, psi_x] = np.gradient( psi, dx )
    [ psiBar_t, psiBar_y, psiBar_x] = np.gradient( psiBar, dx )

    u, v = -psi_y, psi_x
    U, V = -psiBar_y, psiBar_x
    
    # Calcluate filtered-advection term
    [ U_t, U_y, U_x ] = np.gradient( U, dx );  del U_t
    [ V_t, V_y, V_x ] = np.gradient( V, dx );  del V_t

    # ( Ud/dx + Vd/dy )U and ( Ud/dx + Vd/dy )V
    # arthur ok
    adv1_x = U * U_x + V * U_y
    adv1_y = U * V_x + V * V_y

    del U_x, U_y, V_x, V_y

    # Calculate sub-filter advection term
    [u_t, u_y, u_x] = np.gradient(u, dx); del u_t
    [v_t, v_y, v_x] = np.gradient(v, dx); del v_t

    # ( ud/dx + vd/dy )u + ( ud/dx + vd/dy )v
    adv2_x = u * u_x + v * u_y
    adv2_y = u * v_x + v * v_y

    del u_x, u_y, v_x, v_y

    for t in range( adv2_x.shape[0] ) :
       adv2_x[t,:,:] = gaussian_filter( adv2_x[t,:,:], scale_filter/dx )
       adv2_y[t,:,:] = gaussian_filter( adv2_y[t,:,:], scale_filter/dx )

    # Calculate the eddy momentum forcing components
    Sx = adv1_x - adv2_x
    Sy = adv1_y - adv2_y
    return Sx, Sy


cdef filter_dataset(double[:,:,:] data, dx: float, dy: float,
          scale_x: float, scale_y: float = None):
    cdef int nb_samples = data.shape[0]
    filtered_data = np.zeros_like(data)
    cdef int i_sample
    sigma = (scale_x / dx, scale_y / dy)
    #We just loop through the samples
    data_p = data
    for i_sample in range(nb_samples):
        filtered_data[i_sample, ...] = gaussian_filter(data_p[i_sample, ...],
                     sigma)
    return filtered_data

cpdef build_training_data(double[:,:,:] psi, double dx, double dy, 
                          double scale):
    psi_ = filter_dataset(psi, dx, dy, scale, scale)
    psi_eddy = psi - psi_
    sx, sy = calcEddyForcing(psi_eddy, psi_, scale)

    # Apply coarse-graining
    psi_ = RectangularData(data=psi_)
    sx = RectangularData(data=sx)
    sy = RectangularData(data=sy)
    psi_ = psi_.coarse_grain(factor=4, dims=(1,2))
    sx = sx.coarse_grain(factor=4, dims=(1,2))
    sy = sy.coarse_grain(factor=4, dims=(1,2))
    return (psi_.data, sx.data, sy.data)

def block_loop(n_times, n_times_per_loop, data, function, shape_result):
    """Block execution of  loop"""
    cdef int n_iterations = n_times // n_times_per_loop
    cdef int i
    cdef int time_start
    cdef int time_end
    cdef dtype_t[:,:,:] res0
    cdef dtype_t[:,:,:] res1
    cdef dtype_t[:,:,:] res2
    result = np.zeros(shape_result, dtype=dtype)
    cdef dtype_t[:, :, :, :] result_view = result
    cdef dtype_t[:, :, :] data_view = data
    for i in range(n_iterations):
        print('Iteration number {}/{}'.format(i, n_iterations))
        time_start = n_times_per_loop * i
        time_end = n_times_per_loop * (i+1)
        if i == n_iterations - 1:
            time_end = n_times
        f_result = function(data_view[time_start : time_end, :, :])
        res0 = f_result[0]
        res1 = f_result[1]
        res2 = f_result[2]
        
        result_view[time_start : time_end, 0, :, :] = res0
        result_view[time_start : time_end, 1, :, :] = res1
        result_view[time_start : time_end, 2, :, :] = res2
    return result

