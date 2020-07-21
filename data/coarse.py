#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020

@author: arthur
"""

import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt


def inv_cdf(y, sigma):
    def k(x):
        return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    xs = np.arange(1, int(10 * sigma) + 1)
    s = np.cumsum(k(xs))
    s = 2 * s + k(0)
    s /= s[-1]
    print(s)
    plt.plot(xs / sigma, s)
    return (np.argwhere(s >= y))[0][0]

def cdf(y, sigma):
    def k(x):
        return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    xs = np.arange(1, int(10 * sigma) + 1)
    s = np.cumsum(k(xs))
    s = 2 * s + k(0)
    s /= s[-1]
    print(s)
    plt.plot(xs / sigma, s)
    plt.show()
    return (s[int(y)])


def advections(u_v_field, grid_data):
    """
    Return the advection terms corresponding to the passed velocity field.

    Parameters
    ----------
    u_v_field : xarray dataset
        Velocity field, must contains variables usurf and vsurf.
    grid_data : xarray dataset
        Dataset with grid details, must contain variables dxu and dyu.

    Returns
    -------
    advections : xarray dataset
        Advection components, under variable names adv_x and adv_y.

    """
    gradient_x = u_v_field.diff(dim='xu_ocean') / grid_data['dxu']
    gradient_y = u_v_field.diff(dim='yu_ocean') / grid_data['dyu']
    u, v = u_v_field['usurf'], u_v_field['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = u * gradient_x['vsurf'] + v * gradient_y['vsurf']
    result = xr.Dataset({'adv_x': adv_x, 'adv_y': adv_y})
    return result


def spatial_filter(data, sigma):
    """
    Apply a gaussian filter along all dimensions except first one, which
    corresponds to time.

    Parameters
    ----------
    data : numpy array
        Data to filter.
    sigma : float
        Unitless scale of the filter.

    Returns
    -------
    result : numpy array
        Filtered data.

    """
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        data_t = data[t, ...]
        result_t = gaussian_filter(data_t, sigma, mode='constant')
        result[t, ...] = result_t
    return result


def spatial_filter_dataset(dataset, grid_info, sigma: float):
    """
    Apply spatial filtering to the dataset across the spatial dimensions.

    Parameters
    ----------
    dataset : xarray dataset
        Dataset to which filtering is applied. Time must be the first
        dimension, whereas spatial dimensions must come after.
    grid_info : xarray dataset
        Dataset containing details on the grid, in particular must have
        variables dxu and dyu.
    sigma : float
        Unitless scale of the filter

    Returns
    -------
    filt_dataset : xarray dataset
        Filtered dataset.

    """
    # Normalize for computational stability. We multiply back after filtering.
    stds = dataset.std()
    dataset = dataset / stds
    # Apply weights
    dataset = dataset * grid_info['area_u'] / 1e8
    areas = grid_info['area_u'] / 1e8
    # Compute normalization term by applying filter to cell areas only
    norm = xr.apply_ufunc(lambda x: gaussian_filter(x, sigma, mode='constant'),
                          areas, dask='parallelized', output_dtypes=[float, ])
    ufunc = lambda x: spatial_filter(x, sigma)
    filtered_data = xr.apply_ufunc(ufunc, dataset, dask='parallelized',
                                   output_dtypes=[float, ])
    # Apply normalization
    filtered_data /= norm
    filtered_data *= stds
    return filtered_data


def compute_grid_steps(grid_info: xr.Dataset):
    """
    Return the average grid step along each axis. Not used in factor mode.

    Parameters
    ----------
    grid_info : xr.Dataset
        Dataset containing the grid details. Must have variables dxu and dyu

    Returns
    -------
    step_x : float
        Mean step of the grid along the x axis (longitude)
    step_y : float
        Mean step of the grid along the y axis (latitude)

    """
    step_x = grid_info['dxu'].mean().compute().item()
    step_y = grid_info['dyu'].mean().compute().item()
    return step_x, step_y


def eddy_forcing(u_v_dataset, grid_data, scale: float, method: str = 'mean',
                 area: bool = False, scale_mode: str = 'factor',
                 debug_mode=False, gaussian_filter_pp=0.8):
    """
    Compute the sub-grid forcing terms.

    Parameters
    ----------
    u_v_dataset : xarray dataset
        High-resolution velocity field.
    grid_data : xarray dataset
        High-resolution grid info, must contain dxu, dyu, and area_u.
    scale : float
        Scale, in meters, or factor, unitless, if scale_mode is set to 'factor'
    method : str, optional
        Coarse-graining method. The default is 'mean'.
    area: bool, optional
        DEPRECIATED do not use
    scale_mode: str, optional
        'factor' if we set the unitless factor, 'scale' if we set the scale
        in meters.
        Recommanded method is factor.
    debug_mode: bool, optional
        If True, returns all the intermediary quantities
    gaussian_filter_pp: float, optional
        Percentage of information within the "box" [x-scale, x+scale] when
        applying the gaussian filter at location x.
    Returns
    -------
    forcing : xarray dataset
        Dataset containing the low-resolution velocity field and forcing.

    """
    # Replace nan values with zeros. 
    u_v_dataset = u_v_dataset.fillna(0.0)
    if scale_mode == 'factor':
        print('Using factor mode')
        scale_x = scale
        scale_y = scale
    else:
        grid_steps = compute_grid_steps(grid_data)
        print('Average grid steps: ', grid_steps)
        # !!!Should we take integer part here since we do in the 
        # coarse-graining?
        scale_x = scale / grid_steps[0]
        scale_y = scale / grid_steps[1]
    # filter's scale is half the coarse-graining
    scale_f_x = scale_x / 2 
    scale_f_y = scale_y / 2 
    # High res advection terms + filtering
    adv = advections(u_v_dataset, grid_data)
    filtered_adv = spatial_filter_dataset(adv, grid_data, (scale_f_x,
                                                           scale_f_y))
    if not debug_mode:
        # to avoid oom
        del adv
    # Filter u,v field + advection
    u_v_filtered = spatial_filter_dataset(u_v_dataset, grid_data,
                                          (scale_f_x, scale_f_y))
    adv_of_filtered = advections(u_v_filtered, grid_data)
    # Forcing
    forcing = adv_of_filtered - filtered_adv
    forcing = forcing.rename({'adv_x': 'S_x', 'adv_y': 'S_y'})
    # Merge filtered u,v and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # Coarsening
    print('scale: ', (scale_x, scale_y))
    print('scale factor: ', scale)
    if not debug_mode:
        forcing = forcing.coarsen({'xu_ocean': int(scale_x),
                                   'yu_ocean': int(scale_y)}, boundary='trim')
        if method == 'mean':
            forcing = forcing.mean()
        else:
            raise ValueError('Passed coarse-graining method not implemented.')
        return forcing
    # if debug mode
    forcing_coarse = forcing.coarsen({'xu_ocean': int(scale_x),
                                      'yu_ocean': int(scale_y)},
                                     boundary='trim')
    if method == 'mean':
        forcing_coarse = forcing_coarse.mean()
    else:
        raise ValueError('Passed coarse-graining method not implemented.')
    filtered_adv = filtered_adv.rename({'adv_x': 'f_adv_x',
                                        'adv_y': 'f_adv_y'})
    adv_of_filtered = adv_of_filtered.rename({'adv_x': 'adv_x_of_f',
                                              'adv_y': 'adv_y_of_f'})
    u_v_dataset = u_v_dataset.merge(filtered_adv)
    u_v_dataset = u_v_dataset.merge(adv_of_filtered)
    u_v_dataset = u_v_dataset.merge(adv)
    u_v_dataset = u_v_dataset.merge(forcing[['S_x', 'S_y']])
    return forcing_coarse, u_v_dataset
