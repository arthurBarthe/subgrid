#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020

@author: arthur
"""

import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np


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
    # gradient_x = u_v_field.differentiate(coord='xu_ocean')
    # gradient_y = u_v_field.differentiate(coord='yu_ocean')
    u, v = u_v_field['usurf'], u_v_field['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = u * gradient_x['vsurf'] + v * gradient_y['vsurf']
    return xr.Dataset({'adv_x': adv_x, 'adv_y': adv_y})


def spatial_filter(data, sigma):
    """
    Apply a gaussian filter along all dimensions except first one.

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
        Scale of the filtering, same unit as those of the grid (often, meters)

    Returns
    -------
    filt_dataset : xarray dataset
        Filtered dataset.

    """
    dataset = dataset * grid_info['area_u'] / 1e8
    areas = grid_info['area_u'] / 1e8
    norm = xr.apply_ufunc(lambda x: gaussian_filter(x, sigma, mode='constant'),
                          areas, dask='parallelized', output_dtypes=[float, ])
    filtered = xr.apply_ufunc(lambda x: spatial_filter(x, sigma), dataset,
                              dask='parallelized', output_dtypes=[float, ])
    filtered = filtered.where(abs(filtered) > 0)
    return filtered / norm.where(norm > 0)


def compute_grid_steps(grid_info: xr.Dataset):
    """
    Return the average grid step along each axis.

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
                 debug_mode=False):
    """
    Compute the sub-grid forcing terms.

    Parameters
    ----------
    u_v_dataset : xarray dataset
        High-resolution velocity field.
    grid_data : xarray dataset
        High-resolution grid details.
    scale : float
        Scale, in meters, or factor, if scale_mode is set to 'factor'
    method : str, optional
        Coarse-graining method. The default is 'mean'.
    area: bool, optional
        DEPRECIATED do not use
        True if we multiply by the cell area
    scale_mode: str, optional
        'factor' if we set the factor, 'scale' if we set the scale
    Returns
    -------
    forcing : xarray dataset
        Dataset containing the low-resolution velocity field and forcing.

    """
    # Replace nan values with zeros. 
    u_v_dataset = u_v_dataset.fillna(0.0)
    # Grid steps
    grid_steps = compute_grid_steps(grid_data)
    print('Average grid steps: ', grid_steps)
    if scale_mode == 'factor':
        print('Using factor mode')
        scale_x = scale
        scale_y = scale
    scale_filter = (scale_x / 2, scale_y / 2)
    # High res advection terms
    adv = advections(u_v_dataset, grid_data)
    filtered_adv = spatial_filter_dataset(adv, grid_data, scale_filter)
    # Filtered u,v field
    u_v_filtered = spatial_filter_dataset(u_v_dataset, grid_data, scale_filter)
    # Advection term from filtered velocity field
    adv_filtered = advections(u_v_filtered, grid_data)
    # Forcing
    forcing = adv_filtered - filtered_adv
    forcing = forcing.rename({'adv_x': 'S_x', 'adv_y': 'S_y'})
    # Merge filtered u,v and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # Reweight using the area of the cell
    if area:
        forcing = forcing * grid_data['area_u'] / 1e8
    print(forcing)
    # Coarsen
    print('scale: ', (scale_x, scale_y))
    print('scale factor: ', scale)
    print('step: ', grid_steps)
    forcing_coarse = forcing.coarsen({'xu_ocean': int(scale_x),
                                      'yu_ocean': int(scale_y)},
                                     boundary='trim')
    if method == 'mean':
        forcing_coarse = forcing_coarse.mean()
    else:
        raise ValueError('Passed coarse-graining method not implemented.')
    if not debug_mode:
        return forcing_coarse
    else:
        u_v_dataset = u_v_dataset.merge(adv)
        filtered_adv = filtered_adv.rename({'adv_x': 'f_adv_x',
                                            'adv_y': 'f_adv_y'})
        adv_filtered = adv_filtered.rename({'adv_x': 'adv_f_x',
                                            'adv_y': 'adv_f_y'})
        u_v_filtered = u_v_filtered.rename({'usurf': 'f_usurf',
                                            'vsurf': 'f_vsurf'})
        u_v_dataset = xr.merge((u_v_dataset, u_v_filtered, adv, filtered_adv,
                                adv_filtered, forcing[['S_x', 'S_y']]))
        return u_v_dataset, forcing_coarse
