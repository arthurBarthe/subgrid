#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020

@author: arthur
TODO 
Add some verification that the call to eddy forcing is not for a dataset
that is too large as a region.
"""

import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np

def advections(u_v_field, grid_data):
    """Computes advection terms"""
    gradient_x = u_v_field.diff(dim='xu_ocean') / grid_data['dxu']
    gradient_y = u_v_field.diff(dim='yu_ocean') / grid_data['dyu']
    u, v = u_v_field['usurf'], u_v_field['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = u * gradient_x['vsurf'] + v * gradient_y['vsurf']
    return xr.Dataset({'adv_x': adv_x, 'adv_y' : adv_y})


# def spatial_filter(data, scale):
#     print('scale', scale)
#     result = None
#     for time in range(data.shape[0]):
#         print(time)
#         gf = dask.delayed(gaussian_filter)(data[time, ...], scale)
#         if result is None:
#             result = dask.array.from_delayed(gf, shape=data.shape[1:],
#                                              dtype=float)
#         else:
#             gf = dask.array.from_delayed(gf, shape = data.shape[1:], 
#                                          dtype=float)
#             result = dask.array.concatenate((result, gf))
#     return result

def spatial_filter(data, sigma):
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        result[t, ...] = gaussian_filter(data[t, ...], sigma,
                                         mode='constant')
    return result

def spatial_filter_dataset(dataset, sigma: float):
    """Applies spatial filtering to the dataset across the spatial dimensions
    """
    return xr.apply_ufunc(lambda x: spatial_filter(x, sigma), dataset, 
                                  dask='parallelized', 
                                  output_dtypes=[float,])

#Old version
# def compute_grid_steps(u_v_dataset):
#     """Computes the grid steps for the (x,y) grid"""
#     grid_step = [0, 0]
#     steps_x = u_v_dataset.coords['x'].diff('x')
#     steps_y = u_v_dataset.coords['y'].diff('y')
#     grid_step[0] = abs(steps_x.mean().item())
#     grid_step[1] = abs(steps_y.mean().item())
#     return tuple(grid_step)


def compute_grid_steps(grid_info: xr.Dataset):
    step_x = grid_info['dxu'].mean().item()
    step_y = grid_info['dyu'].mean().item()
    return step_x, step_y


def eddy_forcing(u_v_dataset, grid_data, scale: float, method='mean'):
    """Computes the eddy forcing terms on high resolution"""
    # TODO check if we can do something smarter here
    # Replace nan values with zeros
    u_v_dataset = u_v_dataset.fillna(0.0)
    # High res advection terms
    adv = advections(u_v_dataset, grid_data)
    # Grid steps
    grid_steps = compute_grid_steps(grid_data)
    # Filtered u,v field
    u_v_filtered = spatial_filter_dataset(u_v_dataset, 
                                          (scale / grid_steps[0],
                                          scale / grid_steps[1]))
    # Advection term from filtered velocity field
    adv_filtered = advections(u_v_filtered, grid_data)
    # Forcing
    forcing = adv_filtered - adv
    forcing = forcing.rename({'adv_x' : 'S_x', 'adv_y' : 'S_y'})
    # Merge filtered u,v and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # Reweight using the area of the cell
    forcing = forcing * grid_data['area_u'] / 1e8
    print(forcing)
    # Coarsen
    forcing = forcing.coarsen({'xu_ocean' : int(scale / grid_steps[0]),
                               'yu_ocean' : int(scale / grid_steps[1])},
                                boundary='trim')
    if method == 'mean':
        forcing = forcing.mean()
    else:
        raise('Passed method does not correspond to anything.')
    return forcing
