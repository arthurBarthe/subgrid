#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:15:35 2020

@author: arthur
"""

import xarray as xr
import dask
import dask.array as da
import dask.bag as db
from scipy.ndimage import gaussian_filter
import numpy as np

def advections(u_v_dataset):
    """Computes advection terms"""
    gradient_x = u_v_dataset.differentiate('x')
    gradient_y = u_v_dataset.differentiate('y')
    u, v = u_v_dataset['usurf'], u_v_dataset['vsurf']
    adv_x = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_y = v * gradient_x['vsurf'] + v * gradient_y['vsurf']
    return xr.Dataset({'adv_x': adv_x, 'adv_y' :adv_y})


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

def spatial_filter(data, scale):
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        result[t, ...] = gaussian_filter(data[t, ...], scale)
    return result


def eddy_forcing(u_v_dataset, scale: float, method='mean'):
    step_x = abs(u_v_dataset.coords['x'].values.mean())
    step_y = abs(u_v_dataset.coords['y'].values.mean())
    adv = advections(u_v_dataset)
    def ufunc(x):
        return gaussian_filter(x, (0, scale / step_x, scale / step_y))
    u_v_filtered = xr.apply_ufunc(lambda x: spatial_filter(x, 2), u_v_dataset, dask='parallelized', 
                                  output_dtypes=[float,])
    adv_filtered = advections(u_v_filtered)
    forcing = adv_filtered - adv
    forcing = forcing.rename({'adv_x' : 'S_x', 'adv_y' : 'S_y'})
    forcing = forcing.coarsen({'x': int(scale / step_x), 
                               'y': int(scale / step_y)}, boundary='trim')
    if method == 'mean':
        return forcing.mean()
    else:
        raise('Passed method does not correspond to anything.')


if __name__ == '__main__':
    test = da.random.randint(0, 10, (200, 20, 20), chunks = (1, 20, 20))
    filtered = spatial_filter(test, 2)