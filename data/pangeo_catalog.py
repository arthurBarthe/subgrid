#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:20 2020

@author: arthur
"""

import intake
import xarray as xr
import numpy as np

from intake.config import conf
conf['persist_path'] = '/scratch/ag7531/'
CACHE_FOLDER = '/scratch/ag7531/pangeo_cache'


def get_patch(catalog_url, ntimes: int = None, bounds: list = None,
              cO2_level=0, *selected_vars):
    """
    Return a tuple with a patch of uv velocities along with the grid details.

    Parameters
    ----------
    catalog_url : str
        url where the catalog lives.
    ntimes : int, optional
        Number of days to use. The default is None which corresponds to all.
    bounds : list, optional
        Bounds of the path, (lat_min, lat_max, long_min, long_max). Note that
        the order matters!
    cO2_level : int, optional
        CO2 level, 0 (control) or 1 (1 percent increase C02 per year).
        The default is 0.
    *selected_vars : str
        Variables selected from the surface velocities dataset.


    Returns
    -------
    uv_data : xarray dataset
        xarray dataset containing the requested u,v velocities.
    grid_data : xarray dataset
        xarray dataset with the grid details.

    """
    catalog = intake.open_catalog(catalog_url)
    if cO2_level == 0:
        source = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
        cache_folder = CACHE_FOLDER
    elif cO2_level == 1:
        source = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
        cache_folder = CACHE_FOLDER + '1percent'
    else:
        raise ValueError('Unrecognized cO2 level. Should be O or 1.')
    s_grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_grid
    # The following lines are for caching
    source.urlpath = 'filecache::' + source.urlpath
    s_grid.urlpath = 'filecache::' + source.urlpath
    source.storage_options['filecache'] = dict(
        cache_storage=cache_folder)
    s_grid.storage_options['filecache'] = dict(
        cache_storage=cache_folder)
    # Convert to dask
    uv_data = uv_data.to_dask()
    grid_data = grid_data.to_dask()
    # Following line is necessary to transform non-primary coords into vars
    grid_data = grid_data.reset_coords()
    if bounds is not None:
        uv_data = uv_data.sel(xu_ocean=slice(*bounds[2:]),
                              yu_ocean=slice(*bounds[:2]))
        grid_data = grid_data.sel(xu_ocean=slice(*bounds[2:]),
                                  yu_ocean=slice(*bounds[:2]))
    if ntimes is not None:
        uv_data = uv_data.isel(time=slice(0, ntimes))

    if len(selected_vars) == 0:
        return uv_data, grid_data
    else:
        return uv_data[list(selected_vars)], grid_data


def get_whole_data(url, c02_level):
    data, grid = get_patch(url, None, None, c02_level, 'usurf', 'vsurf')
    return data, grid


def get_cm2_5_grid():
    grid = xr.open_dataset('/home/arthur/ocean.static.nc')
    dy_u = np.diff(grid['yu_ocean']) / 360 * 2 * np.pi * 6400 * 1e3
    # dx_u = np.diff(grid['xu_ocean']) * np.cos(grid['yu_ocean'] / 360 * 2 * np.pi)
    dx_u = None
    return grid, dx_u, dy_u


if __name__ == '__main__':
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/arthur/\
access_key.json"
    CATALOG_URL = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
        /master/intake-catalogs/master.yaml'
    data = get_whole_data(CATALOG_URL, 0)
