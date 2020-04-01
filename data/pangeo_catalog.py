#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:20 2020

@author: arthur
"""

import matplotlib
import os
# if os.environ['LOGNAME'] is not 'arthur':
#     # If ran remotely
#     matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import intake

from intake.config import conf
conf['persist_path'] = '/scratch/ag7531/'
catalog_url = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'
CACHE_FOLDER = '/scratch/ag7531/pangeo_cache'


def get_patch(catalog_url, ntimes : int = None, bounds : list = None,
              c02_level = 0, *selected_vars):
    """Returns a patch of data of the cmip 2.6 model along the grid info"""
    catalog = intake.open_catalog(catalog_url)
    if c02_level == 0:
        source = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    else:
        raise NotImplementedError('Only control implemented for now.')
    s_grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_grid
    my_data = source.get()
    grid_data = s_grid.get()
    my_data.storage_options['cache_folder'] = CACHE_FOLDER
    grid_data.storage_options['cache_folder'] = CACHE_FOLDER
    my_data = my_data.to_dask()
    grid_data = grid_data.to_dask()
    # Following line is necessary to transform non-primary coords into vars
    grid_data = grid_data.reset_coords()
    if bounds is not None:
        my_data = my_data.sel(xu_ocean=slice(*bounds[2:]),
                           yu_ocean=slice(*bounds[:2]))
        grid_data = grid_data.sel(xu_ocean=slice(*bounds[2:]),
                                  yu_ocean=slice(*bounds[:2]))
    if ntimes is not None:
        my_data = my_data.isel(time=slice(0, ntimes))
    
    if selected_vars is None:
        return my_data, grid_data
    else:
        return my_data[list(selected_vars)], grid_data


def get_whole_data():
    data, grid = get_patch(catalog_url, None, None, 0, 'usurf', 'vsurf')
    return data, grid


if __name__ == '__main__':
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/arthur/\
access_key.json"
    data = get_whole_data()
    