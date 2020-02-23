#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:19:16 2020

@author: arthur
This script allows to select data from the cmip2.6 simulations made 
available on the Pangeo intake data catalog. Run cmip26 -h to display
help.
"""

import argparse
import dask
from dask.diagnostics import ProgressBar
import intake
from convert_lat_long import *
from coarse import *
import mlflow

# Script parameters
catalog_url = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'
description = 'Read data from the CMIP2.6 from a particular region and \
    applies coarse graining.'


def get_patch(params, catalog_url, *selected_vars):
    catalog = intake.open_catalog(catalog_url)
    
    if params.CO2 == 0:
        s = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    else:
        raise('Only control implemented for now.')
    data = s.to_dask()
    my_data = data.sel(xu_ocean=slice(*params.bounds[2:]),
                       yu_ocean=slice(*params.bounds[:2]))
    my_data = my_data.isel(time=slice(0, params.ntimes))
    
    if selected_vars is None:
        return my_data
    else:
        return my_data[list(selected_vars)]



if __name__ == '__main__':
    import sys
    # Parse the command-line parameters
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('scale', type=float, default=30,
                        help='scale in kilometers')
    parser.add_argument('bounds', type=float, nargs=4, 
                        help='min lat, max_lat, min_long, max_long')
    parser.add_argument('--ntimes', type=int, default=100)
    parser.add_argument('--CO2', type=int, default=0, choices=[0,1])
    if len(sys.argv) > 1:
        params = parser.parse_args()
    else:
        params = parser.parse_args('15 0 20 10 15'.split())

    # Retrieve the patch of data specified in the command-line args
    patch_data = get_patch(params, catalog_url, 'usurf', 'vsurf')
    patch_data.chunk({'time' : 50})
    
    # Convert to x-y coordinates
    patch_data = latlong_to_euclidean(patch_data, 'yu_ocean', 'xu_ocean')
    print(patch_data)

    # Calculate eddy-forcing dataset for that particular patch
    scale_m = params.scale * 1e3
    forcing = eddy_forcing(patch_data, scale=scale_m, method='mean')
    pbar = ProgressBar()
    pbar.register()
    # Specify input vs output type for each variable of the dataset. Might
    # be used later on for training or testing.
    forcing['S_x'].attrs['type'] = 'output'
    forcing['S_y'].attrs['type'] = 'output'
    forcing['usurf'].attrs['type'] = 'input'
    forcing['vsurf'].attrs['type'] = 'input'
    forcing = forcing.compute()
    # export data
    forcing.to_zarr('/data/ag7531/outputs/forcing')
    patch_data.to_zarr('data/ag7531/outputs/original')
    # Log as an artifact the forcing data
    mlflow.log_artifact('/data/ag7531/outputs/forcing')
    print('Completed...')   