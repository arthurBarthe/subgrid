#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:19:16 2020

@author: arthur
This script allows to select data from the cmip2.6 simulations made 
available on the Pangeo intake data catalog. Run cmip26 -h to display
help.
"""
print('Starting the script')
import sys

import argparse
from dask.diagnostics import ProgressBar
import mlflow

from data.convert_lat_long import *
from data.coarse import *
from data.pangeo_catalog import get_patch

# Script parameters
catalog_url = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'
description = 'Read data from the CMIP2.6 from a particular region and \
    applies coarse graining.'

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
    params = parser.parse_args('15 35 50 -48 -22 --ntimes 2000'.split())

# Retrieve the patch of data specified in the command-line args
patch_data, grid_data = get_patch(catalog_url, params.ntimes, params.bounds,
                                  0, 'usurf', 'vsurf')
patch_data = patch_data.chunk({'time' : 50})

# Convert to x-y coordinates
print(patch_data)
print(grid_data)

# Calculate eddy-forcing dataset for that particular patch
scale_m = params.scale * 1e3
forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean')
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
forcing.to_zarr('forcing', mode='w')
# Log as an artifact the forcing data
mlflow.log_artifact('forcing')
print('Completed...')   