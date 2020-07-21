#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:19:16 2020

@author: arthur
This script computes the subgrid forcing for a given region, using
data from cmip2.6 on one of the pangeo data catalogs.
Command line parameters include region specification.
Run cmip26 -h to display help.
"""

import argparse
from dask.diagnostics import ProgressBar
import mlflow
from copy import copy

from data.coarse import eddy_forcing
from data.pangeo_catalog import get_patch

# Script parameters
CATALOG_URL = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'
DESCRIPTION = 'Read data from the CMIP2.6 from a particular region and \
    applies coarse graining.'

# Parse the command-line parameters
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('scale', type=float, help='scale in kilometers')
parser.add_argument('bounds', type=float, nargs=4, help='min lat, max_lat,\
                    min_long, max_long')
parser.add_argument('--ntimes', type=int, default=100, help='number of days,\
                    starting from first day.')
parser.add_argument('--CO2', type=int, default=0, choices=[0, 1], help='CO2\
                    level, O (control) or 1.')
parser.add_argument('--factor', type=int, default=0,
                    help='Factor of degrading')
params = parser.parse_args()

# Use a larger patch to compute the eddy forcing then we will crop
extra_bounds = copy(params.bounds)
extra_bounds[0] -= 2 * params.scale / 10
extra_bounds[2] -= 2 * params.scale / 10
extra_bounds[1] += 2 * params.scale / 10
extra_bounds[3] += 2 * params.scale / 10
# Retrieve the patch of data specified in the command-line args
patch_data, grid_data = get_patch(CATALOG_URL, params.ntimes, extra_bounds,
                                  params.CO2, 'usurf', 'vsurf')
# patch_data = patch_data.chunk({'time': 50})

print(patch_data)
print(grid_data)

# Calculate eddy-forcing dataset for that particular patch
if params.factor != 0:
    scale_m = params.factor
    forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean',
                           scale_mode='factor')
else:
    scale_m = params.scale * 1e3
    forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean')

# Progress bar
ProgressBar().register()

# Specify input vs output type for each variable of the dataset. Might
# be used later on for training or testing.
forcing['S_x'].attrs['type'] = 'output'
forcing['S_y'].attrs['type'] = 'output'
forcing['usurf'].attrs['type'] = 'input'
forcing['vsurf'].attrs['type'] = 'input'

# Crop according to bounds
bounds = params.bounds
forcing = forcing.sel(xu_ocean=slice(bounds[2], bounds[3]),
                      yu_ocean=slice(bounds[0], bounds[1]))
print('Preparing forcing data')
print(forcing)
# export data
# forcing = forcing.compute()
forcing.to_zarr('forcing', mode='w')

# Log as an artifact the forcing data
mlflow.log_artifact('forcing')
print('Completed...')
