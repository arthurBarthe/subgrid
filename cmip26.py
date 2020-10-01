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

from data.utils import cyclize_dataset
from data.coarse import eddy_forcing
from data.pangeo_catalog import get_patch
import logging
import tempfile
from os.path import join
import os

import dask
dask.config.set(dict(temporary_directory='/scratch/ag7531/dasktemp/'))

# logging config
logging_level = os.environ.get('LOGGING_LEVEL')
if logging_level is not None:
    logging_level = getattr(logging, logging_level)
    logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)


# Script parameters
CATALOG_URL = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'
DESCRIPTION = 'Read data from the CM2.6 from a particular region and \
        apply coarse graining. Stores the resulting dataset into an MLFLOW \
        experiment within a specific run.'

data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')

# Parse the command-line parameters
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('scale', type=float, help='scale in kilometers')
parser.add_argument('bounds', type=float, nargs=4, help='min lat, max_lat,\
                    min_long, max_long')
parser.add_argument('--global_', type=bool, help='True if global data. In this\
                    case the data is made cyclic along longitude', 
                    default=False)
parser.add_argument('--ntimes', type=int, default=100, help='number of days,\
                    starting from first day.')
parser.add_argument('--CO2', type=int, default=0, choices=[0, 1], help='CO2\
                    level, O (control) or 1 (1 percent CO2 increase)')
parser.add_argument('--factor', type=int, default=0,
                    help='Factor of degrading. Should be integer > 1.')
parser.add_argument('--chunk_size', type=str, default='50',
                    help='Chunk size along the time dimension')
params = parser.parse_args()

# Use a larger patch to compute the eddy forcing then we will crop.
# This is to mitigate effects due to filtering near the border.
extra_bounds = copy(params.bounds)
extra_bounds[0] -= 2 * params.scale / 10
extra_bounds[2] -= 2 * params.scale / 10
extra_bounds[1] += 2 * params.scale / 10
extra_bounds[3] += 2 * params.scale / 10

# Retrieve the patch of data specified in the command-line args
patch_data, grid_data = get_patch(CATALOG_URL, params.ntimes, extra_bounds,
                                  params.CO2, 'usurf', 'vsurf')

logger.debug(patch_data)
logger.debug(grid_data)

# If global data, we make the dataset cyclic along longitude
if params.global_:
    pass
    # logger.info('Cyclic data... Making the dataset cyclic along longitude...')
    # patch_data = cyclize_dataset(patch_data, 'xu_ocean', 4)
    # grid_data = cyclize_dataset(grid_data, 'xu_ocean', 4)

chunk_sizes = list(map(int, params.chunk_size.split('/')))
patch_data = patch_data.chunk(dict(zip(('time', 'xu_ocean', 'yu_ocean'),
                                       chunk_sizes)))

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
while len(chunk_sizes) < 3:
    chunk_sizes.append('auto')
forcing = forcing.chunk(dict(zip(('time', 'xu_ocean', 'yu_ocean'),
                                 chunk_sizes)))
logger.info('Preparing forcing data')
logger.debug(forcing)
# export data
forcing.to_zarr(join(data_location, 'forcing'), mode='w')

# Log as an artifact the forcing data
logger.info('Logging processed dataset as an artifact...')
mlflow.log_artifact(join(data_location, 'forcing'))
logger.info('Completed...')
