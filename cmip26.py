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


# logging config
logging_level = os.environ.get('LOGGING_LEVEL')
if logging_level is not None:
    logging_level = getattr(logging, logging_level)
    logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)


# Script parameters
CATALOG_URL = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'

DESCRIPTION = 'Read data from the CM2.6 and \
        apply coarse graining. Stores the resulting dataset into an MLFLOW \
        experiment within a specific run.'

data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')

# Parse the command-line parameters
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('bounds', type=float, nargs=4, help='min lat, max_lat,\
                    min_long, max_long')
parser.add_argument('--global_', type=int, help='True if global data. In this\
                    case the data is made cyclic along longitude', 
                    default=False)
parser.add_argument('--ntimes', type=int, default=10000, help='number of days,\
                    starting from first day.')
parser.add_argument('--CO2', type=int, default=0, choices=[0, 1], help='CO2\
                    level, O (control) or 1 (1 percent CO2 increase)')
parser.add_argument('--factor', type=int, default=0,
                    help='Factor of degrading. Should be integer > 1.')
parser.add_argument('--chunk_size', type=str, default='50',
                    help='Chunk size along the time dimension')
params = parser.parse_args()


# Retrieve the patch of data specified in the command-line args
patch_data, grid_data = get_patch(CATALOG_URL, params.ntimes, params.bounds,
                                  params.CO2, 'usurf', 'vsurf')


logger.debug(patch_data)
logger.debug(grid_data)

# If global data, we make the dataset cyclic along longitude
if params.global_ == 1:
    logger.info('Cyclic data... Making the dataset cyclic along longitude...')
    patch_data = cyclize_dataset(patch_data, 'xu_ocean', params.factor)
    grid_data = cyclize_dataset(grid_data, 'xu_ocean', params.factor)
    # Rechunk along the cyclized dimension
    patch_data = patch_data.chunk(dict(xu_ocean=-1))
    grid_data = grid_data.chunk(dict(xu_ocean=-1))

# Calculate eddy-forcing dataset for that particular patch
debug_mode = os.environ.get('DEBUG_MODE')
if params.factor != 0 and not debug_mode:
    scale_m = params.factor
    forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean',
                           scale_mode='factor')
elif not debug_mode:
    scale_m = params.scale * 1e3
    forcing = eddy_forcing(patch_data, grid_data, scale=scale_m, method='mean')
else:
    logger.info('!!!Debug mode!!!')
    forcing = patch_data

# Progress bar
ProgressBar().register()

# Specify input vs output type for each variable of the dataset. Might
# be used later on for training or testing.
if not debug_mode:
    forcing['S_x'].attrs['type'] = 'output'
    forcing['S_y'].attrs['type'] = 'output'
    forcing['usurf'].attrs['type'] = 'input'
    forcing['vsurf'].attrs['type'] = 'input'

# Crop according to bounds
bounds = params.bounds
forcing = forcing.sel(xu_ocean=slice(bounds[2], bounds[3]),
                      yu_ocean=slice(bounds[0], bounds[1]))

logger.info('Preparing forcing data')
logger.debug(forcing)
# export data
forcing.to_zarr(join(data_location, 'forcing'), mode='w')

# Log as an artifact the forcing data
logger.info('Logging processed dataset as an artifact...')
mlflow.log_artifact(join(data_location, 'forcing'))
logger.info('Completed...')
