#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:39:23 2020

@author: arthur
Script that lists the data runs with the relevant information
"""
DATA_EXPERIMENT_ID = '1'

import mlflow
from mlflow.tracking import MlflowClient
import xarray as xr
import matplotlib.pyplot as plt

def show_data_sample(run_id, index: int):
    """Plots the data for the given index"""
    client = MlflowClient()
    forcing_data = client.download_artifacts(run_id, 'forcing')
    forcing = xr.open_zarr(forcing_data)
    sample = forcing.isel(time=index)
    for var in sample.values():
        plt.figure()
        var.plot()
    plt.show()


runs = mlflow.search_runs(experiment_ids=[DATA_EXPERIMENT_ID,])
runs_short = runs[['run_id', 'start_time', 'params.scale', 'params.ntimes', 
            'params.lat_min', 'params.lat_max', 
            'params.long_min', 'params.long_max']]

print('Select a run by entering its integer id (0, 1, ...) as listed\
      from below, in order to obtain more details.')
print(runs_short)
while True:
    selection = input('Select run: ')
    try:
        selection = int(selection)
    except ValueError as e:
        break
    print(runs.iloc[selection, :])
    show_data_sample(runs.iloc[selection, :]['run_id'], 0)

