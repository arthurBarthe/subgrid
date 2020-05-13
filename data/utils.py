#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:40:39 2020

@author: arthur
"""
import mlflow
import xarray as xr

def load_data_from_run(run_id):
    mlflow_client = mlflow.tracking.MlflowClient()
    data_file = mlflow_client.download_artifacts(run_id, 'forcing')
    xr_dataset = xr.open_zarr(data_file).load()
    return xr_dataset

def load_data_from_runs(run_ids):
    xr_datasets = list()
    for run_id in run_ids:
        xr_datasets.append(load_data_from_run(run_id))
    return xr_datasets