#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:13:35 2020

@author: arthur
"""
import numpy as np
import xarray as xr
import mlflow
import os.path

class TestDataset:
    def __init__(ds):
        self.ds = ds

    def __getitem__(self, i):
        return self.ds[i]

    def errors(self, normalized=False):
        sx_error = self['S_xpred'] - self['S_x']
        sy_error = self['S_ypred'] - self['S_y']
        if normalized:
            sx_error *= self['S_xscale']
            sy_error *= self['S_yscale']
        return xr.Dataset({'S_x': sx_error, 'S_y': sy_error})

    def rmse(self, dim: str, normalized=False):
        errors = self.errors(normalized)
        return np.sqrt((errors['S_x']**2 + errors['S_y']**2).mean(dim=dim))

def get_test_datasets(run_id: str):
    """Return a list of the test datasets for the provided run id"""
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    test_outputs = list()
    for a in artifacts:
        if a.is_dir():
            continue
        basename = os.path.basename(a.path)
        if basename.startswith('test_ouput_'):
            ds = xr.open_zarr(client.download_artifacts(run_id, basename))
            test_outputs.append(TestDataset(ds))
    return test_outputs

def get_merged_errors(test_datasets, error_func, dim, *args):
    """Compute the summary errors for a list of test datasets and merge those
    errors"""
    errors = list()
    for i, ds in enumerate(test_datasets):
        error = getattr(ds, error_func)(dim, *args)
        error.name = error_func
        errors.append(error)
    return xr.merge(errors)
    