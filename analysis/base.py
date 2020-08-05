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
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        return self.ds[i]

    def __setitem__(self, name, value):
        self.ds[name] = value

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

    def __getattr__(self, attr_name):
        if hasattr(self.ds, attr_name):
            return getattr(self.ds, attr_name)
        else:
            raise AttributeError()

    def __setattr__(self, name, value):
        if name != 'ds' and hasattr(self.ds, name):
            setattr(self.ds, name, value)
        else:
            self.__dict__[name] = value

def get_test_datasets(run_id: str):
    """Return a list of the test datasets for the provided run id"""
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    test_outputs = list()
    for a in artifacts:
        basename = os.path.basename(a.path)
        print('.', basename, '.')
        if basename.startswith('test_output_'):
            print('loading')
            ds = xr.open_zarr(client.download_artifacts(run_id, basename))
            test_outputs.append(TestDataset(ds))
    return test_outputs


class DatasetsOnGlobalGrid(xr.Dataset):
    def __init__(self, latitudes, longitudes):
        self.lat = latitudes
        self.lon = longitudes
        self._datasets = []

    def add_dataset(self, dataset: xr.Dataset):
        