#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:13:35 2020

@author: arthur
"""
import numpy as np
import xarray as xr

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

    