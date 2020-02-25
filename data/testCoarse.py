#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:16:33 2020

@author: arthur
Unit tests for the coarse-graining operation from coarse.py
"""

import unittest
from xarray import DataArray
from xarray import Dataset
import numpy as np
from coarse import *
import matplotlib.pyplot as plt

class TestEddyForcing(unittest.TestCase):
    def test_compute_grid_steps(self):
        a1 = DataArray(data = np.zeros((10, 4, 4)), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(10) * 3,
                                 'x' : np.arange(4) * 7,
                                 'y' : np.arange(4) * 11})
        ds = Dataset({'var0' : a1})
        s_x, s_y = compute_grid_steps(ds)
        self.assertEqual(s_x, 7)
        self.assertEqual(s_y, 11)

    def test_spatial_filter_dataset(self):
        a1 = DataArray(data = np.zeros((10, 4, 4)), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(10) * 3,
                                 'x' : np.arange(4) * 7,
                                 'y' : np.arange(4) * 11})
        ds = Dataset({'var0' : a1})
        filtered = spatial_filter_dataset(ds, 2)
        self.assertEqual(filtered.dims, ds.dims)

    def test_spatial_filter_dataset2(self):
        a1 = DataArray(data = np.random.randn(1000, 40, 40), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(1000) * 3,
                                 'x' : np.arange(40) * 7,
                                 'y' : np.arange(40) * 11})
        ds = Dataset({'var0' : a1})
        ds = ds.chunk({'time' : 100})
        filtered = spatial_filter_dataset(ds, 100).compute()
        filtered2 = spatial_filter(ds['var0'].compute().data, 100)
        # ds['var0'].isel(time=0).plot(cmap='coolwarm')
        # plt.figure()
        # filtered['var0'].isel(time=0).plot(cmap='coolwarm')
        # plt.figure()
        # plt.imshow(filtered2[0, :, :], cmap='coolwarm', origin='lower')
        # plt.colorbar()
        test = (filtered.to_array().values == filtered2).all()
        self.assertTrue(test.item())

if __name__ == '__main__':
    unittest.main()
