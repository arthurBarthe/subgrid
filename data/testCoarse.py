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
from numpy import ma
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
    
    def test_eddy_forcing(self):
        a1 = DataArray(data = np.random.randn(1000, 100, 100), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(1000) * 3,
                                 'x' : np.arange(100) * 10,
                                 'y' : np.arange(100) * 11})
        a2 = DataArray(data = np.random.randn(1000, 100, 100), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(1000) * 3,
                                 'x' : np.arange(100) * 10,
                                 'y' : np.arange(100) * 11})
        ds = Dataset({'usurf' : a1, 'vsurf' : a2})
        forcing = eddy_forcing(ds, 40)
        self.assertTrue(forcing.dims != ds.dims)

    def test_eddy_forcing2(self):
        a1 = DataArray(data = ma.zeros((100, 10, 10)), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(100) * 3,
                                 'x' : np.arange(10) * 10,
                                 'y' : np.arange(10) * 11})
        a1.data.mask = np.zeros((100, 10, 10), dtype = np.bool)
        a1.data.mask[0] = True
        a2 = DataArray(data = np.zeros((100, 10, 10)), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(100) * 3,
                                 'x' : np.arange(10) * 10,
                                 'y' : np.arange(10) * 11})
        ds = Dataset({'usurf' : a1, 'vsurf' : a2})
        forcing = eddy_forcing(ds, 40)

    def test_advections(self):
        a1 = DataArray(data = np.zeros((1000, 40, 40)), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(1000) * 3,
                                 'x' : np.arange(40) * 7,
                                 'y' : np.arange(40) * 11})
        a2= DataArray(data = np.zeros((1000, 40, 40)), 
                       dims = ['time', 'x', 'y'],
                       coords = {'time' : np.arange(1000) * 3,
                                 'x' : np.arange(40) * 7,
                                 'y' : np.arange(40) * 11})
        ds = Dataset({'usurf' : a1, 'vsurf' : a2})
        adv = advections(ds)
        self.assertTrue((adv == 0).all().to_array().all().item())
        

if __name__ == '__main__':
    unittest.main()
