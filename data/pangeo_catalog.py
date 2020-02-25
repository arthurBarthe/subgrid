#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:20 2020

@author: arthur
"""

import intake

catalog_url = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'

def get_patch(catalog_url, ntimes : int = None, bounds : list = None,
              c02 = 0, *selected_vars):
    catalog = intake.open_catalog(catalog_url)
    if c02 == 0:
        s = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    else:
        raise('Only control implemented for now.')
    my_data = s.to_dask()
    if bounds is not None:
        my_data = my_data.sel(xu_ocean=slice(*bounds[2:]),
                           yu_ocean=slice(*bounds[:2]))
    if ntimes is not None:
        my_data = my_data.isel(time=slice(0, ntimes))
    
    if selected_vars is None:
        return my_data
    else:
        return my_data[list(selected_vars)]
    