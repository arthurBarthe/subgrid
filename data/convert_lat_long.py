#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:33:16 2020

@author: arthur
Simple code that locally approximate the planet with a plane to convert
from lat long coordinates to euclidean coordinates in meters.
"""
import numpy as np
from xarray import DataArray, Coordinate

EARTH_RADIUS = 6.371e6

def deg_to_rad(deg: float):
    return deg/ 360.0 * 2 * np.pi

def latlong_to_euclidean_(lat: np.ndarray, long: np.ndarray,
                         ref_lat: float = None, ref_long: float = None):
    # In the case no reference is passed we use the middle points
    if ref_lat is None:
        ref_lat = lat[int(len(lat) / 2)]
    if ref_long is None:
        ref_long = long[int(len(long) / 2)]
    lat_rad = deg_to_rad(lat)
    long_rad = deg_to_rad(long)
    ref_lat = deg_to_rad(ref_lat)
    ref_long = deg_to_rad(ref_long)
    return (EARTH_RADIUS * np.cos(ref_lat) * (long_rad - ref_long), 
            EARTH_RADIUS * (lat_rad - ref_lat))


def latlong_to_euclidean(data: DataArray, lat_name: str, long_name: str):
    lat = data.coords[lat_name]
    long = data.coords[long_name]
    x, y = latlong_to_euclidean_(lat, long)
    data2 = data.assign_coords({long_name: x, lat_name: y})
    data2 = data2.rename({long_name: 'x', lat_name: 'y'})
    return data2


if __name__ == '__main__':
    lat, long = np.arange(10, 11, 0.1), np.arange(0, 360, 1)
    x, y = latlong_to_euclidean(lat, long)