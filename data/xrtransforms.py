#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:21:16 2020

@author: arthur
"""

import xarray as xr
from abc import ABC, abstractmethod
import pickle


class Transform(ABC):
    """Abstract transform class for xarray datasets"""

    @abstractmethod
    def fit(self, x: xr.Dataset):
        pass

    @abstractmethod
    def transform(self, x: xr.Dataset):
        pass

    def apply(self, x: xr.Dataset):
        return self.transform(x)

    def __call__(self, x: xr.Dataset):
        return self.apply(x)

    def fit_transform(self, x: xr.Dataset):
        self.fit(x)
        return self.transform(x)

    def inv_transform(self, x: xr.DataArray):
        raise NotImplementedError('Inverse transform not implemented.')

    def dump(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
        

class ChainedTransform(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def fit(self, x: xr.Dataset):
        for transform in self.transforms:
            x = transform.fit_transform(x)

    def transform(self, x: xr.Dataset):
        for transform in self.transforms:
            x = transform.apply(x)
        return x

    def inv_transform(self, x: xr.Dataset):
        for transform in reversed(self.transforms):
            x = transform.inv_transform(x)
        return x

    def __repr__(self, level=0):
        tabs = '\t' * (level + 1)
        s = 'ChainedTransform(\n' + tabs
        reprs = [t.__repr__(level + 1) if isinstance(t, ChainedTransform)
                 else t.__repr__() for t in self.transforms]
        s2 = (',\n' + tabs).join(reprs)
        s3 = '\n' + tabs[:-1] + ')'
        return ''.join((s, s2, s3))


class ScalingTransform(Transform):
    def __init__(self, factor: dict = None):
        self.factor = factor

    def fit(self, x):
        pass

    def transform(self, x: xr.Dataset):
        return self.factor * x

    def inv_transform(self, x: xr.Dataset):
        return 1 / self.factor * x


class SeasonalStdizer(Transform):
    def __init__(self, by: str = 'time.month', dim: str = 'time',
                 std: bool = True):
        self.by = by
        self.dim = dim
        self._means = None
        self._stds = None
        self._grouped = None
        self.apply_std = std

    @property
    def grouped(self):
        return self._grouped

    @grouped.setter
    def grouped(self, value):
        self._grouped = value

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, value):
        self._means = value

    @property
    def stds(self):
        return self._stds

    @stds.setter
    def stds(self, value):
        self._stds = value

    def fit(self, x):
        self.grouped = x.groupby(self.by)
        self.means = self.grouped.mean(dim=self.dim).compute()
        self.stds = self.grouped.std(dim=self.dim).compute()

    def transform(self, x):
        # TODO this only works for the specific case 'time' and 'month' rn
        months = x['time'].dt.month
        means = self.means.sel(month=months)
        y = (x - means)
        if self.apply_std:
            stds = self.stds.sel(month=months)
            return y / stds
        return y

    def inv_transform(self, x):
        months = x['time'].dt.month
        means = self.means.sel(month=months)
        stds = self.stds.sel(month=months)
        return x * stds + means


class CropToNewShape(Transform):
    def __init__(self, new_shape: dict = None):
        self.new_shape = new_shape

    def fit(self, x):
        pass

    @staticmethod
    def get_slice(length: int, length_to: int):
        d_left = max(0, (length - length_to) // 2)
        d_right = d_left + max(0, (length - length_to)) % 2
        return slice(d_left, length - d_right)

    def transform(self, x):
        dims = x.dims
        idx = {dim_name: self.get_slice(dims[dim_name], dim_size)
               for dim_name, dim_size in self.new_shape.items()}
        return x.isel(idx)

    def __repr__(self):
        return f'CropToNewShape({self.new_shape})'


class CropToMinSize(CropToNewShape):
    def __init__(self, datasets, dim_names: list):
        new_shape = {dim_name: min([dataset.dims[dim_name]
                                    for dataset in datasets])
                     for dim_name in dim_names}
        super().__init__(new_shape)

    def __repr__(self):
        return super().__repr__() + '(CropToMinSize)'


class CropToMultipleOf(CropToNewShape):
    def __init__(self, multiples: dict):
        self.multiples = multiples

    @staticmethod
    def get_multiple(p: int, m: int):
        return p // m * m

    def transform(self, x):
        dims = x.dims
        new_sizes = {dim_name: self.get_multiple(dims[dim_name], m)
                     for dim_name, m in self.multiples.items()}
        idx = {dim_name: self.get_slice(dims[dim_name], new_sizes[dim_name])
               for dim_name, multiple in self.multiples.items()}
        return x.isel(idx)

    def __repr__(self):
        return f'CropToMultipleOf({self.multiples})'
