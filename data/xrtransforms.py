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


class ScalingTransform(Transform):
    def __init__(self, factor: dict):
        self.factor = factor

    def fit(self, x):
        pass

    def transform(self, x: xr.Dataset):
        return self.factor * x

    def inv_transform(self, x: xr.Dataset):
        return 1 / self.factor * x


class SeasonalStdizer(Transform):
    def __init__(self, by: str = 'time.month', dim: str = 'time'):
        self.by = by
        self.dim = dim
        self._means = None
        self._stds = None
        self._grouped = None

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
        # TODO unefficient
        y = x.groupby(self.by) - self.means
        y = y.groupby(self.by) / self.stds
        del y['month']
        return y

    def inv_transform(self, x):
        y = x.groupby(self.by) * self.stds
        y = y.groupby(self.by) + self.means
        del y['month']
        return y
