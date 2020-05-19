# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:38:40 2020

@author: Arthur
TODO list
balance the weights when mixing data sets

"""
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np
import os.path
import matplotlib.pyplot as plt
# import mlflow
# from sklearn.preprocessing import StandardScaler
import xarray as xr
import logging
import bisect
from copy import deepcopy
from abc import ABC, abstractmethod


def call_only_once(f):
    """Decorator that ensures a function is only called at most once for
    a given set of parameters."""
    f.called = list()

    def new_f(*args, **kargs):
        if not (args, kargs) in f.called:
            f.called.append((args, kargs))
            return f(*args, **kargs)
        else:
            raise Exception("This method should be called at most once \
                            for a given set of parameters.")
    return new_f


class FeaturesTargetsDataset(Dataset):
    """Simple dataset based on an array of features and an array of targets"""
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets
        assert(len(self.features) == len(self.targets))

    def __getitem__(self, index: int):
        return (self.features[index], self.targets[index])

    def __len__(self):
        return len(self.features)


def prod(l):
    """Returns the product of the elements of an iterable."""
    if len(l) == 0:
        return 1
    else:
        return l[0] * prod(l[1:])


class DatasetTransformer:
    def __init__(self, features_transform, targets_transform=None):
        self.transforms = dict()
        self.transforms['features'] = features_transform
        if targets_transform is None:
            targets_transform = deepcopy(features_transform)
        self.transforms['targets'] = targets_transform

    def fit(self, x: torch.utils.data.Dataset):
        features, targets = x[:]
        self.transforms['features'].fit(features)
        self.transforms['targets'].fit(targets)
        return self

    def transform(self, x):
        features, targets = x
        new_features = self.transforms['features'].transform(features)
        new_targets = self.transforms['targets'].transform(targets)
        return new_features, new_targets

    def __call__(self, x):
        return self.transform(x)

    def inverse_transform(self, x: Dataset):
        features, targets = x
        new_features = self.transforms['features'].inverse_transform(features)
        new_targets = self.transforms['targets'].transform(targets)
        return FeaturesTargetsDataset(new_features, new_targets)


class ArrayTransform(ABC):
    def __call__(self, x):
        return self.transform(x)

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass


class ComposeTransforms(ArrayTransform):
    def __init__(self, *transforms):
        self.transforms = transforms

    def fit(self, x):
        for transform in self.transforms:
            transform.fit(x)
            # TODO for something more general the following line should work
            # x = transform.transform(x)

    def transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class CropToMultipleof(ArrayTransform):
    def __init__(self, multiple_of: int = 2):
        self.multiple_of = multiple_of

    def fit(self, x):
        shape = x.shape
        new_shape_1 = shape[2] // self.multiple_of * self.multiple_of
        new_shape_2 = shape[3] // self.multiple_of * self.multiple_of
        self.new_shape = (shape[1], new_shape_1, new_shape_2)

    def transform(self, x):
        return x[:, :self.new_shape[1], :self.new_shape[2]]


class PerChannelNormalizer(ArrayTransform):
    def __init__(self, fit_only_once=True):
        self.fit_only_once = fit_only_once
        self._std = None
        self._mean = None

    def fit(self, X: np.ndarray):
        if (not self.fit_only_once) or (self._mean is None):
            mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            std = np.std(X, axis=(0, 2, 3), keepdims=True)
            self._mean = mean.reshape(mean.shape[1:])
            self._std = std.reshape(std.shape[1:])

    def transform(self, X: np.ndarray):
        assert(self._mean is not None)
        return (X - self._mean) / self._std

    def inverse_transform(self, X):
        return X * self._std + self._mean


class PerLocationNormalizer(ArrayTransform):
    def __init__(self):
        self._std = None
        self._mean = None

    def fit(self, X: np.ndarray):
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        self._mean = mean.reshape(mean.shape[1:])
        self._std = std.reshape(std.shape[1:])

    def transform(self, X: np.ndarray):
        assert(self._mean is not None)
        return (X - self._mean) / self._std


class PerInputNormalizer(ArrayTransform):
    def fit(self, X):
        pass

    def transform(self, X: np.ndarray):
        mean = np.mean(X, axis=(1, 2), keepdims=True)
        std = np.std(X, axis=(1, 2), keepdims=True)
        return (X - mean) / std


class DatasetClippedScaler(DatasetTransformer):
    def __init__(self, apply_both=True):
        super().__init__(transformer_class=StandardScaler)

    def fit(self, X: Dataset):
        super().fit(X)
        scale_features = np.clip(self.transformers['features'].scale_,
                                 1e-4, np.inf)
        scale_targets = np.clip(self.transformers['targets'].scale_,
                                1e-4, np.inf)
        self.transformers['features'].scale_ = scale_features
        self.transformers['targets'].scale_ = scale_targets


class RawDataFromXrDataset(Dataset):
    """This class allows to define a Pytorch Dataset based on an xarray 
    dataset easily, specifying features and targets."""
    def __init__(self, dataset: xr.Dataset):
        self.xr_dataset = dataset
        self._input_arrays = list()
        self._output_arrays = list()
        self._index = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: str):
        self._index = index

    @property
    def output_arrays(self):
        return self._output_arrays

    @output_arrays.setter
    def output_arrays(self, str_list: list):
        for array_name in str_list:
            self._check_varname(array_name)
        self._output_arrays = str_list

    @property
    def input_arrays(self):
        return self._input_arrays

    @input_arrays.setter
    def input_arrays(self, str_list):
        for array_name in str_list:
            self._check_varname(array_name)
        self._input_arrays = str_list

    @property
    def features(self):
        return self.xr_dataset[self.input_arrays]

    @property
    def targets(self):
        return self.xr_dataset[self.output_arrays]

    @property
    def n_targets(self):
        return len(self.targets)

    @property
    def n_features(self):
        return len(self.features)

    def add_output(self, varname):
        self._check_varname(varname)
        self._output_arrays.append(varname)

    def add_input(self, varname: str):
        self._check_varname(varname)
        self._input_arrays.append(varname)

    @property
    def width(self):
        candidates = []
        for dim_name in self.xr_dataset.dims:
            if dim_name.startswith('x'):
                candidates.append(dim_name)
        if len(candidates) == 1:
            x_dim_name = candidates[0]
        elif 'x' in candidates:
            x_dim_name = 'x'
        else:
            raise Exception('Could not determine width axis according \
                            to convention')
        return len(self.xr_dataset[x_dim_name])

    @property
    def height(self):
        candidates = []
        for dim_name in self.xr_dataset.dims:
            if dim_name.startswith('y'):
                candidates.append(dim_name)
        if len(candidates) == 1:
            y_dim_name = candidates[0]
        elif 'y' in candidates:
            y_dim_name = 'y'
        else:
            raise Exception('Could not determine width axis according \
                            to convention')
        return len(self.xr_dataset[y_dim_name])

    def __getitem__(self, index):
        try:
            features = self.features.isel({self._index: index})
            features = features.to_array().data
            targets = self.targets.isel({self._index: index})
            targets = targets.to_array().data
            # to_array method stacks variables along first dim, hence next line
            if not isinstance(index, (int, np.int64, np.int_)):
                features = features.swapaxes(0, 1)
                targets = targets.swapaxes(0, 1)
        except KeyError as e:
            e.msg = e.msg + '\n Make sure you have defined the index, inputs,\
                and outputs.'
            raise e
        return features, targets

    def __len__(self):
        try:
            return len(self.xr_dataset[self._index])
        except KeyError as e:
            raise type(e)('Make sure you have defined the index: ' + str(e))

    def _check_varname(self, var_name: str):
        if var_name not in self.xr_dataset:
            raise KeyError('Variable not in the xarray dataset.')
        if var_name in self._input_arrays or var_name in self._output_arrays:
            raise ValueError('Variable already added as input or output.')


class DatasetWithTransform:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    @property
    def height(self):
        """Since the transform can modify the height..."""
        x = self[0][0]
        return x.shape[1]

    @property
    def width(self):
        x = self[0][0]
        return x.shape[2]

    def __getitem__(self, index: int):
        return self.transform(self.dataset[index])

    def __getattr__(self, attr):
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError()

    def __len__(self):
        return len(self.dataset)

    def inverse_transform(self, x):
        return self.transform.inverse_transform(x)


class Subset_(Subset):
    """Extends the Pytorch Subset class to allow for attributes of the 
    dataset to be propagated to the subset dataset"""
    def __init__(self, dataset, indices):
        super(Subset_, self).__init__(dataset, indices)

    def __getattr__(self, attr):
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError()


class ConcatDataset_(ConcatDataset):
    def __init__(self, datasets, enforce_same_dims=True):
        super(ConcatDataset_, self).__init__(datasets)
        self.enforce_same_dims = enforce_same_dims
        if enforce_same_dims:
            heights = [dataset.height for dataset in self.datasets]
            widths = [dataset.width for dataset in self.datasets]
        self.height = min(heights)
        self.width = min(widths)

    def __getattr__(self, attr):
        print('Trying ', attr)
        if hasattr(self.datasets[0], attr):
            return getattr(self.datasets[0], attr)
        else:
            raise AttributeError()

    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        if self.enforce_same_dims:
            result = (result[0][:, :self.height, :self.width],
                      result[1][:, :self.height, :self.width])
        return result


class LensDescriptor:
    def __get__(self, obj, type=None):
        lens = np.array([len(dataset) for dataset in obj.datasets])
        obj.__dict__[self.name] = lens
        return lens

    def __set_name__(self, owner, name):
        self.name = name


class RatiosDescriptor:
    def __get__(self, obj, type=None):
        if not obj.balanced:
            ratios = (obj.lens * obj.precision) // np.min(obj.lens)
        else:
            ratios = np.ones((len(obj.lens),))
        obj.__dict__[self.name] = ratios
        return ratios

    def __set_name__(self, owner, name):
        self.name = name


class MixedDatasets(Dataset):
    """Similar to the ConcatDataset from pytorch, with the difference that 
    the datasets are not concatenated one after another, but instead mixed.
    For instance if we mix two datasets d1 and d2 that have the same size,
    and d = MixedDatasets((d1, d2)), then d[0] returns the first element of 
    d1, d[1] returns the first element of d2, d[2] returns the second element
    of d1, and so on. In the case where the two datasets do not have the same
    size, see the __getitem__ documentation for more information of selection
    behaviour."""
    lens = LensDescriptor()
    ratios = RatiosDescriptor()

    def __init__(self, datasets, transforms=None, balanced=True):
        self.datasets = datasets
        self.precision = 1
        self.transforms = transforms
        self.balanced = balanced

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
        self._datasets = datasets
        # Delete instance attribute lens if it exists so that the descriptor
        # is called on next access to re-compute
        self.__dict__.pop('lens', None)
        self.__dict__.pop('ratios', None)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value: int):
        self._precision = value
        self.__dict__.pop('ratios', None)

    @property
    def balanced(self):
        return self._balanced

    @balanced.setter
    def balanced(self, balanced):
        self._balanced = balanced
        self.__dict__.pop('ratios', None)

    def __len__(self):
        return min(self.lens // self.ratios) * np.sum(self.ratios)

    def __getitem__(self, index):
        cum_sum = np.cumsum(self.ratios)
        remainer = index % cum_sum[-1]
        dataset_idx = bisect.bisect_right(cum_sum, remainer)
        sub_idx = index // cum_sum[-1]
        if self.transforms is not None:
            transform = self.transforms[dataset_idx]
        else:
            transform = lambda x: x
        return transform(self.datasets[dataset_idx][sub_idx * cum_sum[-1]])


class MixedDataFromXrDataset(MixedDatasets):
    def __init__(self, datasets, index: str, transforms):
        self.datasets = list(map(RawDataFromXrDataset, datasets))
        self.index = index
        super().__init__(self.datasets, transforms)

    @staticmethod
    def all_equal(l):
        v = l[0]
        for value in l:
            if value != v:
                return False
        return True

    @property
    def features(self):
        for dataset in self.datasets:
            yield dataset.features

    @property
    def targets(self):
        for dataset in self.datasets:
            yield dataset.targets

    @property
    def n_features(self):
        n_features = [d.n_features for d in self.datasets]
        if not self.all_equal(n_features):
            raise ValueError('All datasets do not have the same number of \
                             features')
        else:
            return n_features[0]

    @property
    def n_targets(self):
        n_targets = [d.n_targets for d in self.datasets]
        if not self.all_equal(n_targets):
            raise ValueError('All datasets do not have the same number of \
                             targets')
        else:
            return n_targets[0]

    @property
    def height(self):
        heights = [dataset.height for dataset in self.datasets]
        if not self.all_equal(heights):
            logging.warn('Concatenated datasets do not have the same height')
        return heights[0]

    @property
    def width(self):
        widths = [dataset.width for dataset in self.datasets]
        if not self.all_equal(widths):
            logging.warn('Concatenated datasets do not have the same height')
        return widths[0]

    def add_input(self, var_name: str) -> None:
        for dataset in self.datasets:
            dataset.add_input(var_name)

    def add_output(self, var_name: str) -> None:
        for dataset in self.datasets:
            dataset.add_output(var_name)

    @property
    def index(self):
        for dataset in self.datasets:
            yield dataset.index

    @index.setter
    def index(self, index: str):
        for dataset in self.datasets:
            dataset.index = index


class RawData(Dataset):
    # TODO This should be made an abstract class.
    """This class produces a raw dataset by doing the basic transformations,
    using the psi, s_x and s_y data."""
    def __init__(self, data_location: str, 
                 f_name_psi: str = 'psi_coarse.npy',
                 f_name_sx: str = 'sx_coarse.npy',
                 f_name_sy: str = 'sy_coarse.npy'):
        # Load the data from the disk
        psi = np.load(os.path.join(data_location, f_name_psi))
        sx = np.load(os.path.join(data_location, f_name_sx))
        sy = np.load(os.path.join(data_location, f_name_sy))
        width, height = psi.shape[-2:]
        # Scale uniformly
        psi /= np.std(psi)
        sx /= np.std(sx)
        sy /= np.std(sy)
        # Add channel dimension to input
        psi = self._add_channel(psi)
        sx = self._flatten(sx)
        sy = self._flatten(sy)
        self.psi = psi
        self.sx = sx
        self.sy = sy
        self.features = self.psi
        self.target = np.hstack((self.sx, self.sy))
        self.width = width
        self.height = height

    def __getitem__(self, index):
        return (self.features[index], self.target[index])

    def _add_channel(self, array: np.ndarray) -> np.ndarray:
        return np.reshape(array, (-1, 1, array.shape[1], array.shape[2]))

    def _flatten(self, array: np.ndarray) -> np.ndarray:
        return np.reshape(array, (-1, array.shape[1] * array.shape[2]))

    @property
    def n_output_targets(self):
        """Returns the size of the prediction."""
        return self.sx.shape[1] + self.sy.shape[1]

    @property
    def data_length(self):
        return self.psi.shape[0]

    def __len__(self):
        return self.psi.shape[0]

    def plot_true_vs_pred(self, true: np.ndarray, predicted: np.ndarray):
        true_sx, true_sy = np.split(true, 2, axis=1)
        pred_sx, pred_sy = np.split(predicted, 2, axis=1)
        fig = plt.figure()
        plt.subplot(231)
        self._imshow(self.flat_to_2d(true_sx))
        plt.title('true S_x')
        plt.subplot(234)
        self._imshow(self.flat_to_2d(true_sy))
        plt.title('true S_y')
        plt.subplot(232)
        self._imshow(self.flat_to_2d(pred_sx))
        plt.title('predicted S_x')
        plt.subplot(235)
        self._imshow(self.flat_to_2d(pred_sy))
        plt.title('predicted S_y')
        plt.subplot(233)
        self._imshow(self.flat_to_2d(pred_sx - true_sx))
        plt.title('relative difference')
        plt.subplot(236)
        self._imshow(self.flat_to_2d(pred_sy - true_sy))
        plt.title('relative difference')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        return fig

    def plot_features(self, features: np.ndarray):
        """Plots the passed features, accounting for the shape of the data.
        This only plots one feature sample (possibly having different
        channels though)."""
        assert(features.ndim == 3)
        features = self.inv_pre_process(features)
        plt.figure()
        n_channels = features.shape[0]
        for i_channel in range(n_channels):
            plt.subplot(1, n_channels, i_channel + 1)
            self._imshow(features[i_channel, ...])

    def plot_targets(self, targets: np.ndarray):
        assert(targets.ndim == 1)
        targets = self.inv_pre_process(None, targets)
        sx, sy = np.split(targets, 2)
        plt.subplot(121)
        self._imshow(self.flat_to_2d(sx))
        plt.subplot(122)
        self._imshow(self.flat_to_2d(sy))

    def _imshow(self, data: np.ndarray, *args, **kargs):
        """Wrapper function for the imshow function that normalizes the data
        beforehand."""
        data = data / np.std(data)
        plt.imshow(data, origin='lower', cmap='jet',
                   vmin=-1.96, vmax=1.96, *args, **kargs)
        plt.colorbar()

    def flat_to_2d(self, data: np.ndarray):
        return data.reshape(self.width, self.height)


class MultipleTimeIndices(Dataset):
    """Class to create a dataset based on an existing dataset where we
    concatenate multiple time indices along the channel dimension to create a
    new feature"""
    def __init__(self, dataset: Dataset, time_indices: list() = None):
        self.dataset = dataset
        self._time_indices = None
        self._shift = 0
        if time_indices is not None:
            self.time_indices = time_indices

    @property
    def time_indices(self):
        if self._time_indices:
            return self._time_indices
        else:
            return [0, ]

    @time_indices.setter
#    @call_only_once
    def time_indices(self, indices: list):
        for i in indices:
            if i > 0:
                raise ValueError('The indices should be 0 or negative')
        self._time_indices = indices
        self._shift = max([abs(v) for v in indices])
        mlflow.log_param('time_indices', indices)

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value: int):
        raise Exception('The shift cannot be set manually. Instead set \
                        the time indices.')

    def _build_features(self):
        indices = np.arange(len(self))[:, np.newaxis]
        indices = indices + self.shift + self.time_indices
        features = self.dataset[:][0][indices]
        self.features = np.take(features, 0, axis=2)

    def __getitem__(self, index):
        """Returns the sample indexed by the passed index."""
        if not hasattr(self, 'features'):
            self._build_features()
        # TODO check this does not slows things down. Hopefully should not,
        # as it should just be a memory view.
        feature = self.features[index]
        target = self.dataset[index + self.shift][1]
        return (feature, target)

    def __len__(self):
        """Returns the number of samples available in the dataset. Note that
        this might be less than the actual size of the first dimension
        if self.indices contains values other than 0, i.e. if we are
        using some data from the past to make predictions"""
        return len(self.dataset) - self.shift


class Dataset_psi_s(Dataset):
    """Loads the data from the disk into memory and produces data items on
    demand."""
    def __init__(self, data_location, file_name_psi, file_name_sx,
                 file_name_sy):
        # Load the data from the disk
        self.psi = np.load(os.path.join(data_location, file_name_psi))
        self.sx = np.load(os.path.join(data_location, file_name_sx))
        self.sy = np.load(os.path.join(data_location, file_name_sy))
        assert self.psi.shape[0] == self.sx.shape[0] == self.sy.shape[0], \
            'Error: the lengths of the arrays differ'
        self.indices = [0, ]
        self.shift = 0
        self.clip_psi_std = (1e-9, np.inf)
        self.clip_target_std = (1e-9, np.inf)
        # TODO implement this in the parent class
        mlflow.log_param('clip_features', str(self.clip_psi_std))
        mlflow.log_param('clip_targets', str(self.clip_target_std))

    def pre_process(self, train_index):
        """Operates some basic pre-processing on the data, such as
        rescaling."""
        self.width, self.height = self.psi.shape[-2:]
        # input: add channel dimension
        self.psi = self.psi.reshape(-1, 1, self.width, self.height)
        # input: remove mean, normalize
        self.mean_psi = np.mean(self.psi[:train_index], axis=0)
        self.std_psi = np.std(self.psi[:train_index], axis=0)
        clip_min, clip_max = self.clip_psi_std
        self.std_psi = np.clip(self.std_psi, clip_min, clip_max)
        self.psi = (self.psi - self.mean_psi) / self.std_psi
        # output: flatten
        self.sx = self.sx.reshape(self.sx.shape[0], -1)
        self.sy = self.sy.reshape(self.sy.shape[0], -1)
        mlflow.log_param('output_mean_removal', 'False')
        # output: divide by std
        clip_min, clip_max = self.clip_target_std
        self.std_sx = np.std(self.sx[:train_index], axis=0)
        self.std_sy = np.std(self.sy[:train_index], axis=0)
        self.std_sx = np.clip(self.std_sx, clip_min, clip_max)
        self.std_sy = np.clip(self.std_sy, clip_min, clip_max)
        self.std_targets = np.concatenate((self.std_sx, self.std_sy))
        self.sx /= self.std_sx
        self.sy /= self.std_sy


if __name__ == '__main__':
    import xarray as xr
    from xarray import DataArray
    from xarray import Dataset as xrDataset
    from torch.utils.data import DataLoader
    from numpy.random import randint
    from copy import deepcopy
    da = DataArray(data=randint(0, 10, (20, 32, 48)), dims=('time', 'yu', 'xu'))
    da2 = DataArray(data=randint(0, 3, (20, 32, 48)), dims=('time', 'yu', 'xu'))
    da3 = DataArray(data=randint(0, 100, (20, 32, 48)) * 10, dims=('time', 'yu', 'xu'))
    da4 = DataArray(data=randint(0, 2, (20, 32, 48)) * 20, dims=('time', 'yu', 'xu'))
    ds = xrDataset({'in0': da, 'in1': da2,
                    'out0': da3, 'out1': da4}, 
                   coords={'time': np.arange(20),
                           'xu': np.arange(48) * 5, 
                           'yu': np.arange(32) * 2})
    dataset = RawDataFromXrDataset(ds)
    dataset.index = 'time'
    dataset.add_input('in0')
    dataset.add_input('in1')
    dataset.add_output('out0')
    dataset.add_output('out1')
    
    loader = DataLoader(dataset, batch_size=7, drop_last=True)
    
    ds2 = ds.isel(yu=slice(0, 28), xu=slice(0, 37))
    dataset2 = RawDataFromXrDataset(ds2)
    dataset2.index = 'time'
    dataset2.add_input('in0')
    dataset2.add_input('in1')
    dataset2.add_output('out0')
    dataset2.add_output('out1')
    t = DatasetTransformer(ComposeTransforms(CropToMultipleof(5),
                                             PerChannelNormalizer()))
    t2 = deepcopy(t)
    train_dataset = Subset(dataset, np.arange(5))
    train_dataset2 = Subset(dataset2, np.arange(5))
    t.fit(train_dataset)
    t2.fit(train_dataset2)
    new_dataset = DatasetWithTransform(dataset, t)
    new_dataset2 = DatasetWithTransform(dataset2, t2)
    datasets = (new_dataset, new_dataset2)
    c = ConcatDataset_(datasets)
