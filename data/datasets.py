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

    def add_features_transform(self, transform):
        feature_t = self.transforms['features']
        if not isinstance(feature_t, ComposeTransforms):
            self.transforms['features'] = ComposeTransforms([feature_t, ])
        self.transforms['features'].add_transform(transform)

    def add_targets_transform(self, transform):
        target_t = self.transforms['targets']
        if not isinstance(target_t, ComposeTransforms):
            self.transforms['targets'] = ComposeTransforms([target_t, ])
        self.transforms['targets'].add_transform(transform)

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

    def get_features_coords(self, coords):
        result = dict()
        for k, v in coords.items():
            result[k] = self.transforms['features'].transform_coordinate(v, k)
        return result

    def get_targets_coords(self, coords):
        result = dict()
        for k, v in coords.items():
            result[k] = self.transforms['targets'].transform_coordinate(v, k)
        return result

    def __call__(self, x):
        return self.transform(x)

    def inverse_transform(self, x: Dataset):
        features, targets = x
        new_features = self.transforms['features'].inverse_transform(features)
        new_targets = self.transforms['targets'].inverse_transform(targets)
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
        self.transforms = list(transforms)

    def fit(self, x):
        for transform in self.transforms:
            transform.fit(x)
            y = []
            for i in range(x.shape[0]):
                y.append(transform(x[i, ...]))
            x = np.array(y)

    def transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def transform_coordinate(self, coord, dim):
        for transform in self.transforms:
            if isinstance(transform, CropToNewShape):
                x = transform.transform_coordinate(coord, dim)
        return x

    def add_transform(self, transform):
        self.transforms.append(transform)


class Randommult(ArrayTransform):
    def __init__(self):
        self.min = 0.1
        self.max = 2
        self.i = 0

    def fit(self, x):
        size = x.shape[0]
        self._multipliers = np.random.rand(size) * (self.max - self.min)
        self._multipliers += self.min

    def transform(self, x):
        x = x * self._multipliers[self.i]
        self.i += 1
        return x


# class CropToMultipleof(ArrayTransform):
#     def __init__(self, multiple_of: int = 2):
#         self.multiple_of = multiple_of

#     def fit(self, x):
#         shape = x.shape
#         new_shape_1 = shape[2] // self.multiple_of * self.multiple_of
#         new_shape_2 = shape[3] // self.multiple_of * self.multiple_of
#         self.new_shape = (shape[1], new_shape_1, new_shape_2)
#         print('debugging:', self.new_shape)

#     def transform(self, x):
#         return x[:, :self.new_shape[1], :self.new_shape[2]]


class CropToNewShape(ArrayTransform):
    """Crops to a new shape. Keeps the array centered, modulo 1 in which case
    the top left is priviledged. If the passed data is smaller than the
    requested shape, the data is unchanged."""

    def __init__(self, height=None, width=None):
        self.height = height
        self.width = width

    def fit(self, x):
        pass

    def get_slice(self, length: int, length_to: int):
        d_left = max(0, (length - length_to) // 2)
        d_right = d_left + max(0, (length - length_to)) % 2
        return slice(d_left, length - d_right)

    def transform(self, x):
        height, width = x.shape[1:]
        return x[:, self.get_slice(height, self.height),
                 self.get_slice(width, self.width)]

    def transform_coordinate(self, coords, dim):
        length = len(coords)
        if dim == 'height':
            return coords[self.get_slice(length, self.height)]
        if dim == 'width':
            return coords[self.get_slice(length, self.width)]

    def __repr__(self):
        return f'CropToNewShape({self.height}, {self.width})'


class CropToMultipleof(CropToNewShape):
    def __init__(self, multiple_of: int = 2):
        super().__init__()
        self.multiple_of = multiple_of

    def fit(self, x):
        shape = x.shape
        self.height = shape[2] // self.multiple_of * self.multiple_of
        self.width = shape[3] // self.multiple_of * self.multiple_of

    def __repr__(self):
        return f'CropToMultipleOf({self.multiple_of})'


class SignedSqrt(ArrayTransform):
    def fit(self, x):
        pass

    def transform(self, x):
        x = np.sign(x) * np.sqrt(np.abs(x))
        return x


class PerChannelNormalizer(ArrayTransform):
    def __init__(self, use_mean=False, fit_only_once=False):
        self.fit_only_once = fit_only_once
        self._std = None
        self._mean = None
        self._use_mean = use_mean

    def fit(self, X: np.ndarray):
        if (not self.fit_only_once) or (self._mean is None):
            mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            std = np.std(X, axis=(0, 2, 3), keepdims=True)
            self._mean = mean.reshape(mean.shape[1:])
            self._std = std.reshape(std.shape[1:])

    def transform(self, x: np.ndarray):
        assert(self._mean is not None)
        if self._use_mean:
            x = x - self._mean
        return x / self._std

    def inverse_transform(self, X):
        return X * self._std + self._mean


class FixedNormalizer(ArrayTransform):
    def fit(self, x):
        pass

    def transform(self, x):
        return x / self.std


class FixedVelocityNormalizer(FixedNormalizer):
    std = 0.1


class FixedForcingNormalizer(FixedNormalizer):
    std = 1e-7


class ArctanPerChannelNormalizer(PerChannelNormalizer):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def transform(self, x):
        assert(self._mean is not None)
        if self._use_mean:
            x = x - self._mean
        return np.arctan(x / self._std)


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


class RawDataFromXrDataset(Dataset):
    """This class allows to define a Pytorch Dataset based on an xarray 
    dataset easily, specifying features and targets."""
    def __init__(self, dataset: xr.Dataset):
        self.xr_dataset = dataset
        self._input_arrays = list()
        self._output_arrays = list()
        self._index = None

    @property
    def output_coords(self):
        return dict(self.xr_dataset.coords)

    @property
    def input_coords(self):
        return dict(self.xr_dataset.coords)

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

    def __getattr__(self, attr_name):
        if hasattr(self.xr_dataset, attr_name):
            return getattr(self.xr_dataset, attr_name)
        else:
            raise AttributeError()


class DatasetWithTransform:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    @property
    def output_coords(self):
        coords = {'height': self.coords['yu'], 'width': self.coords['xu']}
        new_coords = self.transform.get_targets_coords(coords)
        return {'yu': new_coords['height'], 'xu': new_coords['width'],
                'time': self.coords['time']}

    @property
    def input_coords(self):
        coords = {'height': self.coords['yu'], 'width': self.coords['xu']}
        new_coords = self.transform.get_features_coords(coords)
        return {'yu': new_coords['height'], 'xu': new_coords['width'],
                'time': self.coords['time']}

    @property
    def height(self):
        """Since the transform can modify the height..."""
        x = self[0][0]
        return x.shape[1]

    @property
    def width(self):
        x = self[0][0]
        return x.shape[2]

    @property
    def output_height(self):
        return self[0][1].shape[1]

    @property
    def output_width(self):
        return self[0][1].shape[2]

    def __getitem__(self, index: int):
        return self.transform(self.dataset[index])

    def __getattr__(self, attr):
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError()

    def __len__(self):
        return len(self.dataset)

    def add_transforms_from_model(self, model):
        features_transforms = self.add_features_transform_from_model(model)
        targets_transforms = self.add_targets_transform_from_model(model)
        return {'features transform': features_transforms,
                'targets_transform': targets_transforms}

    def add_features_transform_from_model(self, model):
        """Automatically adds features transform required by the model.
        For instance Unet will require features to be reshaped to multiples
        of 2, 4 or higher."""
        if hasattr(model, 'get_features_transform'):
            transform = model.get_features_transform()
            self.add_features_transform(transform)
            return transform
        else:
            return None

    def add_targets_transform_from_model(self, model):
        """Automatically reshapes the targets of the dataset to match
        the shape of the output of the model."""
        output_height = model.output_height(self.height, self.width)
        output_width = model.output_width(self.height, self.width)
        transform = CropToNewShape(output_height, output_width)
        self.add_targets_transform(transform)
        return transform

    def inverse_transform(self, x):
        return self.transform.inverse_transform(x)

    def add_features_transform(self, transform):
        self.transform.add_features_transform(transform)

    def add_targets_transform(self, transform):
        self.transform.add_targets_transform(transform)


class Subset_(Subset):
    """Extends the Pytorch Subset class to allow for attributes of the 
    dataset to be propagated to the subset dataset"""
    def __init__(self, dataset, indices):
        super(Subset_, self).__init__(dataset, indices)

    @property
    def output_coords(self):
        new_coords = deepcopy(self.dataset.output_coords)
        new_coords['time'] = new_coords['time'][self.indices]
        return new_coords

    @property
    def intput_coords(self):
        new_coords = self.dataset.intput_coords
        new_coords['time'] = new_coords['time'][self.indices]
        return new_coords

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
        for dataset in self.datasets:
            crop_transform = CropToNewShape(self.height, self.width)
            dataset.add_features_transform(crop_transform)
            dataset.add_targets_transform(crop_transform)

    def __getattr__(self, attr):
        print('Trying ', attr)
        if hasattr(self.datasets[0], attr):
            return getattr(self.datasets[0], attr)
        else:
            raise AttributeError()


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
        else:
            self.time_indices = [0,]

    @property
    def time_indices(self):
        if self._time_indices:
            return np.array(self._time_indices)

    @time_indices.setter
    def time_indices(self, indices: list):
        for i in indices:
            if i > 0:
                raise ValueError('The indices should be 0 or negative')
        self._time_indices = indices
        self._shift = max([abs(v) for v in indices])

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value: int):
        raise Exception('The shift cannot be set manually. Instead set \
                        the time indices.')

    @property
    def n_features(self):
        return self.dataset.n_features * len(self.time_indices)

    def __getitem__(self, index):
        """Returns the sample indexed by the passed index."""
        # TODO check this does not slows things down. Hopefully should not,
        # as it should just be a memory view.
        indices = index + self.shift + self.time_indices
        features = [self.dataset[i][0] for i in indices]
        feature = np.concatenate(features)
        target = self.dataset[index + self.shift][1]
        return (feature, target)

    def __len__(self):
        """Returns the number of samples available in the dataset. Note that
        this might be less than the actual size of the first dimension
        if self.indices contains values other than 0, i.e. if we are
        using some data from the past to make predictions"""
        return len(self.dataset) - self.shift

    def __getattr__(self, attr_name):
        if hasattr(self.dataset, attr_name):
            return getattr(self.dataset, attr_name)
        else:
            raise AttributeError()


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
                                             SignedSqrt(),
                                             PerChannelNormalizer()))
    t.add_features_transform(CropToMultipleof(3))
    t2 = deepcopy(t)
    train_dataset = Subset_(dataset, np.arange(5))
    train_dataset2 = Subset_(dataset2, np.arange(5))
    t.fit(train_dataset)
    t2.fit(train_dataset2)
    new_dataset = DatasetWithTransform(dataset, t)
    new_dataset2 = DatasetWithTransform(dataset2, t2)
    datasets = (new_dataset, new_dataset2)
    c = ConcatDataset_(datasets)
