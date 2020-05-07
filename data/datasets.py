# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:38:40 2020

@author: Arthur
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os.path
import matplotlib.pyplot as plt
import mlflow
# from sklearn.preprocessing import StandardScaler
import xarray as xr
import logging


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


class LoggedTransformer:
    """Class that wrapps an sklearn transformer and logs some info about it"""
    def __init__(self, transformer, data_name: str):
        self.transformer = transformer
        self.name = transformer.__class__.__name__ + '_' + data_name

    def fit(self, X: np.ndarray):
        """Fit method. Note that we allow for more general shapes of data
        compared to standard sklearn transformers."""
        if X.ndim > 2:
            X = X.reshape((-1, prod(X.shape[1:])))
        return self.transformer.fit(X)

    def transform(self, X: np.ndarray):
        """Transform method. Logs some info through mlflow."""
        initial_shape = X.shape
        if X.ndim > 2:
            X = X.reshape((-1, prod(X.shape[1:])))
        mlflow.log_param(self.name, 'True')
        return self.transformer.transform(X).reshape(initial_shape)

    def inverse_transform(self, X: np.ndarray):
        initial_shape = X.shape
        if X.ndim > 2:
            X = X.reshape((-1, prod(X.shape[1:])))
        mlflow.log_param(self.name, 'True')
        return self.transformer.inverse_transform(X).reshape(initial_shape)

    def __getattr__(self, attr):
        try:
            return getattr(self.transformer, attr)
        except AttributeError as e:
            raise e(f'Attribute {attr} not found')


class DatasetTransformer:
    """Wrapper class to apply to sklearn-type data transformer to a pytorch
    Dataset, i.e. the transformation, by default, is applied to both the
    features and targets, independently though."""
    def __init__(self, transformer_class: type, apply_both: bool = True):
        # TODO apply_both
        self.transformer_class = transformer_class
        self.apply_both = apply_both
        self.transformers = {'features': None, 'targets': None}
        # TODO add the possibility to include transfomer params
        features_transformer = LoggedTransformer(transformer_class(),
                                                 'features')
        targets_transformer = LoggedTransformer(transformer_class(),
                                                'targets')
        self.transformers['features'] = features_transformer
        self.transformers['targets'] = targets_transformer

    @property
    def targets_transformer(self):
        return self.transformers['targets']

    def fit(self, X: Dataset):
        features, targets = X[:]
        self.transformers['features'].fit(features)
        self.transformers['targets'].fit(targets)
        return self

    def transform(self, X: Dataset):
        features, targets = X[:]
        new_features = self.transformers['features'].transform(features)
        new_targets = self.transformers['targets'].transform(targets)
        return FeaturesTargetsDataset(new_features, new_targets)

    def fit_transform(self, X: Dataset):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Dataset):
        features, targets = X[:]
        new_features = self.transformers['features'].inverse_transform(features)
        new_targets = self.transformers['targets'].transform(targets)
        return FeaturesTargetsDataset(new_features, new_targets)


class UniformScaler:
    def __init__(self):
        self._std = None

    @property
    def std(self):
        return self._std

    def fit(self, X: np.ndarray):
        self._std = np.std(X)

    def transform(self, X: np.ndarray):
        assert(self._std is not None)
        return X / self.std

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)


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


class MLFLowPreprocessing(Dataset):
    """Makes some basic processing and automatically logs it through MLFlow."""
    def __init__(self, raw_data: Dataset):
        self.new_features = raw_data[:][0]
        self.new_targets = raw_data[:][1]
        self.means = dict()
        self.stds = dict()

    @staticmethod
    def _normalize(data: np.ndarray, clip_min: float = 0,
                   clip_max: float = np.inf) -> np.ndarray:
        std = np.clip(np.std(data), clip_min, clip_max)
        return data / std, std

    def substract_mean_features(self):
        # TODO add something to ensure this is called before the normalization
        self.new_features = self.new_features - np.mean(self.new_features, 0)

    def substract_mean_targets(self):
        self.new_targets = self.new_targets - np.mean(self.new_targets, 0)

    @call_only_once
    def normalize_features(self, clip_min: float = 1e-9,
                           clip_max: float = np.inf):
        """Normalizes the data, by dividing each componenent by its clipped
        std, where the clipping parameters are passed to the function.
        Note that it is particularly important to use a clip_min for components
        that are close to constant."""
        features = self._normalize(self.new_features, clip_min, clip_max)
        self.new_features = features
        mlflow.log_param('clip_features', str((clip_min, clip_max)))

    @call_only_once
    def normalize_targets(self, clip_min: float = 1e-9,
                          clip_max: float = np.inf):
        targets = self._normalize(self.new_targets, clip_min, clip_max)
        self.new_targets = targets
        mlflow.log_param('clip_targets', str((clip_min, clip_max)))

    def __getitem__(self, index: int):
        return (self.new_features[index], self.new_targets[0])


class RawDataFromXrDataset(Dataset):
    """This class allows to define a Pytorch Dataset based on an xarray 
    dataset easily."""
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

    @property
    def input_arrays(self):
        return self._input_arrays

    def add_output(self, varname):
        self._check_varname(varname)
        self._output_arrays.append(varname)

    def add_input(self, varname: str):
        self._check_varname(varname)
        self._input_arrays.append(varname)

    def __getitem__(self, index):
        try:
            features = self.xr_dataset[self.input_arrays].isel({self._index: index})
            features = features.to_array().data
            targets = self.xr_dataset[self.output_arrays].isel({self._index: index})
            targets = targets.to_array().data
        except KeyError as e:
            e.msg = e.msg + '\n Make sure you have defined the index, inputs,\
                and outputs.'
            raise e
        return features, targets

    def n_output_targets(self):
        logging.warning('Depreciated call to \
                        RawDataFromXrDataset.n_output_targets(). \
                        To be removed.')
        t = self[0][1]
        return t.shape[0]
    
    @property
    def width(self):
        return len(self.xr_dataset['xu_ocean'])

    @property
    def height(self):
        return len(self.xr_dataset['yu_ocean'])

    def __len__(self):
        try:
            return len(self.xr_dataset[self._index])
        except KeyError as e:
            e.msg = e.msg + '\n Make sure you have defined the index.'
            raise e

    def _check_varname(self, var_name: str):
        if var_name not in self.xr_dataset:
            raise KeyError('Variable not in the xarray dataset.')
        if var_name in self._input_arrays or var_name in self._output_arrays:
            raise ValueError('Variable already added as input or output.')


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
    da = DataArray(data=np.zeros((20, 3, 4)), dims=('time', 'y', 'x'))
    da2 = DataArray(data=np.ones((20, 3, 4)), dims=('time', 'y', 'x'))
    da3 = DataArray(data=np.ones((20, 3, 4)) * 10, dims=('time', 'y', 'x'))
    da4 = DataArray(data=np.ones((20, 3, 4)) * 20, dims=('time', 'y', 'x'))
    ds = xrDataset({'in0': da, 'in1': da2,
                    'out0': da3, 'out1': da4}, 
                   coords={'time': np.arange(20),
                           'x': [0, 5, 10, 15], 
                           'y': [0, 5, 10]})
    dataset = RawDataFromXrDataset(ds)
    dataset.index = 'time'
    dataset.add_input('in0')
    dataset.add_input('in1')
    dataset.add_output('out0')
    dataset.add_output('out1')
    
    loader = DataLoader(dataset, batch_size=3, drop_last=True)