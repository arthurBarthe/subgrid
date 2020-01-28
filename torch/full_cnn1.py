# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:24:37 2020

@author: Arthur

TODOs:
-Try some standard image classification network whose last layer you'll change
- change the color map of plots
- study different values of time indices
------BUGS-----
-when we run less than 100 epochs the figures from previous runs are
logged.
"""
# TODO Log the data run that is used to create the dataset. Log any
# transformation applied to the data. Later we might want to allow from
# stream datasets.

from enum import Enum
import torch
from torch.utils.data import Dataset
from torch.nn import Module, Sequential, ModuleDict
from torch.nn import functional as F
import numpy as np
import os.path
import matplotlib.pyplot as plt
import mlflow


def call_only_once(f):
    """Decorator that ensures a function is only called at most once."""
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
        self.apply_both = True
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

    def inverse_transform(self, X: Dataset):
        features, targets = X[:]
        new_features = self.transformers['features'].inverse_transform(features)
        new_targets = self.transformers['targets'].transform(targets)
        return FeaturesTargetsDataset(new_features, new_targets)


#class MLFlowDataset(Dataset):
#    """Abstract Wrapper class for a pytorch dataset that takes care of logging
#    some artifacts through mlflow."""
#    def __init__(self, dataset):
#        assert isinstance(dataset, Dataset), \
#            'The passed dataset should be an instance \
#             of torch.utils.data.Dataset'
#        self.dataset = dataset
#        self.first_get_done = False
#
#    def __getitem__(self, index):
#        features, targets = self.dataset[index]
#        # On first call we log the extracted features and targets as artifacts
#        if not self.first_get_done:
#            self.dataset.plot_features(features)
#            plt.savefig('/data/ag7531/figures/example_features.png')
#            plt.close()
#            self.dataset.plot_targets(targets)
#            plt.savefig('/data/ag7531/figures/example_targets.png')
#            mlflow.log_artifact('/data/ag7531/figures/example_features.png')
#            mlflow.log_artifact('/data/ag7531/figures/example_targets.png')
#            self.first_get_done = True
#        return (features, targets)
#
#    def __len__(self):
#        return len(self.dataset)
#
#    def __getattr__(self, attr):
#        try:
#            return getattr(self.dataset, attr)
#        except AttributeError as e:
#            raise e


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


class RawData(Dataset):
    # TODO This should be made an abstract class.
    """This class produces a raw dataset by doing the basic transformations,
    using the psi, s_x and s_y data."""
    def __init__(self, data_location: str, f_name_psi: str, f_name_sx: str,
                 f_name_sy: str):
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


class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input


class ScaledModule(Module):
    def __init__(self, factor: float, module: torch.nn.Module):
        super().__init__()
        self.factor = factor
        self.module = module

    def forward(self, input: torch.Tensor):
        return self.factor * self.module.forward(input)


class MLFlowNN(Module):
    """Abstract class for a pytorch NN whose characteristics are automatically
    logged through MLFLOW."""
    def __init__(self, input_depth: int, input_width: int, input_height: int,
                 output_size: int):
        super().__init__()
        self.input_depth = input_depth
        self.input_width = input_width
        self.input_height = input_height
        self.output_size = output_size
        self.layers = torch.nn.ModuleList()
        self._n_layers = 0
        self.activation_choices = {'relu': torch.nn.ReLU(),
                                   'selu': torch.nn.SELU(),
                                   'tanh': torch.nn.Tanh(),
                                   '2tanh': ScaledModule(2, torch.nn.Tanh()),
                                   'identity': Identity()}
        self.activations = []
        self.params_to_log = {'max_pool': False,
                              'max_kernel_size': 0,
                              'max_depth': 1,
                              'groups': False,
                              'batch_normalization': False
                              }
        self.logged_params = False

    @property
    def n_layers(self) -> int:
        """Returns the number of layers. Note that we consider that
        activations are not layers in this count, but are part of a layer,
        hence the division by two."""
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self._n_layers = value

    def add_activation(self, activation: str) -> None:
        self.layers.append(self.activation_choices[activation])
        self.params_to_log['default_activation'] = activation
        self.activations.append(activation)

    def add_final_activation(self, activation: str) -> None:
        """Use this funtion to specify the final activation. This is
        required to log a specific parameter through mlflow corresponding
        to this activation, as it plays a specific role."""
        self.layers.append(self.activation_choices[activation])
        self.params_to_log['last_layer_activation'] = activation

    def add_conv2d_layer(self, in_channels: int, out_channels: int,
                         kernel_size: int, stride: int = 1, padding: int = 0,
                         dilation: int = 1, groups: int = 1, bias: int = True):
        """Adds a convolutional layer. Same parameters as the torch Conv2d,
        the difference is that we log some of these parameters through mlflow.
        """
        conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        self.layers.append(conv_layer)
        i_layer = self.n_layers
        self.params_to_log['layer{}'.format(i_layer)] = 'Conv2d'
        self.params_to_log['kernel{}'.format(i_layer)] = str(kernel_size)
        self.params_to_log['groups{}'.format(i_layer)] = groups
        self.params_to_log['depth{}'.format(i_layer)] = out_channels
        if groups > 1:
            self.params_to_log['groups'] = True
        if kernel_size > self.params_to_log['max_kernel_size']:
            self.params_to_log['max_kernel_size'] = kernel_size
        if out_channels > self.params_to_log['max_depth']:
            self.params_to_log['max_depth'] = out_channels
        # Register that we have added a layer
        self.n_layers += 1

    def add_divergence2d_layer(self, n_input_channels: int,
                               n_output_channels: int):
        div2d_layer = Divergence2d(n_input_channels, n_output_channels)
        self.layers.append(div2d_layer)
        self.params_to_log['divergence2d'] = True
        # Register we have added a layer
        self.n_layers += 1

    def add_max_pool_layer(self, kernel_size: int, stride=None,
                           padding: int = 0, dilation: int = 1):
        """Adds a max pool layer and logs some parameters corresponding to
        this layer."""
        layer = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation)
        self.layers.append(layer)
        i_layer = self.n_layers
        self.params_to_log['layer{}'.format(i_layer)] = 'MaxPoolv2d'
        self.params_to_log['kernel{}'.format(i_layer)] = str(kernel_size)
        self.params_to_log['max_pool'] = True

    def add_batch_norm_layer(self, num_features: int, eps: float = 1e-5,
                             momentum: float = 0.1, affine: bool = True,
                             track_running_stats: bool = True):
        """Adds a batch normalization layer and makes some logs accordingly"""
        layer = torch.nn.BatchNorm2d(num_features, eps, momentum, affine,
                                     track_running_stats)
        self.layers.append(layer)
        self.params_to_log['batch_normalization'] = True

    def log_params(self):
        """Logs the parameters for the built neural net."""
        print('Logging neural net parameters...')
        mlflow.log_param('n_layers', self.n_layers)
        for param_name, param_value in self.params_to_log.items():
            mlflow.log_param(param_name, str(param_value))
        self.logged_params = True

    def forward(self, input: torch.Tensor):
        """Overwrites the abstract method of the Module class."""
        # Log the params if it has not been done already.
        if not self.logged_params:
            self.log_params()
        # Go through the layers
        output = input
        for i_layer,  layer in enumerate(self.layers):
            output = layer(output)
        return output.view(-1, self.output_size)

class Divergence2d(Module):
    """Class that defines a fixed layer that produces the divergence of the
    input field."""
    def __init__(self, n_input_channels: int, n_output_channels: int):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.x_derivative = torch.tensor([[[[0, 0, 0],
                                          [-1, 0, 1],
                                          [0, 0, 0]]]])
        self.x_derivative = self.x_derivative.expand(1, n_input_channels // 4,
                                                     -1, -1)
        self.y_derivative = torch.tensor([[0, 1, 0],
                                          [0, 0, 0],
                                          [0, -1, 0]])
        self.y_derivative = self.y_derivative.expand(1, n_input_channels // 4,
                                                     -1, -1)
        self.x_derivative = self.x_derivative.to(dtype=torch.float32,
                                                 device=device)
        self.y_derivative = self.y_derivative.to(dtype=torch.float32,
                                                 device=device)

    def forward(self, input: torch.Tensor):
        n, c, h, w = input.size()
        output1 = F.conv2d(input[:, :c//4, :, :], self.x_derivative,
                           padding=1)
        output1 += F.conv2d(input[:, c//4:c//2, :, :], self.y_derivative,
                            padding=1)
        output2 = F.conv2d(input[:, c//2:c//2+c//4, :, :], self.x_derivative,
                           padding=1)
        output2 += F.conv2d(input[:, c//2+c//4:, :, :], self.y_derivative,
                            padding=1)
        return torch.stack((output1, output2), dim=1)


class FullyCNN(MLFlowNN):
    def __init__(self, input_depth: int, input_width: int, input_height: int,
                 output_size: int):
        super().__init__(input_depth, input_width, input_height,
                         output_size)
        self.build()

    def build(self):
        self.add_conv2d_layer(self.input_depth, 128, 5, padding=2)
        self.add_activation('relu')
        self.add_batch_norm_layer(128)
        self.add_conv2d_layer(128, 64, 5, padding=2)
        self.add_activation('relu')
        self.add_batch_norm_layer(64)
        self.add_conv2d_layer(64, 32, 5, padding=2)
        self.add_activation('relu')
        self.add_batch_norm_layer(32)
        self.add_conv2d_layer(32, 32, 5, padding=2)
        self.add_activation('relu')
        self.add_batch_norm_layer(32)
#        self.add_conv2d_layer(32, 64, 5, padding=2)
#        self.add_activation('relu')
#        self.add_batch_norm_layer(64)
        self.add_divergence2d_layer(32, 2)
        self.add_final_activation('identity')


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, Subset
    batch_size = 8

    dataset = RawData(r'D:\Data sets\NYU\processed_data',
                      'psi_coarse.npy', 'sx_coarse.npy', 'sy_coarse.npy')
    # Split train/test
    train_split = 0.4
    test_split = 0.75
    n_indices = len(dataset)
    train_index = int(train_split * n_indices)
    test_index = int(test_split * n_indices)
    train_dataset = Subset(dataset, np.arange(train_index))
    test_dataset = Subset(dataset, np.arange(test_index, n_indices))

    # Apply basic normalization transforms (using the training data only)
#    s = DatasetTransformer(StandardScaler)
#    s.fit(train_dataset)
#    train_dataset = s.transform(train_dataset)
#    test_dataset = s.transform(test_dataset)

    # Specifies which time indices to use for the prediction
    indices = [0, -2, -4, -6]
    train_dataset = MultipleTimeIndices(train_dataset)
    train_dataset.time_indices = indices
    test_dataset = MultipleTimeIndices(test_dataset)
    test_dataset.time_indices = indices

    # Dataloaders are responsible for sending batches of data to the NN
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False)
