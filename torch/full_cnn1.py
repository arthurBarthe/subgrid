# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:24:37 2020

@author: Arthur

TODOs:
-Add batch normalization, and a corresponding logging parameter.
-Apply a non-linear function such as tanh * scale on the last layer. Add
a corresponding logging parameter.
-Log as a parameter the number of layers.
-log as a parameter whether maxpooling was used (try to use a DictModule)
-log as a parameter whether you use groups=2 (again, DictModule?)
-Create a git repo
-log the source data
-log the preprocessing

-Try some standard image classification network whose last layer you'll change
-BUG: when we run less than 100 epochs the figures from previous runs are 
logged.

"""
from enum import Enum
import torch
from torch.utils.data import Dataset
from torch.nn import Module, Sequential, ModuleDict
from torch.nn import functional as F
import numpy as np
import os.path
import matplotlib.pyplot as plt
import mlflow


class MLFlowDataset(Dataset):
    """Abstract Wrapper class for a pytorch dataset that takes care of logging
    some artifacts through mlflow."""
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset), \
            'The passed dataset should be an instance \
             of torch.utils.data.Dataset'
        self.dataset = dataset
        self.first_get_done = False

    def __getitem__(self, index):
        features, targets = self.dataset[index]
        # On first call we log the extracted features and targets as artifacts
        if not self.first_get_done:
            self.dataset.plot_features(features)
            plt.savefig('/data/ag7531/figures/example_features.png')
            plt.close()
            self.dataset.plot_targets(targets)
            plt.savefig('/data/ag7531/figures/example_targets.png')
            mlflow.log_artifact('/data/ag7531/figures/example_features.png')
            mlflow.log_artifact('/data/ag7531/figures/example_targets.png')
            self.first_get_done = True
        return (features, targets)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr):
        try:
            return getattr(self.dataset, attr)
        except AttributeError as e:
            raise e


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

    def pre_process(self, train_index):
        """Operates some basic pre-processing on the data, such as
        rescaling."""
        self.width, self.height = self.psi.shape[-2:]
        # input: add channel dimension
        self.psi = self.psi.reshape(-1, 1, self.width, self.height)
        # input: remove mean, normalize
        self.mean_psi = np.mean(self.psi[:train_index], axis=0)
        self.std_psi = np.std(self.psi[:train_index], axis=0)
        self.psi = (self.psi - self.mean_psi) / self.std_psi
        # output: flatten
        self.sx = self.sx.reshape(self.sx.shape[0], -1)
        self.sy = self.sy.reshape(self.sy.shape[0], -1)
        mlflow.log_param('output_mean_removal', 'False')
        # output: divide by std
        self.std_sx = np.std(self.sx[:train_index], axis=0)
        self.std_sy = np.std(self.sy[:train_index], axis=0)
        self.std_targets = np.concatenate((self.std_sx, self.std_sy))
        self.sx /= self.std_sx
        self.sy /= self.std_sy

    def inv_pre_process(self, features=None, targets=None):
        """Inverse function of the pre-processing."""
        if features is not None:
            features = (features * self.std_psi) + self.mean_psi
            return features
        if targets is not None:
            targets = targets * self.std_targets
            return targets

    def set_indices(self, indices: list):
        """Sets the relative time indices that are used to make the prediction
        of the subgrid forcing.
        For instance if indices is [0, -1], the prediction at time t will use
        the psi field at times t and t-1."""
        for i in indices:
            if i > 0:
                raise ValueError('The indices should be 0 or negative')
        self.indices = np.array(indices)
        self.shift = max([abs(v) for v in self.indices])

    def __getitem__(self, index):
        """Returns the sample indexed by the passed index."""
        index = np.int64(index)
        shift = self.shift
        indices = index + shift + self.indices
        features = np.concatenate([self.psi[j] for j in indices])
        target = np.hstack((self.sx[index + shift], self.sy[index + shift]))
        return (features, target)   

    def __len__(self):
        """Returns the number of samples available in the dataset. Note that
        this might be less than the actual size of the first dimension
        if self.indices contains values other than 0, i.e. if we are
        using some data from the past to make predictions"""
        return self.data_length - self.shift

    @property
    def n_output_targets(self):
        """Returns the size of the prediction. This assumes square data."""
        return 2 * self.sx.shape[1]

    @property
    def data_length(self):
        return self.psi.shape[0]

    def plot_true_vs_pred(self, true: np.ndarray, predicted: np.ndarray):
        true = self.inv_pre_process(None, true)
        predicted = self.inv_pre_process(None, predicted)
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
        self.activation_choices = {'relu': torch.nn.ReLU(),
                                   'selu': torch.nn.SELU(),
                                   'tanh': torch.nn.Tanh(),
                                   '2tanh': ScaledModule(2, 
                                                            torch.nn.Tanh()),
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
        return len(self.layers) // 2

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


class FullyCNN(MLFlowNN):
    def __init__(self, input_depth: int, input_width: int, input_height: int,
                 output_size: int):
        super().__init__(input_depth, input_width, input_height,
                         output_size)
        self.build()

    def build(self):
        self.add_conv2d_layer(self.input_depth, 64, 5, padding=2)
        self.add_activation('relu')
        self.add_conv2d_layer(64, 64, 5, padding=2)
        self.add_activation('relu')
        self.add_conv2d_layer(64, 64, 5, padding=2)
        self.add_activation('relu')
        self.add_conv2d_layer(64, 64, 5, padding=2)
        self.add_activation('relu')
        self.add_conv2d_layer(64, 64, 5, padding=2)
        self.add_activation('relu')
        self.add_conv2d_layer(64, 2, 3, padding=1)
        self.add_final_activation('identity')


if __name__ == '__main__':
    dataset = Dataset_psi_s(r'D:\Data sets\NYU\processed_data',
                            'psi_coarse.npy', 'sx_coarse.npy', 'sy_coarse.npy')
