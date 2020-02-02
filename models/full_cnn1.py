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

import torch
from torch.nn import Module
from torch.nn import functional as F
from data.datasets import RawData, MultipleTimeIndices
import numpy as np
import mlflow


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


#class ModuleFactory:
#    def __init__(self):
#        pass
#
#    def build(self, *args, **kargs):
#        pass
#
#
#class FullCNNfactory(ModuleFactory):
#    def __init__(self):
#        super().__init__()
#
#    def build(self, n_layers: int, activation, )


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
