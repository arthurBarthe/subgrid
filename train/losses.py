#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:08:26 2020

@author: arthur
In this module we define custom loss functions. In particular we define
a loss function based on the Gaussian likelihood with two parameters, 
mean and precision.
"""
import torch
from torch.nn.modules.loss import _Loss

# DEPRECIATED
class HeteroskedasticGaussianLoss(_Loss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, 2, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        term1 = - 1 / 2 * torch.log(precision)
        term2 = 1 / 2 * (target - mean)**2 * precision
        return (term1 + term2).mean()


class HeteroskedasticGaussianLossV2(_Loss):
    """Class for Gaussian likelihood"""

    def __init__(self, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    @property
    def precision_indices(self):
        return list(range(self.n_required_channels // 2,
                          self.n_required_channels))

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        n_channels = target.size(1)
        mean, precision = torch.split(input, n_channels, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        term1 = - torch.log(precision)
        term2 = 1 / 2 * (target - mean)**2 * precision**2
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        lkhs = self.pointwise_likelihood(input, target)
        return lkhs.mean()


class MultimodalLoss(_Loss):
    """General class for a multimodal loss. By default, each location on
    each channel can choose its mode independently. If share_mode is set to
    'C' for channels, the mode is shared accross channels"""

    def __init__(self, n_modes, n_target_channels, base_loss_cls,
                 share_mode='C', *base_loss_params):
        super().__init__()
        self.n_modes = n_modes
        self.n_target_channels = n_target_channels
        self.losses = [base_loss_cls(n_target_channels, *base_loss_params)
                       for i in range(n_modes)]
        self.share_mode = share_mode

    @property
    def n_required_channels(self):
        if self.share_mode == 'C':
            return sum(self.splits)

    @property
    def precision_indices(self):
        indices = []
        for i, loss in enumerate(self.losses):
            sub_indices = loss.precision_indices
            for j, index in enumerate(sub_indices):
                sub_indices[j] = index + 2 + i * loss.n_required_channels
            indices.extend(sub_indices)
        return indices

    @property
    def splits(self):
        """Return how to split the input to recover the different parts:
            - probabilities of the modes
            - quantities definining each mode
        """
        return [self.n_modes, ] + [loss.n_required_channels for
                                   loss in self.losses]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = torch.split(input, self.splits, dim=1)
        probas, inputs = input[0], input[1:]
        probas = torch.softmax(probas, dim=1)
        losses = [-torch.log(proba) + loss.pointwise_likelihood(input, target)
                  for (proba, loss, input) in
                  zip(probas, self.losses, inputs)]
        final_loss = torch.stack(losses)
        return final_loss.mean()


class BimodalGaussianLoss(MultimodalLoss):
    """Class for a bimodal Gaussian loss. For one target channel, the input
    should have 6 channels: 
        - 2 channels for the probabilities of each mode
        - 2 channels for the mean and precision of the first mode
        - 2 channels for the mean and precision of the second mode
    For two target channels, the input should have 10 channels:
        - 2 channels for the probability of each mode
        - 4 channels for the mean (2 C) and precision (2 C) of the first mode 
        - 4 channels for the mean and precision of the second mode"""

    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class TrimodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(3, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)