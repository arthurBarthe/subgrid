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
from enum import Enum
from abc import ABC
import numpy as np


class VarianceMode(Enum):
    variance = 0
    precision = 1



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


class StudentLoss(_Loss):
    def __init__(self, nu: float = 30, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Temporary fix
        input, nu = input
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.lgamma((nu + 1) / 2)
        term2 = 1 / 2 * torch.log(nu) + torch.lgamma(nu / 2)
        term3 = - torch.log(precision)
        temp = (target - mean) * precision
        term4 = (nu + 1) / 2 * torch.log(1 + 1 / nu * temp**2) 
        return term1 + term2 + term3 + term4

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        input, nu = input
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    def predict_mean(self, input: torch.Tensor):
        input, nu = input
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))


class CauchyLoss(_Loss):
    def __init__(self, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        mean, scale = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.log(scale)
        term2 = torch.log((target - mean)**2 + scale**2)
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))


class HeteroskedasticGaussianLossV2(_Loss):
    """Class for Gaussian likelihood"""

    def __init__(self, n_target_channels: int = 1, bias: float = 0.,
                 mode=VarianceMode.precision):
        super().__init__()
        self.n_target_channels = n_target_channels
        self.bias = bias
        self.mode = mode

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    @property
    def channel_names(self):
        return ['S_x', 'S_y', 'S_xscale', 'S_yscale']

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        if self.mode is VarianceMode.precision:
            term1 = - torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias))**2 * precision**2
        elif self.mode is VarianceMode.variance:
            term1 = torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias))**2 / precision**2
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias


class HeteroskedasticGaussianLossV3(_Loss):
    """Loss to be used with transform2 from models/submodels.py"""

    def __init__(self, *args, **kargs):
        super().__init__()
        self.base_loss = HeteroskedasticGaussianLossV2(*args, **kargs)

    def __getattr__(self, name: str):
        try:
            # This is necessary as the class Module defines its own __getattr__
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_loss, name)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.base_loss.forward(input, target)

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        raw_loss = self._base_loss(input, target[:, :self.n_target_channels, ...])
        return raw_loss + torch.log(target[:, self.n_target_channels: self.n_target_channels + 1, ...])


class MultimodalLoss(_Loss):
    """General class for a multimodal loss. Each location on
    each channel can choose its mode independently."""

    def __init__(self, n_modes, n_target_channels, base_loss_cls,
                 base_loss_params=[], share_mode='C'):
        super().__init__()
        self.n_modes = n_modes
        self.n_target_channels = n_target_channels
        self.target_names = ['target' + str(i) for i in range(
            n_target_channels)]
        self.losses = []
        for i in range(n_modes):
            if i < len(base_loss_params):
                params = base_loss_params[i]
                self.losses.append(base_loss_cls(n_target_channels, **params))
            else:
                self.losses.append(base_loss_cls(n_target_channels))
        self.share_mode = share_mode

    @property
    def target_names(self):
        return self._target_names

    @target_names.setter
    def target_names(self, value):
        assert len(value) == self.n_target_channels
        self._target_names = value

    @property
    def n_required_channels(self):
        if self.share_mode == 'C':
            return sum(self.splits)

    @property
    def channel_names(self):
        """Automatically assigns names to output channels depending on the
        target names. For now not really implemented"""
        return [str(i) for i in range(self.n_required_channels)]

    @property
    def precision_indices(self):
        indices = []
        for i, loss in enumerate(self.losses):
            sub_indices = loss.precision_indices
            for j in range(len(sub_indices)):
                sub_indices[j] += self.n_modes * self.n_target_channels + i * loss.n_required_channels
            indices.extend(sub_indices)
        return indices

    @property
    def splits(self):
        """Return how to split the input to recover the different parts:
            - probabilities of the modes
            - quantities definining each mode
        """
        return ([self.n_modes, ] * self.n_target_channels 
                + [loss.n_required_channels for loss in self.losses])

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        splits = torch.split(input, self.splits, dim=1)
        probas, inputs = (splits[:self.n_target_channels],
                          splits[self.n_target_channels:])
        probas = [torch.softmax(proba, dim=1) for proba in probas]
        losses_values = []
        for i, (loss, input) in enumerate(zip(self.losses, inputs)):
            proba_i = torch.stack([proba[:, i, ...] for proba in probas], dim=1)
            loss_i = torch.log(proba_i) - loss.pointwise_likelihood(input, target)
            losses_values.append(loss_i)
        loss = torch.stack(losses_values, dim=2)
        final_loss = -torch.logsumexp(loss, dim=2)
        final_loss = final_loss.mean()
        return final_loss

    def predict(self, input: torch.Tensor):
        splits = torch.split(input, self.splits, dim=1)
        probas, inputs = (splits[:self.n_target_channels],
                          splits[self.n_target_channels:])
        probas = [torch.softmax(proba, dim=1) for proba in probas]
        predictions = [loss.predict(input) for loss, input in
                       zip(self.losses, inputs)]
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            proba_i = torch.stack([proba[:, i, ...] for proba in probas], dim=1)
            weighted_predictions.append(proba_i * pred)
        final_predictions = sum(weighted_predictions)
        return final_predictions

class BimodalGaussianLoss(MultimodalLoss):
    """Class for a bimodal Gaussian loss."""

    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class BimodalStudentLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels, base_loss_cls=StudentLoss)


class TrimodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(3, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class PentamodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(5, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)