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
    def __init__(self, nu: float = 3, n_target_channels: int = 1):
        super().__init__()
        self.nu = nu
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.log(scale)
        temp = (target - mean)
        term2 = (self.nu + 1) / 2 * torch.log(scale**2 + 1 / self.nu * temp**2) 
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
    """General class for a multimodal loss. By default, each location on
    each channel can choose its mode independently. If share_mode is set to
    'C' for channels, the mode is shared accross channels"""

    def __init__(self, n_modes, n_target_channels, base_loss_cls,
                 base_loss_params=[], share_mode='C'):
        super().__init__()
        self.n_modes = n_modes
        self.n_target_channels = n_target_channels
        self.losses = []
        for i in range(n_modes):
            if i < len(base_loss_params):
                params = base_loss_params[i]
                self.losses.append(base_loss_cls(n_target_channels, **params))
            else:
                self.losses.append(base_loss_cls(n_target_channels))
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
                sub_indices[j] = index + self.n_modes + i * loss.n_required_channels
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
        # Favour first one
        probas[:, 0, ...] += 2
        probas = torch.softmax(probas, dim=1)
        probas = torch.split(probas, 1, dim=1)
        losses = [torch.log(proba) - loss.pointwise_likelihood(input, target)
                  for (proba, loss, input) in zip(probas, self.losses, inputs)]
        loss = torch.stack(losses, dim=2)
        final_loss = -torch.logsumexp(loss, dim=2)
        final_loss = final_loss.mean()
        return final_loss

    def predict(self, input: torch.Tensor):
        input = torch.split(input, self.splits, dim=1)
        probas, inputs = input[0], input[1:]
        # Favour first one
        probas[:, 0, ...] += 2
        probas = torch.softmax(probas, dim=1)
        predictions = [loss.predict(input) for loss, input in
                       zip(self.losses, inputs)]
        n_channels = predictions[0].size(1)
        predictions = torch.stack(predictions, dim=2)
        sel = torch.argmax(probas, dim=1, keepdim=True)
        sel = sel.unsqueeze(dim=2)
        sel = sel.repeat((1, n_channels, 1, 1, 1))
        final_predictions = torch.gather(predictions, 2, sel)
        final_predictions = final_predictions.squeeze(2)
        return final_predictions


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


class TrimodalGaussianLossV2(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(3, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2,
                         base_loss_params=[dict(bias=0.), dict(bias=0.)])


class PentamodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(5, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)