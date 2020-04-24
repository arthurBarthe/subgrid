#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:08:26 2020

@author: arthur
In this module we define custom loss functions
"""
import torch
from torch.nn.modules.loss import _Loss


class HeteroskedasticGaussianLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(HeteroskedasticGaussianLoss, self).__init__(size_average,
                                                          reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(target, 2, dim=1)
        precision_ = torch.log(1 + torch.exp(precision)) + 0.1 
        # precision = precision**2 + 1e-3
        if not torch.all(precision_ > 0):
            raise ValueError('Got a non-positive precision value. \
                             Pre-processed precision tensor was: \
                                 {}'.format(precision))
        term1 = -torch.log(precision_)
        term2 = 1 / 2 * (input - mean)**2 * precision_**2
        return (term1 + term2).mean()
