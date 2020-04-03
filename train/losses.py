#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:08:26 2020

@author: arthur
In this module we define custom loss functions
"""
import torch
from torch.nn.modules.loss import _Loss
from torch.distributions.normal import Normal

class HeteroskedasticGaussianLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(HeteroskedasticGaussianLoss, self).__init__(size_average, 
                                                          reduce, 
                                                          reduction)

    def forward(self, input : torch.Tensor, target : torch.Tensor):
        # Split the target into mean (first two channels) and scale
        mean, scale = torch.split(target, 2, dim=1)
        scale = torch.log(1 + torch.exp(scale))
        m = Normal(mean, scale)
        return -m.log_prob(input).mean()

