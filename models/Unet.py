#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:22:59 2020

@author: arthur
Implementation of the U-net structure
"""


import torch
from torch.nn import (Module, ModuleList, Parameter, Upsample, Sequential)
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.functional import pad
import torch.nn as nn
import mlflow
import numpy as np

class Unet(Module):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 height=0, width=0, n_scales: int = 2):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_scales = n_scales
        self.down_convs = ModuleList()
        self.up_convs = ModuleList()
        self.up_samplers = ModuleList()
        self.final_convs = None
        self._build_convs()

    def forward(self, x : torch.Tensor):
        blocks = list()
        for i in range(self.n_scales):
            x = self.down_convs[i](x)
            if i != self.n_scales - 1:
                blocks.append(x)
                x = self.down(x)
        blocks.reverse()
        for i in range(self.n_scales - 1):
            x = self.up(x, i)
            x = torch.cat((x, blocks[i]), 1)
            x = self.up_convs[i](x)
        return self.final_convs(x)

    def down(self, x):
        return F.max_pool2d(x, 2)

    def up(self, x, i):
        return self.up_samplers[i](x)

    def _build_convs(self):
        for i in range(self.n_scales):
            if i == 0:
                n_in_channels = self.n_in_channels
                n_out_channels = 64
            else:
                n_in_channels = n_out_channels
                n_out_channels = 2 * n_out_channels
            conv1 = torch.nn.Conv2d(n_in_channels, n_out_channels, 3, 1)
            conv2 = torch.nn.Conv2d(n_out_channels, n_out_channels, 3, 1)
            submodule = Sequential(conv1, nn.ReLU, conv2, nn.ReLU)
            self.down_convs.append(submodule)
        for i in range(self.n_scales):
            # Add the upsampler
            up_sampler = Upsample(mode='bilinear', scale_factor=2)
            conv = torch.nn.Conv2d(n_out_channels, n_out_channels // 2, 1)
            self.up_samplers.append(Sequential(up_sampler, conv))
            # The up convs
            n_in_channels = n_out_channels
            n_out_channels = n_out_channels // 2
            conv1 = torch.nn.Conv2d(n_in_channels, n_out_channels, 3, 1)
            conv2 = torch.nn.Conv2d(n_out_channels, n_out_channels, 3, 1)
            submodule = Sequential(conv1, nn.ReLU, conv2, nn.ReLU)
            self.up_convs.append(submodule)
        #Final convs
        conv1 = torch.nn.Conv2d(n_out_channels, n_out_channels,
                                3, 1)
        conv2 = torch.nn.Conv2d(n_out_channels, self.n_out_channels,
                                3, 1)
        self.final_convs = Sequential(conv1, nn.ReLU, conv2)
            
            