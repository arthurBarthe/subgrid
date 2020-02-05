# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:56:02 2020

@author: Arthur
"""

import torch
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader

from ..models.utils.utils_nn import print_every, RunningAverage


class Trainer:
    def __init__(self, net: Module, device: torch.device):
        self._net = net
        self._device = device
        self._criterion = MSELoss()
        self._print_loss_every = 20
        self._locked = False

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net: Module):
        self._net = net

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        if self._locked:
            raise Exception('The criterion of the trainer cannot be \
                            changed after training has started.')
        self._criterion = criterion

    @property
    def print_loss_every(self):
        return self._print_loss_every

    @print_loss_every.setter
    def print_loss_every(self, value: int):
        self._print_loss_every = value

    def train_for_one_epoch(self, dataloader: DataLoader, optimizer):
        self._locked = True
        running_loss = RunningAverage()
        for i_batch, batch in enumerate(dataloader):
            # Zero the gradients
            self.net.zero_grad()
            # Move batch to the GPU (if possible)
            X = batch[0].to(self._device, dtype=torch.float)
            Y = batch[1].to(self._device, dtype=torch.float)
            # Compute loss
            loss = self.criterion(self.net(X), Y)
            running_loss.update(loss.item(), X.size(0))
            # Print current loss
            loss_text = 'Current loss value is {}'.format(running_loss)
            print_every(loss_text, self.print_loss_every, i_batch)
            # Backpropagate
            loss.backward()
            # Update parameters
            optimizer.step()
        return running_loss.value
