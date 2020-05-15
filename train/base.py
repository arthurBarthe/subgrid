# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:56:02 2020

@author: Arthur
"""

import torch
from torch.nn import Module, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .utils import print_every, RunningAverage


class Trainer:
    """Training object for a neural network on a specific device. Defines
    a training method that trains for one epoch.

    Properties
    ----------

    :net: Module,
        Neural network that is trained

    :criterion: Loss,
        Criterion used in the objective function.

    :print_loss_every: int,
        Sets the number of batches that the average loss is printed.

    :metrics: list,
        List of metrics reported on the test data. These are distinct from
        the criterion in the sense that they are not use for backpropagation,
        they are only reported on the test dataset (although this might
        change in the future).
        Note: for now the passed metric are expected to be linear as we use
        a running average to compute them.
    """

    def __init__(self, net: Module, device: torch.device):
        self._net = net
        self._device = device
        self._criterion = MSELoss()
        self._metrics = dict()
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
    def print_loss_every(self) -> int:
        return self._print_loss_every

    @print_loss_every.setter
    def print_loss_every(self, value: int):
        self._print_loss_every = value

    @property
    def metrics(self):
        return self._metrics

    def register_metric(self, metric_name, metric_func):
        self._metrics[metric_name] = metric_func

    def train_for_one_epoch(self, dataloader: DataLoader, optimizer,
                            clip=None) -> float:
        """Trains the neural network for one epoch using the data provided
        through the dataloader passed as an argument, and the optimizer
        passed as an argument.

        Parameters
        ----------

        :dataloader: DataLoader,
            The Pytorch DataLoader object used to provide training data.

        :optimizer: Optimizer,
            The Pytorch Optimizer used to update the parameters after each
            forward-backward pass.

        Returns
        -------
        float
            The average train loss for this epoch.
        """
        self.net.train()
        self._locked = True
        running_loss = RunningAverage()
        running_loss_ = RunningAverage()
        for i_batch, batch in enumerate(dataloader):
            # Zero the gradients
            self.net.zero_grad()
            # Move batch to the GPU (if possible)
            X = batch[0].to(self._device, dtype=torch.float)
            Y = batch[1].to(self._device, dtype=torch.float)
            Y_hat = self.net(X)
            # Compute loss
            loss = self.criterion(Y, Y_hat)
            running_loss.update(loss.item(), X.size(0))
            running_loss_.update(loss.item(), X.size(0))
            # Print current loss
            loss_text = 'Loss value {}'.format(running_loss_.average)
            if print_every(loss_text, self.print_loss_every, i_batch):
                # Every time we print we reset the running average
                running_loss_.reset()
            # Backpropagate
            if clip:
                clip_grad_norm_(self.net.parameters(), clip)
            loss.backward()
            # Update parameters
            optimizer.step()
        return running_loss.value

    def test(self, dataloader) -> float:
        """Returns the validation loss on the provided data. The criterion
        used is the same as the one used for the training.

        Parameters
        ----------

        :dataloader: Dataloader,
            The Pytorch dataloader providing the data for validation.


        Returns
        ----------
        (float, dict)
            The validation loss calculated over the provided data.
            A dictionary of the computed metrics over the test dataset.
        """
        # TODO add something to check that the dataloader is different from
        # that used for the training
        self.net.eval()
        running_loss = RunningAverage()
        metrics_results = dict()
        for metric_name in self.metrics:
            metrics_results[metric_name] = RunningAverage()
        with torch.no_grad():
            for i_batch, batch in enumerate(dataloader):
                # Move batch to GPU
                X = batch[0].to(self._device, dtype=torch.float)
                Y = batch[1].to(self._device, dtype=torch.float)
                Y_hat = self.net(X)
                # Compute loss
                loss = self.criterion(Y, Y_hat)
                running_loss.update(loss.item(), X.size(0))
                # Compute metrics
                for metric_name, metric_func in self.metrics.items():
                    metrics_results[metric_name].update(metric_func(Y, Y_hat),
                                                        X.size(0))
            return running_loss.value, {metric_name: running_avg.value for
                                        metric_name, running_avg in
                                        metrics_results.items()}
