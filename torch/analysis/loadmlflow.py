# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:12:49 2020

@author: Arthur
Contains a class for loading an MLFlow run.
"""
import torch
from os.path import join
import numpy as np
import warnings


class LoadMLFlow:
    """Class to load an MLFlow run. In particular this allows to load the
    pytorch model if it was logged as an artifact, as well as the train and
    test split indices, and the predictions that were made on the test
    set."""
    def __init__(self,  run_id: str, experiment_id: int = 0,
                 mlruns_path: str = 'mlruns'):
        self.mlruns_path = mlruns_path
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.path = join(mlruns_path, str(experiment_id), run_id)
        self.paths = dict()
        self.paths['params'] = join(self.path, 'params')
        self.paths['artifacts'] = join(self.path, 'artifacts')
        # Neural net attributes
        self._net_class = None
        self._net_filename = ''
        self._net = None
        # Dataset attributes
        self._train_split = None
        self._test_split = None
        self._train_dataset = None
        self._test_dataset = None
        # Prediction attirbutes
        self._predictions = None
        self._true_targets = None

    @property
    def net_class(self):
        return self._net_class

    @net_class.setter
    def net_class(self, net_class: type):
        """Specifies the class used for the neural network"""
        self._net_class = net_class

    @property
    def net_filename(self):
        return join(self.paths['artifacts'], self._net_filename)

    @net_filename.setter
    def net_filename(self, net_filename: str):
        self._net_filename = net_filename

    @property
    def net(self):
        if not self._net:
            self._load_net()
        return self._net

    def _load_net(self, *net_params):
        net = self._net_class(*net_params)
        net.load_state_dict(torch.load(self.net_filename))
        self._net = net

    @property
    def train_split(self):
        # TODO generalize this by writing a single method for all params.
        if self._train_split is None:
            with open(join(self.paths['params'], 'train_split')) as f:
                self._train_split = float(f.readline())
        return self._train_split

    @train_split.setter
    def train_split(self, train_split: int):
        raise Exception('This should not be set by the user.')

    @property
    def test_split(self):
        # TODO generalize this by writing a single method for all params.
        if self._test_split is None:
            with open(join(self.paths['params'], 'train_split')) as f:
                self._test_split = float(f.readline())
        return self._test_split

    @test_split.setter
    def test_split(self, test_split: int):
        raise Exception('This should not be set by the user.')

    @property
    def predictions(self):
        """Returns the predictions made on the test dataset"""
        if self._predictions is None:
            try:
                self._predictions = np.load(join(self.paths['artifacts'],
                                                 'predictions.npy'))
            except FileNotFoundError:
                print('Predictions file not found for this run.')
        return self._predictions

    @property
    def true_targets(self):
        """Returns the true targets of the test dataset"""
        if self._true_targets is None:
            try:
                self._true_targets = np.load(join(self.paths['artifacts'],
                                                  'truth.npy'))
            except FileNotFoundError:
                warnings.warn('True targets not found for this run')
        return self._true_targets
