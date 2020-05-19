# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:25:09 2020

@author: Arthur
Here we test a trained model on an unseen region. The user is prompted to
select a trained model within a list and a new region to test that model.
Fine-tuning is an option through the n_epochs parameter of the script.
"""
import mlflow
import torch
import torch.nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import xarray as xr
from analysis.utils import select_run
from train.utils import learning_rates_from_string
from data.datasets import (RawDataFromXrDataset, DatasetTransformer,
                           ConcatDatasetWithTransforms, Subset_)
from train.base import Trainer
from train.losses import (HeteroskedasticGaussianLoss, 
                          HeteroskedasticGaussianLossV2)

import os.path
import importlib

import tempfile
import logging

import argparse
import pickle

from sys import modules

# Parse arguments
# n_epochs : Number of epochs we fine-tune the model on the new data
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=0)
parser.add_argument('--lr_ratio', type=float, default=1)
script_params = parser.parse_args()
n_epochs = script_params.n_epochs
lr_ratio = script_params.lr_ratio

# Location used to write generated data before it is logged through MLFlow
data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')
model_output_dir = 'model_output'

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prompt user to retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
cols = ['metrics.test mse', 'start_time', 'params.time_indices',
        'params.model_cls_name', 'params.source.run_id']
model_run = select_run(sort_by='start_time', cols=cols,
                       experiment_ids=['2', ])

# Load some extra parameters of the model.
# TODO allow general time_indices
time_indices = [0, ]
train_split = float(model_run['params.train_split'])
test_split = float(model_run['params.test_split'])
batch_size = int(model_run['params.batchsize'])
source_data_id = model_run['params.source.run_id']
model_module_name = model_run['params.model_module_name']
model_cls_name = model_run['params.model_cls_name']
loss_cls_name = model_run['params.loss_cls_name']
learning_rates = learning_rates_from_string(model_run['params.learning_rate'])
weight_decay = float(model_run['params.weight_decay'])

learning_rate = learning_rates[0] * lr_ratio

# Load the model's file
client = mlflow.tracking.MlflowClient()
model_file = client.download_artifacts(model_run.run_id,
                                       'models/trained_model.pth')
transformation_file = client.download_artifacts(model_run.run_id,
                                                'models/transformation')
features_transform_file = client.download_artifacts(model_run.run_id,
                                                'models/features_transform')
try:
    targets_transform_file = client.download_artifacts(model_run.run_id,
                                                    'models/targets_transform')
except FileNotFoundError:
    targets_transform_file = None


with open(transformation_file, 'rb') as f:
    transformation = pickle.load(f)
with open(features_transform_file, 'rb') as f:
    features_transform = pickle.load(f)
if targets_transform_file is not None:
    with open(targets_transform_file, 'rb') as f:
        targets_transform = pickle.load(f)


# Prompt user to select the test dataset
mlflow.set_experiment('forcingdata')
cols = ['params.lat_min', 'params.lat_max', 'params.long_min',
        'params.long_max', 'params.scale']
data_run = select_run(cols=cols)
# TODO check that the run_id is different from source_data_id
client = mlflow.tracking.MlflowClient()
data_file = client.download_artifacts(data_run.run_id, 'forcing')

# Set the experiment to 'multiscale'
print('Logging to experiment multiscale')
mlflow.set_experiment('multiregion')
mlflow.start_run()

# Log the run_id of the loaded model (useful to recover info
# about the scale that was used to train this model for
# instance.
mlflow.log_param('model_run_id', model_run.run_id)
# Log the run_id for the data
mlflow.log_param('data_run_id', data_run.run_id)
mlflow.log_param('n_epochs', n_epochs)

# Read the dataset file
xr_dataset = xr.open_zarr(data_file).load()

# To PyTorch Dataset
dataset = RawDataFromXrDataset(xr_dataset)
dataset.index = 'time'
dataset.add_input('usurf')
dataset.add_input('vsurf')
dataset.add_output('S_x')
dataset.add_output('S_y')

train_index = int(train_split * len(dataset))
test_index = int(test_split * len(dataset))
train_dataset = Subset_(dataset, np.arange(train_index))
test_dataset = Subset_(dataset, np.arange(test_index, len(dataset)))

if targets_transform is None:
    transform = DatasetTransformer(features_transform)
else:
    transform = DatasetTransformer(features_transform, targets_transform)
transform.fit(train_dataset)
train_dataset = ConcatDatasetWithTransforms((train_dataset,), (transform,))
test_dataset = ConcatDatasetWithTransforms((test_dataset,), (transform,))

# TODO Allow multiple time indices.
# test_dataset = MultipleTimeIndices(test_dataset)
test_dataset.time_indices = time_indices
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)
print('Size of training data: {}'.format(len(train_dataset)))
print('Size of validation data : {}'.format(len(test_dataset)))

# Load the model itself
logging.info('Creating the neural network model')
try:
    module = importlib.import_module(model_module_name)
    model_cls = getattr(module, model_cls_name)
except ModuleNotFoundError as e:
    raise type(e)('Could not retrieve the module in which the trained model \
                  is defined: ' + str(e))
except AttributeError as e:
    raise type(e)('Could not retrieve the model\'s class. ' + str(e))

net = model_cls(dataset.n_features, 2*dataset.n_targets, dataset.height,
                dataset.width)
net._final_transformation = transformation

net.to(device=device)
logging.info('Loading the neural net parameters')
# Load parameters of pre-trained model
net.load_state_dict(torch.load(model_file))

# Set up training criterion and select parameters to train
try:
    criterion = getattr(modules['__main__'], loss_cls_name)()
except AttributeError as e:
    raise type(e)('Could not find the loss class used for training.')


print('width: {}, height: {}'.format(dataset.width, dataset.height))
# If the model has defined a linear layer we train that only. Otherwise
# we train all the parameters for now
if net.linear_layer is not None:
    print('Training final layer only')
    parameters = net.linear_layer.parameters()
else:
    print('Fine-tuning whole network')
    parameters = net.parameters()
optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                             weight_decay=0)
net.to(device)
trainer = Trainer(net, device)
trainer.criterion = criterion

# Training itself
for i_epoch in range(n_epochs):
    train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
    test_loss, metrics_results = trainer.test(test_dataloader)
    print('Epoch {}'.format(i_epoch))
    print('Train loss for this epoch is {}'.format(train_loss))
    print('Test loss for this epoch is {}'.format(test_loss))
    mlflow.log_metric('train loss', train_loss, i_epoch)
    mlflow.log_metric('test loss', test_loss, i_epoch)

# Final test
train_loss, train_metrics_results = trainer.test(train_dataloader)
test_loss, test_metrics_results = trainer.test(test_dataloader)
print(f'Final train loss is {train_loss}')
print(f'Final test loss is {test_loss}')

# Do the predictions for that dataset using the loaded model
velocities = np.zeros((len(test_dataset), 2, dataset.height, dataset.width))
predictions = np.zeros((len(test_dataset), 4, dataset.height, dataset.width))
truth = np.zeros((len(test_dataset), 2, dataset.height, dataset.width))

net.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(i)
        velocities[i * batch_size: (i + 1) * batch_size] = data[0].numpy()
        truth[i * batch_size: (i + 1) * batch_size] = data[1].numpy()
        X = data[0].to(device, dtype=torch.float)
        pred_i = net(X)
        pred_i = pred_i.cpu().numpy()
        predictions[i * batch_size: (i+1) * batch_size] = pred_i

# Put this into an xarray dataset before saving
new_dims = ('time', 'latitude', 'longitude')
coords = xr_dataset.coords
new_coords = {'time': coords['time'][test_index:],
              'latitude': coords['yu_ocean'].data,
              'longitude': coords['xu_ocean'].data}
u_surf = xr.DataArray(data=velocities[:, 0, ...], dims=new_dims,
                      coords=new_coords)
v_surf = xr.DataArray(data=velocities[:, 1, ...], dims=new_dims,
                      coords=new_coords)
s_x = xr.DataArray(data=truth[:, 0, ...], dims=new_dims, coords=new_coords)
s_y = xr.DataArray(data=truth[:, 1, ...], dims=new_dims, coords=new_coords)
s_x_pred = xr.DataArray(data=predictions[:, 0, ...], dims=new_dims,
                        coords=new_coords)
s_y_pred = xr.DataArray(data=predictions[:, 1, ...], dims=new_dims,
                        coords=new_coords)
s_x_pred_scale = xr.DataArray(data=predictions[:, 2, ...], dims=new_dims,
                              coords=new_coords)
s_y_pred_scale = xr.DataArray(data=predictions[:, 3, ...], dims=new_dims,
                              coords=new_coords)
output_dataset = xr.Dataset({'u_surf': u_surf, 'v_surf': v_surf,
                             'S_x': s_x, 'S_y': s_y, 'S_xpred': s_x_pred,
                             'S_ypred': s_y_pred, 'S_xscale': s_x_pred_scale,
                             'S_yscale': s_y_pred_scale})

# Save dataset
file_path = os.path.join(data_location, 'test_output')
output_dataset.to_zarr(file_path)
mlflow.log_artifact(file_path)
mlflow.end_run()
