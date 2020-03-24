# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:25:09 2020

@author: Arthur
Here we test a trained model on unseen data which was coarse-grained at 
a different scale than that used for the training data of the model.
"""
import mlflow
import torch
import torch.nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import xarray as xr
from analysis.utils import select_run
from data.datasets import RawDataFromXrDataset
from train.base import Trainer

import os.path
import importlib

import tempfile
import logging

# Location used to write generated data before it is logged through MLFlow
data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# First we retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
cols = ['metrics.test mse', 'start_time', 'params.time_indices', 
        'params.model_cls_name', 'params.source.run_id']
model_run = select_run(sort_by='start_time', cols=cols,
                       experiment_ids=['2',])

# Load some extra parameters of the model.
# TODO allow general time_indices
time_indices = [0,]
train_split = float(model_run['params.train_split'])
test_split = float(model_run['params.test_split'])
learning_rate = float(model_run['params.learning_rate'])
batch_size = int(model_run['params.batchsize'])
source_data_id = model_run['params.source.run_id']
n_epochs = int(model_run['params.n_epochs'])
model_module_name = model_run['params.model_module_name']
model_cls_name = model_run['params.model_cls_name']


# Load the model's file
client = mlflow.tracking.MlflowClient()
model_file = client.download_artifacts(model_run.run_id, 'trained_model.pth')

# Test dataset
mlflow.set_experiment('forcingdata')
mlflow_runs = mlflow.search_runs()
cols = ['params.lat_min', 'params.lat_max', 
        'params.long_min', 'params.long_max',
        'params.scale']
        # 'params.scale_coarse', 'params.scale_fine']
data_run = select_run(sort_by=None, cols=cols)
# TODO check that the run_id is different from source_data_id
client = mlflow.tracking.MlflowClient()
data_file = client.download_artifacts(data_run.run_id, 'forcing')
xr_dataset = xr.open_zarr(data_file).load()


# Set the experiment to 'multiscale'
print('Logging to experiment multiscale')
mlflow.set_experiment('multiscale')
mlflow.start_run()

# Generate the dataset
dataset = RawDataFromXrDataset(xr_dataset)
dataset.index = 'time'
dataset.add_input('usurf')
dataset.add_input('vsurf')
dataset.add_output('S_x')
dataset.add_output('S_y')

width = dataset.width
height = dataset.height

train_index = int(train_split * len(dataset))
test_index = int(test_split * len(dataset))
train_dataset = Subset(dataset, np.arange(train_index))
test_dataset = Subset(dataset, np.arange(test_index, len(dataset)))
# TODO Allow multiple time indices.
# test_dataset = MultipleTimeIndices(test_dataset)
test_dataset.time_indices = time_indices
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

# Load the model itself
logging.info('Creating the neural network model')
try:
    module = importlib.import_module(model_module_name)
    model_cls = getattr(module, model_cls_name)
except ModuleNotFoundError as e:
    e.msg = 'Could not retrieve the module in which the trained model is \
        defined.' + e.msg
except AttributeError as e:
    e.msg = 'Could not retrieve the model\'s class. ' + e.msg
net = model_cls(2 * len(time_indices), dataset.n_output_targets(),
               height, width, True)
net.to(device=device)
logging.info('Loading the neural net paraeters')
net.load_state_dict(torch.load(model_file))

# Train the linear layer only
criterion = torch.nn.MSELoss()
print('width: {}, height: {}'.format(width, height))
optimizer = torch.optim.Adam(net.linear_layer.parameters(), lr=learning_rate)
print('Training the fully connected layer...')
net.to(device)
net.train()
trainer = Trainer(net, device)
trainer.criterion = criterion
for i_epoch in range(n_epochs):
    train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
    test_loss = trainer.test(test_dataloader)
    print('Epoch {}'.format(i_epoch))
    print('Train loss for this epoch is {}'.format(train_loss))
    print('Test loss for this epoch is {}'.format(test_loss))


# Log the run_id of the loaded model (useful to recover info
# about the scale that was used to train this model for
# instance.
mlflow.log_param('model_run_id', model_run.run_id)
# Log the run_id for the data
mlflow.log_param('data_run_id', data_run.run_id)
# Do the predictions for that dataset using the loaded model
predictions = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))
truth = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))

net.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(i)
        X = data[0].to(device, dtype=torch.float)
        pred_i = net(X)
        pred_i = pred_i.cpu().numpy()
        pred_i = np.reshape(pred_i, (-1, 2, dataset.width, dataset.height))
        predictions[i * batch_size:(i+1) * batch_size] = pred_i
        Y = np.reshape(data[1], (-1, 2, dataset.width, dataset.height))
        truth[i * batch_size:(i+1) * batch_size] = Y

np.save(os.path.join(data_location, 'predictions.npy'), predictions)
np.save(os.path.join(data_location, 'targets.npy'), truth)
mlflow.log_artifact(data_location)

mlflow.end_run()
