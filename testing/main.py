# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:25:09 2020

@author: Arthur
Here we test a trained model on unseen data which was coarse-grained at 
a different scale than that used for the training data of the model.
"""
import mlflow
import torch
import numpy as np
import xarray as xr
from analysis.utils import select_run
from models.full_cnn1 import FullyCNN
from data.datasets import RawDataFromXrDataset
from torch.utils.data import DataLoader, Subset
import os.path

import tempfile

# Location used to write generated data before it is logged through MLFlow
data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# First we retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
cols = ['metrics.test mse', 'start_time', 'params.time_indices']
model_run = select_run(sort_by=cols[0], cols=cols[1:],
                       experiment_ids=['2',])

# Load some extra parameters of the model.
time_indices = model_run.time_indices
test_split = model_run.test_split
batch_size = model_run.batch_size

# Load the model's file
client = mlflow.tracking.MlflowClient()
model_file = client.download_artifacts(model_run.run_id, 'model.pth')

# Test dataset
mlflow.set_experiment('forcingdata')
mlflow_runs = mlflow.search_runs()
cols = ['params.lat_min, params.lat_max', 
        'params.long_min', 'params.long_max',
        'params.scale_coarse', 'params.scale_fine']
data_run, experiment_id = select_run(sort_by=None, cols=cols)
client = mlflow.tracking.MlflowClient()
data_file = client.download_artifacts(data_run.run_id, 'forcing')
xr_dataset = xr.open_zarr(data_file).load()


# Set the experiment to 'multiscale'
print('Logging to experiment multiscale')
mlflow.set_experiment('multiscale')
mlflow.start_run()

# Generate the dataset
dataset = RawDataFromXrDataset(xr_dataset)
test_index = int(test_split * len(dataset))
test_dataset = Subset(dataset, np.arange(test_index, len(dataset)))
# TODO Allow multiple time indices.
# test_dataset = MultipleTimeIndices(test_dataset)
test_dataset.time_indices = time_indices
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

# Load the model itself
net = FullyCNN(len(time_indices), dataset.n_output_targets)
net.to(device=device)
net.load_state_dict(torch.load(model_file))
net.eval()

# Log the run_id of the loaded model (useful to recover info
# about the scale that was used to train this model for
# instance.
mlflow.log_param('model_run_id', model_run.run_id)
# Log the run_id for the data
mlflow.log_param('data_run_id', data_run.run_id)
# Do the predictions for that dataset using the loaded model
predictions = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))
truth = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))

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
