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
from ..analysis.loadmlflow import LoadMLFlow
from ..analysis.utils import select_run
from ..models.full_cnn1 import FullyCNN
from ..data.datasets import RawData, MultipleTimeIndices
from torch.utils.data import DataLoader, Subset
from os.path import join

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the mlflow tracking uri
mlflow.set_tracking_uri('file:///data/ag7531/mlruns/')

# First we retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
cols = ['metrics.test mse', 'start_time', 'params.time_indices']
model_run_id, experiment_id = select_run(sort_by=cols[0], cols=cols[1:])
mlflow_loader = LoadMLFlow(model_run_id, experiment_id=0,
                           mlruns_path='/data/ag7531/mlruns')

# Load some extra parameters of the model.
time_indices = mlflow_loader.time_indices
test_split = mlflow_loader.test_split
batch_size = mlflow_loader.batch_size

# Test dataset
mlflow.set_experiment('data')
mlflow_runs = mlflow.search_runs()
cols = ['params.scale_coarse', 'params.scale_fine']
data_run_id, experiment_id = select_run(sort_by=None, cols=cols)
mlflow_loader = LoadMLFlow(data_run_id, experiment_id,
                           mlruns_path='/data/ag7531/mlruns')
data_location = mlflow_loader.paths['artifacts']

# Set the experiment to 'multiscale'
print('Logging to experiment multiscale')
mlflow.set_experiment('multiscale')
mlflow.start_run()

# Generate the dataset
dataset = RawData(data_location=data_location)
test_index = int(test_split * len(dataset))
test_dataset = Subset(dataset, np.arange(test_index, len(dataset)))
test_dataset = MultipleTimeIndices(test_dataset)
test_dataset.time_indices = time_indices
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

# Load the model itself
mlflow_loader.net_filename = 'trained_model.pth'
net = FullyCNN(len(time_indices), dataset.n_output_targets)
net.to(device=device)
net.load_state_dict(torch.load(mlflow_loader.net_filename))
net.eval()

# Log the run_id of the loaded model (useful to recover info
# about the scale that was used to train this model for
# instance.
mlflow.log_param('model_run_id', model_run_id)
# Log the run_id for the data
mlflow.log_param('data_run_id', data_run_id)
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

np.save('/data/ag7531/predictions/predictions.npy', predictions)
np.save('/data/ag7531/predictions/targets.npy', truth)
mlflow.log_artifact('/data/ag7531/predictions/predictions.npy')
mlflow.log_artifact('/data/ag7531/predictions/targets.npy')

mlflow.end_run()
