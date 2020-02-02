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
from analysis.loadmlflow import LoadMLFlow
from models.full_cnn1 import FullyCNN
from data.datasets import RawData, MultipleTimeIndices
from torch.utils.data import DataLoader, Subset
from os.path import join

# Select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the mlflow tracking uri
mlflow.set_tracking_uri('file:///data/ag7531/mlruns')

# First we retrieve a trained model based on a run id for the default
# experiment (folder mlruns/0)
mlflow.set_experiment('default')
mlflow_runs = mlflow.search_runs()
print(mlflow_runs)

id_ = int(input('Run id?'))
model_run_id = mlflow_runs['run_id'][id_]
mlflow_loader = LoadMLFlow(model_run_id, experiment_id=0)

# Load some extra parameters of the model.
time_indices = mlflow_loader.time_indices
test_split = mlflow_loader.test_split
batch_size = mlflow_loader.batch_size

# Test dataset
mlflow.set_experiment('data')
mlflow_runs = mlflow.search_runs()
print(mlflow_runs)
id_ = int(input('Data run id?'))
data_run_id = mlflow_runs['run_id'][id_]
data_run = mlflow.get_run(data_run_id)
dataset = RawData(data_location=data_run.info.artifact_uri)
test_index = test_split * len(dataset)
test_dataset = Subset(dataset, np.arange(test_index, len(dataset)))
dataset = MultipleTimeIndices(dataset)
dataset.time_indices = time_indices
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)


# Load the model itself
mlflow_loader.net_filename = 'trained_model.pth'
net = FullyCNN(len(time_indices), dataset.output_size)
net.load_state_dict(mlflow_loader.net_filename)
net.eval()
# Set the experiment to testing multiscale
mlflow.set_experiment('testing multiscale')


with mlflow.start_run():
    # Log the run_id of the loaded model (useful to recover info
    # about the scale that was used to train this model for
    # instance.
    mlflow.log_param('model_run_id', model_run_id)
    # Log the run_id for the data
    mlflow.log_param('data_run_id', data_run_id)
    # Do the predictions for that dataset using the loaded model
    predictions = np.zeros(len(dataset), 2, dataset.width, dataset.height)
    truth = np.zeros(len(dataset), 2, dataset.width, dataset.height)

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            X = data[0].to(device, dtype=torch.float)
            pred_i = net(X)
            pred_i = pred_i.cpu().numpy()
            pred_i = np.reshape(pred_i, (-1, 2, dataset.width, dataset.height))
            predictions[i * batch_size:(i+1) * batch_size] = pred_i
            Y = np.reshape(data[1], (-1, 2, dataset.width, dataset.height))
            truth[i * batch_size:(i+1) * batch_size] = Y

    np.save('predictions', predictions)
    np.save('targets', truth)
    mlflow.log_artifact('predictions')
    mlflow.log_artifact('targets')
