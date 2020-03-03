# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:13:28 2019

@author: Arthur
Torch implementation of a similar form of NN as in Bolton et al

TODO list
- check that the divergence layer works as expected, i.e. check sum of 
output layer is zero.
- analyze the importance of memory 
- train on one scale (e.g. 30 km) and test on a range of different
scales (e.g 10km, 20km, 40km, 50km, 60km). For that I need to write 
a new file in analysis/ that does testing on a specific dataset.
Additionally any dataset should have some info about its scale. During the 
training we need to log on which dataset / scale we train.
- Maybe use the xarray library?
"""
# Necessary when running on remote ssh to obtain the plots locally
import matplotlib
matplotlib.use('Agg')

import numpy as np
import xarray as xr
import mlflow
import os.path
from datetime import datetime

# For pre-processing
# from sklearn.preprocessing import StandardScaler, RobustScaler

# For neural networks
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn

# For plots
import matplotlib.pyplot as plt


# Import our Dataset class and neural network
from data.datasets import RawData, MultipleTimeIndices, DatasetClippedScaler
from data.datasets import RawDataFromXrDataset
from models.full_cnn1 import FullyCNN

# Import some utils functions
from train.utils import RunningAverage, DEVICE_TYPE

# import training class
from train.base import Trainer

# import to parse CLI arguments
import argparse

# Set the mlflow tracking uri
# mlflow.set_tracking_uri('file:///data/ag7531/mlruns')

# PARAMETERS ---------
def negative_int(value: str):
    return -int(value)

description = 'Trains a model on a chosen dataset from the store. Allows \
    to set training parameters via the CLI.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('exp_id', type=int, 
                    help='Experiment id of the source dataset')
parser.add_argument('run_id', type=str,
                    help='Run if of the source dataset')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--test_split', type=float, default=0.8)
parser.add_argument('--time_indices', type=negative_int, nargs='*')
parser.add_argument('--printevery', type=int, default=20)
params = parser.parse_args()

# Log the experiment_id and run_id of the source dataset
mlflow.log_param('source.experiment_id', params.exp_id)
mlflow.log_param('source.run_id', params.run_id)

# Training parameters
# Note that we use two indices for the train/test split. This is because we
# want to avoid the time correlation to play in our favour during test.
batch_size = params.batchsize
learning_rates = {0: params.learning_rate}
n_epochs = params.n_epochs
train_split = params.train_split
test_split = params.test_split

# Parameters specific to the input data
# past specifies the indices from the past that are used for prediction
indices = params.time_indices

# Other parameters
print_loss_every = params.printevery
data_location = '/data/ag7531/'
figures_directory = 'figures'

# Device selection. If available we use the GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = DEVICE_TYPE.GPU if torch.cuda.is_available() \
                              else DEVICE_TYPE.CPU
print('Selected device type: ', device_type.value)

# Should not be necessary anymore as automatic now. To be deleted.
# Log the parameters with mlflow
# mlflow.log_param('batch_size', batch_size)
# mlflow.log_param('learning_rate', learning_rates)
# mlflow.log_param('device', device_type.value)
# mlflow.log_param('train_split', train_split)
# mlflow.log_param('test_split', test_split)

# FIN PARAMETERS -----

# DATA----------------
# Load data from the store, according to experiment id and run id
mlflow_client = mlflow.tracking.MlflowClient()
data_file = mlflow_client.download_artifacts(params.run_id, 'forcing')
xr_dataset = xr.open_zarr(data_file).load()

# Rescale 
xr_dataset = (xr_dataset - xr_dataset.mean()) / xr_dataset.std()
# TODO Deal with this properly :Additional rescaling for the output 
xr_dataset['S_x'] = xr_dataset['S_x']
xr_dataset['S_y'] = xr_dataset['S_y']

dataset = RawDataFromXrDataset(xr_dataset)
dataset.index = 'time'
dataset.add_input('usurf')
dataset.add_input('vsurf')
dataset.add_output('S_x')
dataset.add_output('S_y')

# Split train/test
n_indices = len(dataset)
train_index = int(train_split * n_indices)
test_index = int(test_split * n_indices)
train_dataset = Subset(dataset, np.arange(train_index))
test_dataset = Subset(dataset, np.arange(test_index, n_indices))



# Apply basic normalization transforms (using the training data only)
# s = DatasetClippedScaler()
# s.fit(train_dataset)
# train_dataset = s.transform(train_dataset)
# test_dataset = s.transform(test_dataset)

# Specifies which time indices to use for the prediction
# train_dataset = MultipleTimeIndices(train_dataset)
# train_dataset.time_indices = indices
# test_dataset = MultipleTimeIndices(test_dataset)
# test_dataset.time_indices = indices

# Dataloaders are responsible for sending batches of data to the NN
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)
# FIN DATA------------


# NEURAL NETWORK------
# Remove *2 and make this to adapt to the dataset
width = dataset.width
height = dataset.height
net = FullyCNN(len(indices)*2, dataset.n_output_targets(),
               width, height)
print('--------------------')
print(net)
print('--------------------')
print('***')
# To GPU
net.to(device)

# Log the text representation of the net into a txt artifact
with open('nn_architecture.txt', 'w') as f:
    print('Writing neural net architecture into txt file.')
    f.write(str(net))
mlflow.log_artifact('nn_architecture.txt')

# MSE criterion + Adam optimizer
criterion = torch.nn.MSELoss()
optimizers = {i: optim.Adam([
                                {'params': net.parameters()},
                                {'params': net.linear_layer.parameters(),
                                 'weight_decay': 0.1},
                            ],
                            lr=v, weight_decay=0.0) for (i, v) in
                              learning_rates.items()}

# FIN NEURAL NETWORK - 

trainer = Trainer(net, device)
trainer.criterion = criterion

# Training------------
for i_epoch in range(n_epochs):
    # Set to training mode
    net.train()
    if i_epoch in optimizers:
        optimizer = optimizers[i_epoch]
    print('Epoch number {}.'.format(i_epoch))
    running_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
    # Log the training loss
    print('Train loss for this epoch is ', running_loss)
    mlflow.log_metric('train mse', running_loss, i_epoch)
    
    # Eval mode
    net.eval()

    # At the end of each epoch we compute the test loss and print it
    with torch.no_grad():
        nb_samples = 0
        running_loss = RunningAverage()
        for i, data in enumerate(test_dataloader):
            X = data[0].to(device, dtype=torch.float)
            Y = data[1].to(device, dtype=torch.float)
            loss = criterion(net(X), Y)
            running_loss.update(loss.item(), X.size(0))
    print('Test loss for this epoch is ', running_loss)
    mlflow.log_metric('test mse', running_loss.value, i_epoch)

    # We also save a snapshot figure to the disk and log it
    # TODO rewrite this bit, looks confusing for now
#     ids_data = (np.random.randint(0, len(test_dataset)), 300)
#     with torch.no_grad():
#         for i, id_data in enumerate(ids_data):
#             data = test_dataset[id_data]
#             X = torch.tensor(data[0][np.newaxis, ...]).to(device,
#                                                           dtype=torch.float)
#             true = data[1][np.newaxis, ...]
#             pred = net(X).cpu().numpy()
# #            transformer = s.targets_transformer
# #            true = transformer.inverse_transform(true)
# #            pred = transformer.inverse_transform(pred)
#             fig = dataset.plot_true_vs_pred(true, pred)
#             f_name = 'image{}-{}.png'.format(i_epoch, i)
#             file_path = os.path.join(data_location, figures_directory, f_name)
#             plt.savefig(file_path)
#             plt.close(fig)
    # log the epoch
    mlflow.log_param('n_epochs', i_epoch + 1)


# Save the trained model to disk
print('Saving the neural network learnt parameters to disk...')
#model_name = str(datetime.now()).split('.')[0] + '.pth'
model_name = 'trained_model.pth'
full_path = os.path.join(data_location, 'models', model_name)
torch.save(net.state_dict(), full_path)
mlflow.log_artifact(full_path)
print('Neural network saved and logged in the artifacts.')

# FIN TRAINING -------

# CORRELATION MAP ----
pred = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))
truth = np.zeros((len(test_dataset), 2, dataset.width, dataset.height))

# Predictions on the test set using the trained model
net.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        X = data[0].to(device, dtype=torch.float)
        pred_i = net(X)
        pred_i = pred_i.cpu().numpy()
        pred_i = np.reshape(pred_i, (-1, 2, dataset.width, dataset.height))
        pred[i * batch_size:(i+1) * batch_size] = pred_i
        Y = np.reshape(data[1], (-1, 2, dataset.width, dataset.height))
        truth[i * batch_size:(i+1) * batch_size] = Y

# log the predictions as artifacts
np.save(os.path.join(data_location, 'models', 'predictions'), pred)
np.save(os.path.join(data_location, 'models', 'truth'), truth)
mlflow.log_artifact(os.path.join(data_location, 'models', 'predictions.npy'))
mlflow.log_artifact(os.path.join(data_location, 'models', 'truth.npy'))


# Correlation map, shape (2, dataset.width, dataset.height)
correlation_map = np.mean(truth * pred, axis=0)
correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
correlation_map /= np.maximum(np.std(truth, axis=0) * np.std(pred, axis=0),
                              1e-20)

print('Saving correlation map to disk')
# Save the correlation map to disk and its plot as well.
np.save('/data/ag7531/analysis/correlation_map', correlation_map)

fig = plt.figure()
plt.subplot(121)
plt.imshow(correlation_map[0], vmin=0, vmax=1, origin='lower')
plt.colorbar()
plt.title('Correlation map for S_x')
plt.subplot(122)
plt.imshow(correlation_map[1], vmin=0, vmax=1, origin='lower')
plt.colorbar()
plt.title('Correlation map for S_y')
f_name = 'Correlation_maps.png'
file_path = os.path.join(data_location, figures_directory, f_name)
plt.savefig(file_path)
plt.close(fig)

# FIN CORRELATION MAP 

# log the figures as artifacts
mlflow.log_artifact(os.path.join(data_location, figures_directory))
# log the correlation map figure
mlflow.log_artifact(file_path)
if 'y' in input('register as success?').lower():
    mlflow.set_tag('success', 'True')
else:
    mlflow.set_tag('success', 'False')
