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
"""

# This is required to avoid some issue with matplotlib when running on NYU's 
# prince server
import os
if os.environ.get('DISPLAY', '') == '':
    import matplotlib
    matplotlib.use('agg')
# --------------------------------------------------------------------

import numpy as np
import xarray as xr
import mlflow
import os.path
import os

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

# Import some utils functions
from train.utils import DEVICE_TYPE, learning_rates_from_string

# import training class
from train.base import Trainer

# import to parse CLI arguments
import argparse

# import to create temporary dir used to save the model and predictions
# before logging through MLFlow
import tempfile


# PARAMETERS ---------
def negative_int(value: str):
    return -int(value)

description = 'Trains a model on a chosen dataset from the store. Allows \
    to set training parameters via the CLI.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('exp_id', type=int, 
                    help='Experiment id of the source dataset')
parser.add_argument('run_id', type=str,
                    help='Run id of the source dataset')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=learning_rates_from_string,
                    default={'0\1e-3'})
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--test_split', type=float, default=0.8)
parser.add_argument('--time_indices', type=negative_int, nargs='*')
parser.add_argument('--printevery', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.1,
                    help="Controls the weight decay on the linear layer")
parser.add_argument('--model_module_name', type=str, default='models.models1')
parser.add_argument('--model_cls_name', type=str, default='FullyCNN')
params = parser.parse_args()

# Log the experiment_id and run_id of the source dataset
mlflow.log_param('source.experiment_id', params.exp_id)
mlflow.log_param('source.run_id', params.run_id)

# Training parameters
# Note that we use two indices for the train/test split. This is because we
# want to avoid the time correlation to play in our favour during test.
batch_size = params.batchsize
learning_rates = params.learning_rate
weight_decay = params.weight_decay
n_epochs = params.n_epochs
train_split = params.train_split
test_split = params.test_split
model_module_name = params.model_module_name
model_cls_name = params.model_cls_name

# Parameters specific to the input data
# past specifies the indices from the past that are used for prediction
indices = params.time_indices

# Other parameters
print_loss_every = params.printevery

# Directories where temporary data will be saved
data_location = tempfile.mkdtemp(dir='/scratch/ag7531/temp/')
print('Created temporary dir at  ', data_location)

figures_directory = 'figures'
models_directory = 'models'
model_output_dir = 'model_output'
def _check_dir(dir_path):
    """Tries to create the directory if it does not already exists"""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

_check_dir(os.path.join(data_location, figures_directory))
_check_dir(os.path.join(data_location, models_directory))
_check_dir(os.path.join(data_location, model_output_dir))


# Device selection. If available we use the GPU.
# TODO Allow CLI argument to select the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = DEVICE_TYPE.GPU if torch.cuda.is_available() \
                              else DEVICE_TYPE.CPU
print('Selected device type: ', device_type.value)


# FIN PARAMETERS -----

# DATA----------------
# Load data from the store, according to experiment id and run id
mlflow_client = mlflow.tracking.MlflowClient()
data_file = mlflow_client.download_artifacts(params.run_id, 'forcing')
xr_dataset = xr.open_zarr(data_file).load()

# Rescale 
xr_dataset = xr_dataset / xr_dataset.std()

# Convert to a pytorch dataset and specify which variables are input/output
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
 
def function_used_to_toggle_in_spyder():
    pass
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
    pass

# Dataloaders are responsible for sending batches of data to the NN
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

print('Size of training data: {}'.format(len(train_dataset)))
print('Size of validation data : {}'.format(len(test_dataset)))
# FIN DATA------------


# NEURAL NETWORK------
# Remove *2 and make this to adapt to the dataset
height = dataset.height
width = dataset.width

# Recover the model's class
import importlib
try:
    models_module = importlib.import_module(model_module_name)
    model_cls = getattr(models_module, model_cls_name)
except ModuleNotFoundError as e:
    e.msg = 'Coud not find the specified model\'s module (' + e.msg + ')'
    raise e
except AttributeError as e:
    e.msg = 'Could not find the specified model\'s class (' + e.msg + ')'

net = model_cls(len(indices)*2, dataset.n_output_targets(),
               height, width)
print('----------*----------')
print(net)
print('--------------------')
print('***')
# To GPU
net.to(device)

# Log the text representation of the net into a txt artifact
with open(os.path.join(data_location, 'nn_architecture.txt'), 'w') as f:
    print('Writing neural net architecture into txt file.')
    f.write(str(net))
# FIN NEURAL NETWORK -


# Training------------
# MSE criterion + Adam optimizer
criterion = torch.nn.MSELoss()
linear_layer = net.linear_layer
conv_layers = net.conv_layers
params = [{'params' : layer.parameters()} for layer in conv_layers]
params.append({'params' : linear_layer.parameters(),
               'weight_decay' : weight_decay,
               'lr' : learning_rates[0] / 100})
optimizers = {i: optim.Adam(params, lr=v, weight_decay=0.0) 
              for (i, v) in learning_rates.items()}

trainer = Trainer(net, device)
trainer.criterion = criterion
trainer.print_loss_every = print_loss_every

for i_epoch in range(n_epochs):
    # Set to training mode
    if i_epoch in optimizers:
        optimizer = optimizers[i_epoch]
        print('Switching to new optimizer:\n', optimizer)
    print('Epoch number {}.'.format(i_epoch))
    train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
    test_loss = trainer.test(test_dataloader)
    # Log the training loss
    print('Train loss for this epoch is ', train_loss)
    print('Test loss for this epoch is ', test_loss)
    mlflow.log_metric('train mse', train_loss, i_epoch)
    mlflow.log_metric('test mse', test_loss, i_epoch)

    # We also save a snapshot figure to the disk and log it
    # TODO rewrite this bit, looks confusing for now
    ids_data = (np.random.randint(0, len(test_dataset)), 300)
    with torch.no_grad():
        for i, id_data in enumerate(ids_data):
            data = test_dataset[id_data]
            X, Y = data
            X = torch.tensor(X)
            Y = torch.tensor(Y)
            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, dtype=torch.float)
            X = torch.unsqueeze(X, dim=0)
            Y = torch.unsqueeze(Y, dim=0)
            Y_hat = net(X)
            Y = Y.cpu().numpy().squeeze()
            Y_hat = Y_hat.cpu().numpy().squeeze()
#            transformer = s.targets_transformer
#            true = transformer.inverse_transform(true)
#            pred = transformer.inverse_transform(pred)
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(Y[0])
            plt.subplot(122)
            plt.imshow(Y_hat[0])
            f_name = 'image{}-{}.png'.format(i, i_epoch)
            file_path = os.path.join(data_location, figures_directory, f_name)
            plt.savefig(file_path)
            plt.close(fig)
    # log the epoch
    mlflow.log_param('n_epochs', i_epoch + 1)


# Save the trained model to disk
print('Moving the network to the CPU before saving...')
net.cpu()
print('Saving the neural network learnt parameters to disk...')
model_name = 'trained_model.pth'
full_path = os.path.join(data_location, 'models', model_name)
torch.save(net.state_dict(), full_path)
print('Logging the neural network model...')
print('Neural network saved and logged in the artifacts.')
print('Now putting it back on the gpu')
net.cuda(device)

# FIN TRAINING -------

# CORRELATION MAP ----
u_v_surf = np.zeros(len(test_dataset, 2, dataset.height, dataset.width))
pred = np.zeros((len(test_dataset), 2, dataset.height, dataset.width))
truth = np.zeros((len(test_dataset), 2, dataset.height, dataset.width))

# Predictions on the test set using the trained model
net.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        u_v_surf[i * batch_size : (i+1) * batch_size] = data[0].numpy()
        truth[i * batch_size:(i+1) * batch_size] = data[1].numpy()
        X = data[0].to(device, dtype=torch.float)
        Y_hat = net(X)
        Y_hat = Y_hat.cpu().numpy()
        pred[i * batch_size:(i+1) * batch_size] = Y_hat
# Convert to dataset
new_dims = ('time', 'latitude', 'longitude')
coords = xr_dataset.coords
new_coords = {'time' : coords['time'][test_index:],
                                'latitude' : coords['yu_ocean'],
                                'longitude' : coords['xu_ocean']}
u_surf = xr.DataArray(data=u_v_surf[:, 0, ...], dims = new_dims, 
                      coords = new_coords)
v_surf = xr.DataArray(data=u_v_surf[:, 1, ...], dims = new_dims,
                      coords = new_coords)
s_x = xr.DataArray(data=truth[:, 0, ...], dims = new_dims, 
                      coords = new_coords)
s_y = xr.DataArray(data=truth[:, 1, ...], dims = new_dims, 
                      coords = new_coords)
s_x_pred = xr.DataArray(data=pred[:, 0, ...], dims = new_dims, 
                      coords = new_coords)
s_y_pred = xr.DataArray(data=pred[:, 1, ...], dims = new_dims, 
                      coords = new_coords)
output_dataset = xr.Dataset(data = {'u_surf' : u_surf, 'v_surf' : v_surf,
                                    'S_x': s_x, 'S_y' : s_y,
                                    'S_xpred' : s_x_pred,
                                    'S_ypred' : s_y_pred})

# Save dataset
output_dataset.to_zarr(os.path.join(data_location, model_output_dir,
                                    'test_output'))

# Correlation map, shape (2, dataset.width, dataset.height)
correlation_map = np.mean(truth * pred, axis=0)
correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
correlation_map /= np.maximum(np.std(truth, axis=0) * np.std(pred, axis=0),
                              1e-20)

print('Saving correlation map to disk')
# Save the correlation map to disk and its plot as well.
np.save(os.path.join(data_location, 'correlation_map'), correlation_map)

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

# Log artifacts
print('Logging artifacts...')
mlflow.log_artifact(data_location)
print('Done...')