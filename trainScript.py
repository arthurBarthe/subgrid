# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:13:28 2019

@author: Arthur
Ideas:
    - data augmentation by *lambda -> *lambda**2 forcing
    - 
"""
# This is required to avoid some issue with matplotlib when running on NYU's
# prince server
import os
if os.environ.get('DISPLAY', '') == '':
    import matplotlib
    matplotlib.use('agg')

import numpy as np
import xarray as xr
import mlflow
import os.path

# For pre-processing
# from sklearn.preprocessing import StandardScaler, RobustScaler

# For neural networks
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn
import torch.nn.functional as F

# For plots
import matplotlib.pyplot as plt


# Import our Dataset class and neural network
from data.datasets import RawDataFromXrDataset, DatasetTransformer
import data.datasets

# Import some utils functions
from train.utils import DEVICE_TYPE, learning_rates_from_string

# import training class
from train.base import Trainer

# import losses
import train.losses

import models.transforms

# import to parse CLI arguments
import argparse

# import to create temporary dir used to save the model and predictions
# before logging through MLFlow
import tempfile

# import to import the module containing the model
import importlib
import pickle




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
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help="Controls the weight decay on the linear layer")
parser.add_argument('--model_module_name', type=str, default='models.models1')
parser.add_argument('--model_cls_name', type=str, default='FullyCNN')
parser.add_argument('--loss_cls_name', type=str,
                    default='HeteroskedasticGaussianLoss')
parser.add_argument('--transformation_cls_name', type=str,
                    default='SquareTransform')
parser.add_argument('--data_transform_cls_name', type=str,
                    default='UniformScaler')
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
loss_cls_name = params.loss_cls_name
transformation_cls_name = params.transformation_cls_name
data_transform_cls_name = params.data_transform_cls_name

# Parameters specific to the input data
# past specifies the indices from the past that are used for prediction
indices = params.time_indices

# Other parameters
print_loss_every = params.printevery
model_name = 'trained_model.pth'

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
# TODO normalization should be logged as well as its inverse. Should only be
# based on the training data.
# xr_dataset = xr_dataset / xr_dataset.std()


def arctan_normalize(x, max_value):
    return np.arctan(x / max_value)


def inv_arctan_normalize(x, max_value):
    return torch.tan(x) * max_value


# max_input = max(xr_dataset[['usurf', 'vsurf']].max()).item()
# max_output = max(xr_dataset[['S_x', 'S_y']].max()).item()
# xr_dataset['usurf'] = xr_dataset['usurf'] / xr_dataset
# xr_dataset['vsurf'] = xr_dataset['vsurf'] / max_input
# xr_dataset['S_x'] = xr_dataset['S_x'] / max_output
# xr_dataset['S_y'] = xr_dataset['S_y'] / max_output

# We transform the forcing by taking its square root (while conserving sign)
# xr_dataset['S_x'] = (np.sign(xr_dataset['S_x']) *
#                      np.sqrt(abs(xr_dataset['S_x'])))
# xr_dataset['S_y'] = (np.sign(xr_dataset['S_y']) *
#                      np.sqrt(abs(xr_dataset['S_y'])))


# xr_dataset = xr_dataset / xr_dataset.std()

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

# Apply some normalization
try:
    data_transform = getattr(data.datasets, data_transform_cls_name)
except AttributeError as e:
    raise type(e)('Could not find the dataset transform class: ' +
                  str(e))
# The following line allows to apply the transformation separately to
# features and targets.
dataset_transform = DatasetTransformer(data_transform)
train_dataset = dataset_transform.fit_transform(train_dataset)
test_dataset = dataset_transform.transform(test_dataset)


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
try:
    models_module = importlib.import_module(model_module_name)
    model_cls = getattr(models_module, model_cls_name)
except ModuleNotFoundError as e:
    raise type(e)('Could not find the specified model class: ' +
                  str(e))
except AttributeError as e:
    raise type(e)('Could not find the specified model class: ' +
                  str(e))

net = model_cls(len(indices)*2, dataset.n_output_targets(),
                height, width)
# We only log the structure when the net is used in the training script
net.log_structure = True

try:
    transformation_cls = getattr(models.transforms, transformation_cls_name)
    transformation = transformation_cls()
    # TODO use the property here
    net._final_transformation = transformation
except AttributeError as e:
    raise type(e)('Could not find the specified transformation class: ' +
                  str(e))
print('--------------------')
print(net)
print('--------------------')
print('***')
# To GPU
net.to(device)

# Log the text representation of the net into a txt artifact
with open(os.path.join(data_location, models_directory,
                       'nn_architecture.txt'), 'w') as f:
    print('Writing neural net architecture into txt file.')
    f.write(str(net))
# FIN NEURAL NETWORK -


# Training------------
# MSE criterion + Adam optimizer
criterion = getattr(train.losses, loss_cls_name)()

# metrics saved independently of the training criterion
metrics = {'mse': F.mse_loss}

conv_layers = net.conv_layers
params = [{'params': layer.parameters()} for layer in conv_layers]
linear_layer = net.linear_layer
if linear_layer is not None:
    params.append({'params': linear_layer.parameters(),
                   'weight_decay': weight_decay,
                   'lr': learning_rates[0] / 100})
optimizers = {i: optim.Adam(params, lr=v, weight_decay=0.0)
              for (i, v) in learning_rates.items()}

trainer = Trainer(net, device)
trainer.criterion = criterion
trainer.print_loss_every = print_loss_every
for metric_name, metric_func in metrics.items():
    trainer.register_metric(metric_name, metric_func)

for i_epoch in range(n_epochs):
    # Set to training mode
    if i_epoch in optimizers:
        optimizer = optimizers[i_epoch]
        print('Switching to new optimizer:\n', optimizer)
    print('Epoch number {}.'.format(i_epoch))
    # TODO remove clipping?
    train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer,
                                             clip=None)
    test_loss, metrics_results = trainer.test(test_dataloader)
    # Log the training loss
    print('Train loss for this epoch is ', train_loss)
    print('Test loss for this epoch is ', test_loss)
    for metric_name, metric_value in metrics_results:
        print('Test {} for this epoch is {}'.format(metric_name, metric_value))
    mlflow.log_metric('train loss', train_loss, i_epoch)
    mlflow.log_metric('test loss', test_loss, i_epoch)
    mlflow.log_metrics(metrics_results)
        

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


# FIN TRAINING ----------------------------------------------------------------

# Save the trained model to disk
print('Moving the network to the CPU before saving...')
net.cpu()
print('Saving the neural network learnt parameters to disk...')
full_path = os.path.join(data_location, models_directory, model_name)
torch.save(net.state_dict(), full_path)
print('Logging the neural network model...')
print('Neural network saved and logged in the artifacts.')
net.cuda(device)

# Save other parts of the model
print('Saving other parts of the model')
full_path = os.path.join(data_location, models_directory, 'data_transform')
with open(full_path, 'wb') as f:
    pickle.dump(data_transform, f)
full_path = os.path.join(data_location, models_directory, 'transformation')
with open(full_path, 'wb') as f:
    pickle.dump(transformation, f)

# DEBUT TEST ------------------------------------------------------------------

# CORRELATION MAP ----
u_v_surf = np.zeros((len(test_dataset), 2, dataset.height, dataset.width))
pred = np.zeros((len(test_dataset), 4, dataset.height, dataset.width))
truth = np.zeros((len(test_dataset), 2, dataset.height, dataset.width))

# Predictions on the test set using the trained model
net.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        u_v_surf[i * batch_size: (i+1) * batch_size] = data[0].numpy()
        truth[i * batch_size:(i+1) * batch_size] = data[1].numpy()
        X = data[0].to(device, dtype=torch.float)
        Y_hat = net(X)
        Y_hat = Y_hat.cpu().numpy()
        pred[i * batch_size:(i+1) * batch_size] = Y_hat
# Convert to dataset
new_dims = ('time', 'latitude', 'longitude')
coords = xr_dataset.coords
new_coords = {'time': coords['time'][test_index:],
              'latitude': coords['yu_ocean'].data,
              'longitude': coords['xu_ocean'].data}
u_surf = xr.DataArray(data=u_v_surf[:, 0, ...], dims=new_dims,
                      coords=new_coords)
v_surf = xr.DataArray(data=u_v_surf[:, 1, ...], dims=new_dims,
                      coords=new_coords)
s_x = xr.DataArray(data=truth[:, 0, ...], dims=new_dims, coords=new_coords)
s_y = xr.DataArray(data=truth[:, 1, ...], dims=new_dims, coords=new_coords)
s_x_pred = xr.DataArray(data=pred[:, 0, ...], dims=new_dims, coords=new_coords)
s_y_pred = xr.DataArray(data=pred[:, 1, ...], dims=new_dims, coords=new_coords)
s_x_pred_scale = xr.DataArray(data=pred[:, 2, ...], dims=new_dims,
                              coords=new_coords)
s_y_pred_scale = xr.DataArray(data=pred[:, 3, ...], dims=new_dims,
                              coords=new_coords)
output_dataset = xr.Dataset({'u_surf': u_surf, 'v_surf': v_surf,
                             'S_x': s_x, 'S_y': s_y,
                             'S_xpred': s_x_pred,
                             'S_xpred_scale': s_x_pred_scale,
                             'S_ypred_scale': s_y_pred_scale,
                             'S_ypred': s_y_pred})

# Save model output on the test dataset
output_dataset.to_zarr(os.path.join(data_location, model_output_dir,
                                    'test_output'))

# Correlation map, shape (2, dataset.width, dataset.height)
pred = pred[:, :2, ...]
correlation_map = np.mean(truth * pred, axis=0)
correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
correlation_map /= np.maximum(np.std(truth, axis=0) * np.std(pred, axis=0),
                              1e-20)

print('Saving correlation map to disk')
# Save the correlation map to disk and its plot as well.
np.save(os.path.join(data_location, model_output_dir, 'correlation_map'),
        correlation_map)

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
mlflow.log_artifact(os.path.join(data_location, figures_directory))
mlflow.log_artifact(os.path.join(data_location, model_output_dir))
mlflow.log_artifact(os.path.join(data_location, models_directory))
print('Done...')
