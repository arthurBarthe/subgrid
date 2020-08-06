#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:58:33 2020

@author: arthur
"""
import numpy as np
import xarray as xr
import torch
import progressbar
import mlflow
import pickle

# Use dask for large datasets that won't fit in RAM

import dask.array as da
import dask


def apply_net(net, test_dataloader, device):
    """
    Return the predictions obtained by applying the provided NN on the data
    provided by the data loader.

    Parameters
    ----------
    net : torch.nn.Module
        Neural Network used to make the predictions.
    test_dataloader : torch.utils.data.DataLoader
        Loader providing the mini-batches.
    device : torch.device
        Device on which to put the data.

    Returns
    -------
    output : list
        List of predictions, with each element corresponding to a mini-batch.

    """
    """Return an object that applies the net on the provided
    DataLoader"""
    input_ = []
    output = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            features, targets = data
            input_.append(features)
            features = features.to(device, dtype=torch.float)
            prediction = (net(features)).cpu().numpy()
            output.append(prediction)
    return input_, output


def _dataset_from_channels(array, channels_names: list, dims, coords):
    """
    Return a dataset where variables are obtained by assigning a name to
    different channels of the passed numpy/dask array.

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    channels_names : list
        DESCRIPTION.
    dims : list
        List of dimensions of the resulting dataset.
    coords : list
        List of coordinates of the resulting dataset.

    Returns
    -------
    xarray.Dataset
        Dataset.
    """
    data_arrays = [xr.DataArray(array[:, i, ...], dims=dims, coords=coords)
                   for i in range(len(channels_names))]
    data = {name: d_array for (name, d_array) in zip(channels_names,
                                                     data_arrays)}
    return xr.Dataset(data)


def create_large_test_dataset(net, test_datasets, test_loaders, device):
    """
    Return an xarray dataset with the predictions carried out on the
    provided test datasets. The data of this dataset are dask arrays,
    therefore postponing computations (for instance to the time of writing
    to disk). This means that if we correctly split an initial large test
    dataset into smaller test datasets, each of which fits in RAM, there
    should be no issue.

    Parameters
    ----------
    net : torch.nn.Module
        Neural net used to make predictions
    test_datasets : list
        List of PytTorch datasets containing the input data.
    test_loaders : list
        List of Pytorch DataLoaders corresponding to the datasets
    device : torch.device
        Device on which to put the tensors.

    Returns
    -------
    xarray.Dataset
        Dataset of predictions.

    """
    inputs = []
    outputs = []
    for i, loader in enumerate(test_loaders):
        test_dataset = test_datasets[i]
        delayed_apply = dask.delayed(apply_net)
        temp = delayed_apply(net, loader, device)
        shape = (loader.batch_size, 4, test_dataset.output_height,
                 test_dataset.output_width)
        output = [da.from_delayed(temp[1][i], shape=shape, dtype=np.float64)
                  for i in range(len(loader))]
        output = da.concatenate(output)
        # Same for input
        shape = (loader.batch_size, 2, test_dataset.height, test_dataset.width)
        input_ = [da.from_delayed(temp[0][i], shape=shape, dtype=np.float64)
                  for i in range(len(loader))]
        input_ = da.concatenate(input_)
        # Now we make a proper dataset out of the dask array
        new_dims = ('time', 'latitude', 'longitude')
        coords_s = test_dataset.output_coords
        coords_s['latitude'] = coords_s.pop('yu_ocean')
        coords_s['longitude'] = coords_s.pop('xu_ocean')
        var_names = ['S_xpred', 'S_ypred', 'S_xscale', 'S_yscale']
        output_dataset = _dataset_from_channels(output, var_names, new_dims,
                                                coords_s)
        # same for input
        coords_uv = test_dataset.input_coords
        coords_uv['latitude'] = coords_uv.pop('yu_ocean')
        coords_uv['longitude'] = coords_uv.pop('xu_ocean')
        var_names = ['usurf', 'vsurf']
        input_dataset = _dataset_from_channels(input_, var_names, new_dims,
                                               coords_uv)
        outputs.append(output_dataset)
        inputs.append(input_dataset)
    return xr.merge((xr.concat(outputs, 'time'), xr.concat(inputs,'time')))


def create_test_dataset(net, xr_dataset, test_dataset, test_dataloader,
                        test_index, device):
    velocities = np.zeros((len(test_dataset), 2, test_dataset.height,
                           test_dataset.width))
    predictions = np.zeros((len(test_dataset), 4, test_dataset.output_height,
                            test_dataset.output_width))
    truth = np.zeros((len(test_dataset), 2, test_dataset.output_height,
                      test_dataset.output_width))
    batch_size = test_dataloader.batch_size
    net.eval()
    with torch.no_grad():
        with progressbar.ProgressBar(max_value=len(test_dataset)//batch_size) as bar:
            for i, data in enumerate(test_dataloader):
                uv_data = data[0][:, :2, ...].numpy()
                velocities[i * batch_size: (i + 1) * batch_size] = uv_data
                truth[i * batch_size: (i + 1) * batch_size] = data[1].numpy()
                X = data[0].to(device, dtype=torch.float)
                pred_i = net(X)
                pred_i = pred_i.cpu().numpy()
                predictions[i * batch_size: (i+1) * batch_size] = pred_i
                bar.update(i)

    # Put this into an xarray dataset before saving
    new_dims = ('time', 'latitude', 'longitude')
    coords_uv = test_dataset.input_coords
    coords_s = test_dataset.output_coords
    # Rename to latitude and longitude
    coords_uv['latitude'] = coords_uv.pop('yu_ocean')
    coords_uv['longitude'] = coords_uv.pop('xu_ocean')
    coords_s['latitude'] = coords_s.pop('yu_ocean')
    coords_s['longitude'] = coords_s.pop('xu_ocean')
    # Create data arrays from numpy arrays
    u_surf = xr.DataArray(data=velocities[:, 0, ...], dims=new_dims,
                          coords=coords_uv)
    v_surf = xr.DataArray(data=velocities[:, 1, ...], dims=new_dims,
                          coords=coords_uv)
    s_x = xr.DataArray(data=truth[:, 0, ...], dims=new_dims, coords=coords_s)
    s_y = xr.DataArray(data=truth[:, 1, ...], dims=new_dims, coords=coords_s)
    s_x_pred = xr.DataArray(data=predictions[:, 0, ...], dims=new_dims,
                            coords=coords_s)
    s_y_pred = xr.DataArray(data=predictions[:, 1, ...], dims=new_dims,
                            coords=coords_s)
    s_x_pred_scale = xr.DataArray(data=predictions[:, 2, ...], dims=new_dims,
                                  coords=coords_s)
    s_y_pred_scale = xr.DataArray(data=predictions[:, 3, ...], dims=new_dims,
                                  coords=coords_s)
    # Create dataset from data arrays
    output_dataset = xr.Dataset({'u_surf': u_surf, 'v_surf': v_surf,
                                 'S_x': s_x, 'S_y': s_y, 'S_xpred': s_x_pred,
                                 'S_ypred': s_y_pred, 'S_xscale': s_x_pred_scale,
                                 'S_yscale': s_y_pred_scale})
    return output_dataset



def pickle_artifact(run_id: str, path: str):
    client = mlflow.tracking.MlflowClient()
    file = client.download_artifacts(run_id, path)
    f = open(file, 'rb')
    return pickle.load(f)