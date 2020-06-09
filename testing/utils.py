#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:58:33 2020

@author: arthur
"""
import numpy as np
import xarray as xr
import torch

# TODO correct coordinates

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
        for i, data in enumerate(test_dataloader):
            print(i)
            uv_data = data[0][:, :2, ...].numpy()
            velocities[i * batch_size: (i + 1) * batch_size] = uv_data
            truth[i * batch_size: (i + 1) * batch_size] = data[1].numpy()
            X = data[0].to(device, dtype=torch.float)
            pred_i = net(X)
            pred_i = pred_i.cpu().numpy()
            predictions[i * batch_size: (i+1) * batch_size] = pred_i

    # Put this into an xarray dataset before saving
    new_dims = ('time', 'latitude', 'longitude')
    coords = xr_dataset.coords
    coords_uv = {'time': coords['time']
                      [test_index:test_index+len(test_dataset)],
                      'latitude': coords['yu_ocean'].data[:test_dataset.height],
                      'longitude': coords['xu_ocean'].data[:test_dataset.width]}
    coords_s = {'time': coords['time']
                  [test_index:test_index+len(test_dataset)],
                  'latitude': coords['yu_ocean'].data[:test_dataset.output_height],
                  'longitude': coords['xu_ocean'].data[:test_dataset.output_width]}
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
    output_dataset = xr.Dataset({'u_surf': u_surf, 'v_surf': v_surf,
                                 'S_x': s_x, 'S_y': s_y, 'S_xpred': s_x_pred,
                                 'S_ypred': s_y_pred, 'S_xscale': s_x_pred_scale,
                                 'S_yscale': s_y_pred_scale})
    return output_dataset