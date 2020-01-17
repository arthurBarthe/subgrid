# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:13:22 2019

@author: Arthur
"""

import numpy as np
import matplotlib.pyplot as plt
from BoltonEtAl._utils import *
import os.path


READ_DATA = True

# Dataset location
filepath = 'D:\\Data sets\\NYU\\QG Data for Eugene-20191121T200122Z-002'
filepath_new_data = 'D:\\Data sets\\NYU\\processed_data'
filename = 'out513b-600-75-8_psi1a.mat'

# Dynamic model parameters
# grid steps (m)
dx = 7.5e3
dy = dx
# scale of the blurring (used when applying a Gaussian filter to the high
# resolution data)
scale = 30e3

# Read the data
data = load_tom_data(filepath, filename)

# For some reason the sample index is last in the stored data.
data = np.swapaxes(data, 0, 2)
n_times = data.shape[0]
print('nb of samples: {}'.format(n_times))
length_x, length_y = data.shape[1:]
print(data.shape[1:])

# Processing of the data
n_times_per_loop = 25
shape_result = (n_times, 3, length_x // 4, length_y // 4)
func = lambda x: build_training_data(x, dx, dy, scale)
processed_data = block_loop(n_times, n_times_per_loop, data, 
                            func, shape_result)
psi_coarse = processed_data[:, 0, :, :]
sx_coarse = processed_data[:, 1, :, :]
sy_coarse = processed_data[:, 2, :, :]

# Saving to disk

np.save(os.path.join(filepath_new_data, 'psi_coarse'), psi_coarse)
np.save(os.path.join(filepath_new_data, 'sx_coarse'), sx_coarse)
np.save(os.path.join(filepath_new_data, 'sy_coarse'), sy_coarse)
