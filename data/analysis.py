# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:12:36 2019

@author: Arthur
In this file we analyse the predictions given by the algorithm.
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import os.path

# Location of data
filepath_predictions = '/data/ag7531/predictions/'
filepath_targets = '/data/ag7531/targets/'
filepath_analysis = '/data/ag7531/analysis/'

# Load the predictions
print('Loading the predictions...')
predictions = np.load(os.path.join(filepath_predictions, 'predictionsRF.npy'))

#Load the targets
print('Loading the true targets...')
targets = np.load(os.path.join(filepath_targets, 'targets.npy'))

#Reshape the data for plots
print('Reshape data...')
with open(os.path.join(filepath_targets, 'config.txt')) as f:
    width = int(f.readline())
    height = int(f.readline())

predictions_ = dict()
targets_ = dict()
size = int(targets.shape[1] / 2)
predictions_['sx'] = predictions[:, :size].reshape(-1, width, height)
predictions_['sy'] = predictions[:, size:].reshape(-1, width, height)
targets_['sx'] = targets[:, :size].reshape(-1, width, height)
targets_['sy'] = targets[:, size:].reshape(-1, width, height)

predictions = predictions_['sx']
targets = targets_['sx']


#Distribution of the error
print('Computing properties of the prediction error...')
error = predictions - targets
mean_error = np.mean(error, axis=1)
std_error = np.std(error, axis=1)

relative_error = error / targets
mean_relative_error = np.mean(relative_error, axis=1)
std_relative_error = np.std(relative_error, axis=1)

with open(os.path.join(filepath_analysis, 'report.txt'), 'w') as f:
    f.write('Report....)

print('Generating snapshots...')
for i in numpy.random.randint(0, targets.shape[0], 50):
    print('.', end='')
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(targets[i, ...], cmap='jet', origin='lower', vmin=-2,
               vmax=2)
    plt.colorbar()
    plt.title('True')
    plt.subplot(122)
    plt.imshow(predictions[i, ...], cmap='jet', origin='lower', vmin=-2,
               vmax=2)
    plt.colorbar()
    plt.title('Predicted')
    plt.savefig(os.path.join(filepath_analysis, 'plot{}'.format(i)))
    plt.close(fig)

print('Generating plots for the error')
fig = plt.figure()
plt.subplot(211)
plt.imshow(mean_error, cmap='jet', origin='lower', vmin='-4', vmax='4')
plt.title('Mean')
plt.subplot(212)
plt.imshow(std_error, cmap='jet', origin='lower', vmin='-4', vmax='4')
plt.title('Std')
plt.suptitle('Error distribution properties')
plt.savefig(os.path.join(filepath_analysis, 'error'))
plt.close(fig)

fig = plt.figure()
plt.subplot(211)
plt.imshow(mean_relative_error, cmap='jet', origin='lower')
plt.title('Mean')
plt.subplot(212)
plt.imshow(std_relative_error, cmap='jet', origin='lower')
plt.title('Std')
plt.suptitle('Relative Error distribution properties')
plt.savefig(os.path.join(filepath_analysis, 'relative_error'))
plt.close(fig)
print('All done...')
