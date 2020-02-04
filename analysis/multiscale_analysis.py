# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:04:37 2020

@author: Arthur
Script to analyze the outputs from the 'multiscale' experiment, corresponding
to the script testing/multiscale.py.
"""

from loadmlflow import LoadMLFlow
from analysis import TimeSeriesForPoint
from utils import select_run, correlation_map
import mlflow
from matplotlib import pyplot as plt

# We'll run this locally
mlflow.set_tracking_uri('file:///d:\\Data sets\\NYU\\mlruns')

# Setting the experiment
mlflow.set_experiment('multiscale')

# Select a run and load the predictions and targets for that id.
run_id, experiment_id = select_run()
loader = LoadMLFlow(run_id, experiment_id, 
                    'd:\\Data sets\\NYU\\mlruns')
predictions = loader.predictions[:1000, ...]
targets = loader.true_targets[:1000, ...] * 10

# Compute the correlation map
correlation_map = correlation_map(predictions, targets)

fig = plt.figure()
plt.imshow(correlation_map[0, ...])
plt.colorbar()


def onClick(event):
    time_series0 = TimeSeriesForPoint(predictions=predictions, 
                                      truth=targets)
    time_series0.point = (int(event.xdata), int(event.ydata))
    time_series0.plot_pred_vs_true()


fig.canvas.mpl_connect('button_press_event', onClick)