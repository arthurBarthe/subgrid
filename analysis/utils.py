# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:00:45 2020

@author: Arthur
"""
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import pandas as pd

from analysis.analysis import TimeSeriesForPoint

from enum import Enum


def correlation_map(truth, pred):
    """Computes the correlation at each point of the domain between the
    truth and the prediction."""
    correlation_map = np.mean(truth * pred, axis=0)
    correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
    correlation_map /= np.std(truth, axis=0) * np.std(pred, axis=0)
    return correlation_map


def rmse_map(targets: np.ndarray, predictions: np.ndarray):
    """Computes the rmse of the prediction time series at each point"""
    error = predictions - targets
    stds = np.std(targets, axis = 0)
    rmse_map = np.sqrt(np.mean(np.power(error, 2), axis=0)) / stds
    return rmse_map


def select_run(sort_by=None, cols=None, merge=None):
#    if not hasattr(sys.modules['__main__'], 'mlflow_runs'):
    mlflow_runs = mlflow.search_runs()
    if cols is None:
        cols = list()
    cols = ['run_id', 'experiment_id' ] + cols
    if sort_by is not None:
        mlflow_runs.sort_values(by=sort_by)
        cols.append(sort_by)
    if merge is not None:
        cols[0] = 'run_id_x'
        cols[1] = 'experiment_id_x'
        name, key_left, key_right = merge
        experiment = mlflow.get_experiment_by_name(name)
        df2 = mlflow.search_runs(experiment_ids=experiment.experiment_id)
        mlflow_runs = pd.merge(mlflow_runs, df2, left_on=key_left,
                               right_on=key_right, how='right')
        print(mlflow_runs)
    print(mlflow_runs[cols])
    id_ = int(input('Run id?'))
    return mlflow_runs.loc[id_, cols[:2]]


class DisplayMode(Enum):
    """Enumeration of the different display modes for viewing methods"""
    correlation = correlation_map
    rmse = rmse_map

def view_predictions(predictions: np.ndarray, targets: np.ndarray,
                     display_mode=DisplayMode.correlation):
    """Plots the correlation map for the passed predictions and targets.
    On clicking a point on the correlation map, the time series of targets
    and predictions at that point are shown in a new plot for further
    analysis."""
    # Compute the correlation map
    map_ = display_mode(predictions, targets)
    fig = plt.figure()
    plt.imshow(map_[0, ...], origin='lower')
    plt.colorbar()

    def onClick(event):
        time_series0 = TimeSeriesForPoint(predictions=predictions,
                                          truth=targets)
        time_series0.point = (int(event.xdata), int(event.ydata))
        time_series0.plot_pred_vs_true()
    fig.canvas.mpl_connect('button_press_event', onClick)
