# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:00:45 2020

@author: Arthur
"""
import numpy as np
import mlflow
from  mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import sys
from analysis.analysis import TimeSeriesForPoint
import xarray as xr

from enum import Enum



def correlation_map(truth: np.ndarray, pred: np.ndarray):
    """
    Return the correlation map.

    Parameters
    ----------
    truth : np.ndarray
        True values.
    pred : np.ndarray
        Predicted values

    Returns
    -------
    correlation_map : np.ndarray
        Correlation between true and predictions.

    """

    correlation_map = np.mean(truth * pred, axis=0)
    correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
    correlation_map /= np.std(truth, axis=0) * np.std(pred, axis=0)
    return correlation_map


def rmse_map(targets: np.ndarray, predictions: np.ndarray,
             normalized: bool = False):
    """Computes the rmse of the prediction time series at each point."""
    error = predictions - targets
    stds = np.std(targets, axis = 0)
    if normalized:
        stds = np.clip(stds, np.min(stds[stds > 0]), np.inf)
    else:
        stds = 1
    rmse_map = np.sqrt(np.mean(np.power(error, 2), axis=0)) / stds
    return rmse_map


def select_experiment():
    """
    Prompt user to select an experiment among all experiments in store. Return
    the name of the selected experiment.

    Returns
    -------
    str
        Name of the experiment selected by the user.

    """
    client = MlflowClient()
    list_of_exp = client.list_experiments()
    dict_of_exp = {exp.experiment_id: exp.name for exp in list_of_exp}
    for id_, name in dict_of_exp.items():
        print(id_, ': ', name)
    selection = input('Select the id of an experiment: ')
    return dict_of_exp[selection]


# def merge_predictions_and_data(predictions: xr.Dataset, 


def select_run(sort_by=None, cols=None, merge=None, *args, **kargs):
    """
    Allows to select a run from the tracking store interactively.

    Parameters
    ----------
    sort_by : str, optional
        Name of the column used for sorting the returned runs.
        The default is None.
    cols : list[str], optional
        List of column names printed to user. The default is None.
    merge : list of length-3 tuples, optional
        Describe how to merge information with other experiments.
        Each element of the list is a tuple 
        (experiment_name, key_left, key_right), according to which the 
        initial dataframe of runs will be merged with that corresponding
        to experiment_name, using key_left (from the first dataframe) and
        key_right (from the second dataframe).
    *args : list
        List of args passed on to mlflow.search_runs.
    **kargs : dictionary
        Dictionary of args passed on to mlflow.search_runs. In particular
        one may want to specify experiment_ids to select runs from a given
        list of experiments.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    pandas.Series
        Series describing the interactively selected run.

    """
    mlflow_runs = mlflow.search_runs(*args, **kargs)
    if cols is None:
        cols = list()
    cols = ['run_id', 'experiment_id'] + cols
    if sort_by is not None:
        mlflow_runs.sort_values(by=sort_by)
        cols.append(sort_by)
    # Remove possible duplicate columns
    new_cols = list()
    for e in cols:
        if e not in new_cols:
            new_cols.append(e)
    cols = new_cols
    if merge is not None:
        cols[cols.index('run_id')] = 'run_id_x'
        cols[cols.index('experiment_id')] = 'experiment_id_x'
        for name, key_left, key_right in merge:
            experiment = mlflow.get_experiment_by_name(name)
            df2 = mlflow.search_runs(experiment_ids=experiment.experiment_id)
            mlflow_runs = pd.merge(mlflow_runs, df2, left_on=key_left,
                                   right_on=key_right)
    if len(mlflow_runs) == 0:
        raise Exception('No data found. Check that you correctly set \
                        the store')
    print(mlflow_runs[cols])
    id_ = int(input('Run id?'))
    if id_ < 0:
        return 0
    return mlflow_runs.loc[id_, :]


class DisplayMode(Enum):
    """Enumeration of the different display modes for viewing methods"""
    correlation = correlation_map
    rmse = rmse_map
    difference = lambda x, y: np.mean(x-y, axis=0)


def view_predictions(predictions: np.ndarray, targets: np.ndarray,
                     display_mode=DisplayMode.correlation):
    """Plots the correlation map for the passed predictions and targets.
    On clicking a point on the correlation map, the time series of targets
    and predictions at that point are shown in a new plot for further
    analysis."""
    # Compute the correlation map
    map_ = display_mode(targets, predictions)
    fig = plt.figure()
    plt.imshow(map_, origin='lower')
    plt.colorbar()
    plt.show()

    def onClick(event):
        time_series0 = TimeSeriesForPoint(predictions=predictions,
                                          truth=targets)
        time_series0.point = (int(event.xdata), int(event.ydata))
        time_series0.plot_pred_vs_true()
    fig.canvas.mpl_connect('button_press_event', onClick)


def sample(data : np.ndarray, step_time : int = 1, nb_per_time: int = 5):
    """Samples points from the data, where it is assumed that the data
    is 4-D, with the first dimension representing time , the second
    the channel, and the others representing spatial dimensions. 
    The sampling is done for every step_time image, and for each image 
    nb_per_time points are randomly selected.

    Parameters
    ----------
    
    :data: ndarray, (n_time, n_channels, n_x, n_y)
        The time series of images to sample from.
    
    :step_time: int,
        The distance in time between two consecutive images used for the 
        sampling.

    :nb_per_time: int,
        Number of points used (chosen randomly according to a uniform 
        distribution over the spatial domain) for each image.
    

    Returns
    -------
    :sample: ndarray, (n_time / step_time, n_channels, nb_per_time )
        The sampled data.
    """
    if data.ndim != 4:
        raise ValueError('The data is expected to have 4 dimensions.')
    n_times, n_channels, n_x, n_y = data.shape
    time_indices = np.arange(0, n_times, step_time)
    x_indices = np.random.randint(0, n_x,
                                  (time_indices.shape[0], 2, nb_per_time))
    y_indices = np.random.randint(0, n_y,
                                  (time_indices.shape[0], 2, nb_per_time))
    channel_indices = np.zeros_like(x_indices)
    channel_indices[:, 1, :] = 1
    time_indices = time_indices.reshape((-1, 1, 1))
    time_indices = time_indices.repeat(2, axis=1)
    time_indices = time_indices.repeat(nb_per_time, axis=2)

    selection = time_indices, channel_indices, x_indices, y_indices
    sample = data[selection]
    return sample


def plot_dataset(dataset: xr.Dataset, plot_type = None, *args, **kargs):
    """
    Calls the plot function of each variable in the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset whose variables we wish to plot.
    plot_type : str, optional
        Plot type used for each variable in the dataset. The default is None.
    *args : list
        List of args passed on to the plot function.
    **kargs : dictionary
        Dictionary of args passed on to the plot function.

    Returns
    -------
    None.

    """
    plt.figure(figsize = (20, 5 * int(len(dataset) / 2)))
    kargs_ = [dict() for i in range(len(dataset))]
    def process_list_of_args(name: str):
        if name in kargs:
            if isinstance(kargs[name], list):
                for i, arg_value in enumerate(kargs[name]):
                    kargs_[i][name] = arg_value
            else:
                for i in range(len(dataset)):
                    kargs_[i][name] = kargs[name]
            kargs.pop(name)
    process_list_of_args('vmin')
    process_list_of_args('vmax')
    for i, variable in enumerate(dataset):
        plt.subplot(int(len(dataset) / 2), 2, i + 1)
        if plot_type is None:
            try:
                # By default we set the cmap to coolwarm
                kargs.setdefault('cmap', 'coolwarm')
                dataset[variable].plot(*args, **kargs_[i], **kargs)
            except AttributeError as e:
                kargs.pop('cmap', None)
                dataset[variable].plot(*args, **kargs)
        else:
            plt_func = getattr(dataset[variable].plot, plot_type)
            plt_func(*args, **kargs)
import matplotlib.animation as animation

def dataset_to_movie(dataset : xr.Dataset, interval : int = 50,
                    *args, **kargs):
    """Generates animations for all the variables in the dataset"""
    fig = plt.figure(figsize = (20, 5 * int(len(dataset) / 2)))
    axes = list()
    ims = list()
    for i, variable in enumerate(dataset.keys()):
        axes.append(fig.add_subplot(int(len(dataset) / 2), 2, i + 1))
    for i, t in enumerate(dataset['time']):
        im = list()
        for axis, variable in zip(axes, dataset.keys()):
            plt.sca(axis)
            img = dataset[variable].isel(time=i).plot(vmin=-2, vmax=2,
                                                      cmap='coolwarm')
            cb = img.colorbar
            cb.remove()
            im.append(img)
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, 
                                    interval=interval, blit=True,
                                    repeat_delay=1000)
    return ani


def play_movie(predictions: np.ndarray, title: str = '',
               interval: int = 500):
    fig = plt.figure()
    ims = list()
    mean = np.mean(predictions)
    std = np.std(predictions)
    vmin, vmax = mean - std, mean + std
    for im in predictions:
        ims.append([plt.imshow(im, vmin=vmin, vmax=vmax,
                               cmap='YlOrRd',
                               origin='lower', animated=True)])
    ani = animation.ArtistAnimation(fig, ims, 
                                    interval=interval, blit=True,
                                    repeat_delay=1000)
    plt.title(title)
    plt.show()
    return ani
