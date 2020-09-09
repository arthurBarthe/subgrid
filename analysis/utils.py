# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:00:45 2020

@author: Arthur
"""
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pandas as pd
from analysis.analysis import TimeSeriesForPoint
import xarray as xr
from scipy.ndimage import gaussian_filter
from data.pangeo_catalog import get_whole_data
from cartopy.crs import Projection, PlateCarree


from enum import Enum

CATALOG_URL = 'https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml'


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
    stds = np.std(targets, axis=0)
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


def select_run(sort_by=None, cols=None, merge=None, *args, **kargs) -> object:
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

    def diff_func(x, y):
        return np.mean(x - y, axis=0)
    difference = diff_func


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


def sample(data: np.ndarray, step_time: int = 1, nb_per_time: int = 5):
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


def plot_dataset(dataset: xr.Dataset, plot_type=None, *args, **kargs):
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
    plt.figure(figsize=(20, 5 * int(len(dataset) / 2)))
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
            except AttributeError:
                kargs.pop('cmap', None)
                dataset[variable].plot(*args, **kargs)
        else:
            plt_func = getattr(dataset[variable].plot, plot_type)
            plt_func(*args, **kargs)


def dataset_to_movie(dataset: xr.Dataset, interval: int = 50, *args, **kargs):
    """
    Generates animations for all the variables in the dataset

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset used to generate movie. Must contain dimension 'time'.
    interval : int, optional
        Interval between frames in milliseconds. The default is 50.
    *args : list
        Positional args passed on to plot function.
    **kargs : dictionary
        keyword args passed on to plot function.

    Returns
    -------
    ani : TYPE
        Movie animation.

    """
    fig = plt.figure(figsize=(20, 5 * int(len(dataset) / 2)))
    axes = list()
    ims = list()
    for i, variable in enumerate(dataset.keys()):
        axes.append(fig.add_subplot(int(len(dataset) / 2), 2, i + 1))
    for i, t in enumerate(dataset['time']):
        im = list()
        for axis, variable in zip(axes, dataset.keys()):
            plt.sca(axis)
            img = dataset[variable].isel(time=i).plot(*args, **kargs)
            cb = img.colorbar
            cb.remove()
            im.append(img)
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
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
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=1000)
    plt.title(title)
    plt.show()
    return ani


class GlobalPlotter:
    """General class to make plots for global data. Handles masking of
    continental data + showing a band near coastlines."""

    def __init__(self, margin: int = 10, cbar: bool = True):
        self.mask = self._get_global_u_mask()
        self.margin = margin
        self.cbar = cbar
        self.ticks = dict(x=None, y=None)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def borders(self):
        return self._borders

    @borders.setter
    def borders(self, value):
        self._borders = value

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, margin):
        self._margin = margin
        self.borders = self._get_continent_borders(self.mask, self.margin)

    @property
    def x_ticks(self):
        return self.ticks['x']

    @x_ticks.setter
    def x_ticks(self, value):
        self.ticks['x'] = value

    @property
    def y_ticks(self):
        return self.ticks['y']

    @y_ticks.setter
    def y_ticks(self, value):
        self.ticks['y'] = value

    def plot(self, u: xr.DataArray=None, projection_cls=PlateCarree,
             lon: float = -100.0, lat: float = None, ax=None, animated=False,
             **plot_func_kw):
        """
        Plots the passed velocity component on a map, using the specified
        projection. Uses the instance's mask to set as nan some values.

        Parameters
        ----------
        u : xr.DataArray
            Velocity array. The default is None.
        projection : Projection
            Projection used for the 2D plot.
        lon : float, optional
            Central longitude. The default is -100.0.
        lat : float, optional
            Central latitude. The default is None.

        Returns
        -------
        None.

        """
        fig = plt.figure()
        projection = projection_cls(lon)
        if ax is None:
            ax = plt.axes(projection=projection)
        mesh_x, mesh_y = np.meshgrid(u['longitude'], u['latitude'])
        if u is not None:
            mask = self.mask.interp({k: u.coords[k] for k in ('longitude',
                                                              'latitude')})
            u = u * mask
            im = ax.pcolormesh(mesh_x, mesh_y, u.values,
                               transform=PlateCarree(),
                               animated=animated, **plot_func_kw)
            if self.cbar:
                fig.colorbar(im)
        if self.x_ticks is not None:
            ax.set_xticks(self.x_ticks)
        if self.y_ticks is not None:
            ax.set_yticks(self.y_ticks)
        ax.set_global()
        ax.coastlines()
        if self.margin > 0:
            borders = self.borders.interp({k: u.coords[k]
                                           for k in ('longitude', 'latitude')})
            ax.pcolormesh(mesh_x, mesh_y, borders, animated=animated,
                          transform=PlateCarree(), alpha=0.1)
        return ax

    @staticmethod
    def _get_global_u_mask(factor: int = 4, base_mask: xr.DataArray = None):
        """
        Return the global mask of the low-resolution surface velocities for
        plots. While the coarse-grained velocities might be defined on
        continental points due to the coarse-graining procedures, these are
        not shown as we do not use them -- the mask for the forcing is even
        more restrictive, as it removes any point within some margin of the
        velocities mask.

        Parameters
        ----------
        factor : int, optional
            Coarse-graining factor. The default is 4.

        base_mask: xr.DataArray, optional
            # TODO
            Not implemented for now.

        Returns
        -------
        None.

        """
        if base_mask is not None:
            mask = base_mask
        else:
            _, grid_info = get_whole_data(CATALOG_URL, 0)
            mask = grid_info['wet']
            mask = mask.coarsen(dict(xt_ocean=factor, yt_ocean=factor))
        mask_ = mask.max()
        mask_ = mask_.where(mask_ > 0.1)
        mask_ = mask_.rename(dict(xt_ocean='longitude', yt_ocean='latitude'))
        return mask_.compute()

    @staticmethod
    def _get_continent_borders(base_mask: xr.DataArray, margin: int):
        """
        Returns a boolean xarray DataArray corresponding to a mask of the
        continents' coasts, which we do not process.
        Hence margin should be set according to the model.

        Parameters
        ----------
        mask : xr.DataArray
            Mask taking value 1 where coarse velocities are defined and used
            as input and nan elsewhere.
        margin : int
            Margin imposed by the model used, i.e. number of points lost on
            one side of a square.

        Returns
        -------
        mask : xr.DataArray
            Boolean DataArray taking value True for continents.

        """
        assert margin >= 0, 'The margin parameter should be a non-negative' \
                            ' integer'
        assert base_mask.ndim <= 2, 'Velocity array should have two'\
                                    ' dims'
        # Small trick using the guassian filter function
        mask = xr.apply_ufunc(lambda x: gaussian_filter(x, 1., truncate=margin),
                              base_mask)
        mask = np.logical_and(np.isnan(mask),  ~np.isnan(base_mask))
        mask = mask.where(mask)
        return mask.compute()


def plot_training_subdomains(run_id, global_plotter: GlobalPlotter, alpha=0.5,
                             bg_variable=None, facecolor='blue',
                             edgecolor=None, *plot_args, **plot_kwd_args):
    """
    Plots the training subdomains used for a given training run. Retrieves
    those subdomains from the run's parameters.

    Parameters
    ----------
    run_id : str
        Id of the training run.
    global_plotter : GlobalPlotter
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.5.
    facecolor : TYPE, optional
        DESCRIPTION. The default is 'blue'.
    edgecolor : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # First retrieve the run's data
    run = mlflow.get_run(run_id)
    run_params = run.data.params
    data_ids = run_params['source.run_id'].split('/')
    # Plot the map
    ax = global_plotter.plot(bg_variable, *plot_args, **plot_kwd_args)
    for data_id in data_ids:
        # Recover the coordinates of the rectangular subdomain
        run = mlflow.get_run(data_id)
        run_params = run.data.params
        lat_min, lat_max = run_params['lat_min'], run_params['lat_max']
        lon_min, lon_max = run_params['long_min'], run_params['long_max']
        lat_min, lat_max = float(lat_min), float(lat_max)
        lon_min, lon_max = float(lon_min), float(lon_max)
        x, y = lon_min, lat_min
        width, height = lon_max - lon_min, lat_max - lat_min
        ax.add_patch(Rectangle((x, y), width, height, facecolor=facecolor,
                               edgecolor=edgecolor))
    plt.show()
    return ax