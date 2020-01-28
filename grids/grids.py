# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:52:24 2019

@author: Arthur
In this file we provide a few classes to handle gridded data, and methods
to implement coarse-graining etc. Additionally we store some information
about the scale, in comparison to a simple ndarray.
"""
import numpy as np
import matplotlib.pyplot as plt


class RectangularGrid:
    """Defines a rectangular grid."""
    def __init__(self, dims: tuple, step_sizes: tuple = None):
        self._dims = dims
        if step_sizes is None:
            self.step_sizes = (1,) * len(dims)
        else:
            self.step_sizes = step_sizes
    
    @property
    def dims(self):
        return self._dims
    
    @dims.setter
    def dims(self, value):
        self._dims = value
    
    @property
    def step_sizes(self):
        return self._step_sizes
    
    @step_sizes.setter
    def step_sizes(self, value: tuple):
        assert len(value) == len(self.dims), 'The passed step sizes do not\
        have the expected dimension.'
        self._step_sizes = tuple(value)

    @property
    def lengths(self):
        result = []
        for i in range(len(self.dims)):
            result.append(self.dims[i] * self.step_sizes[i])
        return tuple(result)

    def __repr__(self):
        rep = 'Rectangular Grid\n'
        rep += f'    -dims: {self.dims}\n'
        rep += f'    -step sizes: {self.step_sizes}'
        return rep

    def check_data(self, data: np.ndarray):
        """Checks if the passed array corresponds in terms of its dimensions
        to the definition of the grid."""
        assert data.shape == self.dims, 'The passed data does not fit \
        the specifications of the grid'


class RectangularData:
    """Handles data on a rectangular grid and provides methods for
    coarse-graining"""
    def __init__(self, grid: RectangularGrid = None, data: np.ndarray = None):
        if grid is not None:
            assert isinstance(grid, RectangularGrid), 'The passed grid is not \
            valid.'
            self.grid = grid
            self.data = data
        elif data is not None:
            self.grid = RectangularGrid(data.shape)
            self.data = data
        else:
            raise ValueError
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        self.grid.check_data(value)
        self._data = value

    def _coarse_grain1d(self, data, factor: int, axis: int):
        """Returns the coarse-graining operation along the specified axis.
        Not public, this is used by the class method coarse_grain."""
        new_shape = list(data.shape)
        new_shape[axis] //= factor
        new_data = np.zeros(new_shape)
        for i in range(factor):
            cut_index = data.shape[axis] // factor * factor
            indices = np.arange(i, cut_index, factor)
            new_data += np.take(data, indices, axis)
        new_data /= factor
        return new_data

    def coarse_grain(self, factor: int, dims=None):
        """Returns a coarse-grain version of the initial data. For now we 
        require factor to be an integer. Note that for now all we do is
        subsample, we should average somehow in the future."""
        # TODO implement non integer case
        # TODO check if this can be made faster.
        result_data = self.data
        if dims is None:
            dims = range(self.data.ndim)
        new_step_sizes = list(self.grid.step_sizes)
        for i in dims:
            result_data = self._coarse_grain1d(result_data, factor, i)
            new_step_sizes[i] *= factor
        result_grid = RectangularGrid(result_data.shape, new_step_sizes)
        return RectangularData(result_grid, result_data)

    def plot(self):
        """Plots the data using the imshow function, and using the grid
        info for the axes limits"""
        plt.figure()
        lengths = self.grid.lengths
        plt.imshow(self.data, extent=[0, lengths[1], 0, lengths[0]])

if __name__ == '__main__':
    simple_grid = RectangularGrid((10, 10), (0.1, 0.2))
    data = RectangularData(simple_grid, np.random.randint(0, 10, (10, 10)))
