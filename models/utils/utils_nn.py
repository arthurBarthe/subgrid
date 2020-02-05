# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:40:09 2020

@author: Arthur
"""
from enum import Enum

class DEVICE_TYPE(Enum):
    GPU = 'GPU'
    CPU = 'CPU'

def print_every(to_print: str, every: int, n_iter: int):
    """prints every given number of iterations.
    Parameters
    ----------
    to_print: str
    The string to print
    
    every: int
    The string passed to the function is only printed every 'every' call.
    
    n_iter:int
    The index of the calls, which is to be handled by the user.
    """
    if n_iter % every == every - 1:
        print(to_print)


class RunningAverage:
    """Class for online computing of a running average"""
    def __init__(self):
        self.n_items = 0
        self.average = 0

    @property
    def value(self):
        return self.average

    def update(self, value: float, weight: float = 1):
        """Adds some value to be used in the running average.
        Parameters
        ----------
        value: float
        Value to be added in the computation of the running average.

        weight: int
        Weight to be given to the passed value. Can be useful if the function
        update is called with values that already are averages over some
        given number of elements.
        """
        temp = self.average * self.n_items + value * weight
        self.n_items = self.n_items + weight
        self.average = temp / self.n_items
        return self.average

    def __str__(self):
        return str(self.average)