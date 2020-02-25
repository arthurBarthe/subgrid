#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:39:23 2020

@author: arthur
Script that lists the data runs with the relevant information
"""

import mlflow

runs = mlflow.search_runs(experiment_ids=['1',])
runs_short = runs[['run_id', 'params.scale', 'params.ntimes', 
            'params.lat_min', 'params.lat_max', 
            'params.long_min', 'params.long_max']]

print('Select a run by entering its integer id (0, 1, ...) as listed\
      from below, in order to obtain more details.')
print(runs_short)
while True:
    selection = input('Select run: ')
    try:
        selection = int(selection)
    except ValueError as e:
        break
    print(runs.iloc[selection, :])

