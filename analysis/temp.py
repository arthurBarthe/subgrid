# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:08:24 2020

@author: Arthur
Analysis script.

TODO
-Allow to load a trained model to do some tests on it. For instance I'd like
to check what a zero input gives, see if it can explain the behaviour on 
the east border, and see if we can correct that by enforcing zero bias on all
layers.
-also show the input field for analysis
-might want to study the correlation between the input field and the error,
see if there is anything remaining.
- do something similar for the multiscale
"""
import numpy as np

from analysis.loadmlflow import LoadMLFlow
from analysis.utils import select_run, view_predictions, DisplayMode
from analysis.utils import play_movie
import mlflow

# We'll run this locally
mlflow.set_tracking_uri('file:///d:\\Data sets\\NYU\\mlruns')
mlflow.set_experiment('Default')

# If the runs dataframe already exists we use it. Note: this means you must
# restart the interpreter if the list of runs has changed.
run_id, experiment_id = select_run(sort_by='metrics.test mse')
mlflow_loader = LoadMLFlow(run_id, mlruns_path='d:\\Data sets\\NYU\\mlruns')

# Display some info about the train and validation sets for this run
train_split = mlflow_loader.train_split
test_split = mlflow_loader.test_split
print(f'Train split: {train_split}')
print(f'Test split: {test_split}')

pred = mlflow_loader.predictions[:, 0, ...]
truth = mlflow_loader.true_targets[:, 0, ...]
psi_field = mlflow_loader.psi
psi_field = psi_field / np.std(psi_field)

view_predictions(pred, truth, display_mode=DisplayMode.rmse)
view_predictions(np.zeros_like(psi_field), psi_field,  DisplayMode.difference)

#play_movie(truth, title='target')
animation1 = play_movie(pred, title='prediction')
animation2 = play_movie(psi_field, title='psi field')
