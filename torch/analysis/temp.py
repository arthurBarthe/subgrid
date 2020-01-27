# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:08:24 2020

@author: Arthur
Analysis script.
"""

from full_cnn1 import RawData, MultipleTimeIndices, FullyCNN
from torch.utils.data import Subset, DataLoader
from loadmlflow import LoadMLFlow
import numpy as np
from analysis import TimeSeriesForPoint
import mlflow

import sys


if not hasattr(sys.modules['__main__'], 'mlflow_runs'):
    mlflow_runs = mlflow.search_runs()
#mlflow_runs.sort_values(mlflow_runs['metrics.test mse'])
print(mlflow_runs)

id_ = int(input('Run id?'))
run_id = mlflow_runs['run_id'][id_]
mlflow_loader = LoadMLFlow(run_id)
mlflow_loader.net_class = FullyCNN
mlflow_loader.net_filename = '???'
net = mlflow_loader.net

# Display some info about the train and validation sets for this run
train_split = mlflow_loader.train_split
test_split = mlflow_loader.test_split
print(f'Train split: {train_split}')
print(f'Test split: {test_split}')


batch_size = 8

pred = mlflow_loader.predictions
truth = mlflow_loader.true_targets

time_series0 = TimeSeriesForPoint(predictions=pred, truth=truth)
time_series0.point = (15, 75)
time_series0.plot_pred_vs_true()

