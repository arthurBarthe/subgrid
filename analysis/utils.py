# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:00:45 2020

@author: Arthur
"""
import numpy as np
import mlflow
import sys


def correlation_map(truth, pred):
    """Computes the correlation at each point of the domain between the
    truth and the prediction."""
    correlation_map = np.mean(truth * pred, axis=0)
    correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
    correlation_map /= np.std(truth, axis=0) * np.std(pred, axis=0)
    return correlation_map


def select_run(sort_by=None):
#    if not hasattr(sys.modules['__main__'], 'mlflow_runs'):
    mlflow_runs = mlflow.search_runs()
    cols = ['run_id']
    if sort_by is not None:
        mlflow_runs.sort_values(by=sort_by)
        cols.append(sort_by)
    print(mlflow_runs[cols])
    id_ = int(input('Run id?'))
    return mlflow_runs.loc[id_, ['run_id', 'experiment_id']]
