#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:50:18 2020

@author: arthur
"""


class TaskInfo:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(f'Starting task: {self.name}')

    def __exit__(self, *args):
        print(f'Task completed: {self.name}')
