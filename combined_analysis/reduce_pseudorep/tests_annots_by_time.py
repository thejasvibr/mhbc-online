#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the functions in annots_by_time 
Created on Sun Dec 13 15:30:29 2020

@author: tbeleyur
"""
import numpy as np
from annots_by_time import find_interval_separated_annotations as fisa
from annots_by_time import generate_time_separated_folds as gtsf

start_times = [1, 4, 10, 20, 40,   0]
end_times   = [2, 8, 12, 25, 45, 0.5]
timepoints = np.column_stack((start_times, end_times))

fisa(timepoints, 4)

folds = gtsf(timepoints, 4, num_runs=2500)
