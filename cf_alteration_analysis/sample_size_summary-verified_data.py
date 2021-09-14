# -*- coding: utf-8 -*-
"""
Summarising sample sizes for the multi-call extension of 
the CF duration analysis 

Author: Thejasvi Beleyur 
2021 September 

code released under MIT License 
"""

import pandas as pd

#%% Load the data and check how many multi and single bat CF detections have occured

df['multi_bat'] = df['num_bats'].apply(lambda X:  True if X>1 else False)
multi_detections = df[df['multi_bat']==True]
single_detections = df[df['multi_bat']!=True]

num_multiCF_detections = multi_detections.shape[0]
num_singleCF_detections = single_detections.shape[0]

#%% How many bat activity audio fiels are the detections spread over? 

n_multi_files = len(multi_detections['audio_file'].unique())
n_single_files = len(single_detections['audio_file'].unique())
