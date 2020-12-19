#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Detrending the LED light intensity data. 
Created on Mon Jan 20 13:04:16 2020

This is a very rough hack to make sure the cross-correlations are good. 
Even though there are missing frames and perhaps uneven sampling all throughout 
the video - the average level of the led signal increases only because of shifting LED position 
within the frame or the change in IR lamp intensity. 

The de-trending should remove the effect of these two hopefully. 

When the LED moves a lot in the screen the most effective way to de-trend is 
to fit a cubic spline and then subtract from the spine fit. 
This gives a nice stable signel with fixed and constant max and min values. 


@author: tbeleyur
"""

import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import pandas as pd
import numpy as np 
import scipy.signal as signal 

file_name = 'videosync_OrlovaChukaDome_01_20180819_03.00.00-04.00.00[R][@38d6][2].avi_.csv'

df = pd.read_csv(file_name)
break_points = [40000,60000]

# raw signal 
raw_signal = df['led_intensity']
# de-trend the data
de_trended = signal.detrend(raw_signal, bp=break_points)
spline_fit = signal.cspline1d(np.array(raw_signal).flatten(), 100000)

spline_detrended = raw_signal - spline_fit
plt.figure()
plt.plot(raw_signal)
plt.plot(spline_fit)

plt.figure()
plt.plot(de_trended)
plt.plot(spline_detrended)


# save the de-trended file with new led_intensity
df['led_intensity'] = spline_detrended
df.to_csv('videosync_[detrended]_OrlovaChukaDome_01_20180819_03.00.00-04.00.00[R][@38d6][2].avi_.csv')