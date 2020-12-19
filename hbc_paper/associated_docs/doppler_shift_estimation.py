#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimating the amount of Doppler shift as bats flew across the microphones
==========================================================================


@author: Thejasvi Beleyur

this code is realised with an MIT license
"""

import numpy as np 
import matplotlib.pyplot as plt


speeds = np.array([1,2,3]) # speed in m/s
theta = np.array([45,75,105,135]) # angle relative to microphone perpendicular

f = 100*10**3 # the emitted frequency

vsound = 338.0 

doppler_shift = {}

for v in speeds:
    vcostheta = v*np.cos(np.deg2rad(theta))
    f_obs = f*(vsound/(vsound-vcostheta))
    doppler_shift[v] =  np.column_stack((theta, f-f_obs))
    print(f_obs-f, 'Delta frequency')
    
# %% 
# The min-max range of Doppler shifts is +/-209-623 Hz for a range of speeds
# between 1-3 m/s. The bats are considered to not be flying directly towards
# or away from the microphones because that rarely happens. Assuming this also 
# overestimates the Doppler shift, and thus creates a false impression of 
# how much variation it actually causes. 

plt.figure(figsize=(10,8))
for speed, data in doppler_shift.items():
    plt.plot(data[:,0], data[:,1],'-*',label=f'{speed} m/s')
plt.legend()
plt.ylabel('Doppler shift $F-F_{observed}$, Hz, \n $F=100kHz$', fontsize=12)
plt.xlabel('Source-observer angle, degrees', fontsize=12)
plt.xticks(theta, theta)
plt.tight_layout()
plt.savefig('SI_dopplershift.png')
