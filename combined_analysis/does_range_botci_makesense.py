#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does bootstrapping 'work' for range estimates?
==============================================
I've seen bootstrapping being used to get confidence intervals around the
mean or median - but is it also expected to work for a parameter like range (max-min)?

Here I'll try to see if it works with a simple example. 

Author: Thejasvi Beleyur
Original date of creation : 2020-12-13
"""
import numpy as np 
import scikits.bootstrap as boot

popA = np.arange(0,10000)
popB = np.arange(0,100000,100)

subA, subB = [np.random.choice(each, 1000, replace=False) for each in [popA, popB]]

def get_range_difference(popA, popB):
    range_A = np.nanmax(popA)-np.nanmin(popA)
    range_B = np.nanmax(popB)-np.nanmin(popB)
    
    return range_B-range_A

outs = boot.ci((subA, subB), get_range_difference)

print(f'The 95%ile CI is : {outs}, and the original population level difference\
      in range is: {get_range_difference(popA, popB)}')
      
# %% 
# Bootstrapping CI's do cover the range of the population well!
# -------------------------------------------------------------
