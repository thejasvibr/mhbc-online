#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chooses annotations that are separated by a user-defined interval of time. 
This act of finding of temporally spaced annotations helps reduce the 
extent of pseudo-replication in the data. 
"""
import datetime as dt
import numpy as np
import random
import scipy.spatial as spatial 

def split_into_clustered_and_nonclustered(time_points, cluster_width):
    '''
    

    Parameters
    ----------
    time_points : np.array
        Ntimepoints x 2 np.array with POSIX timestamps.
        Column 1 is the start and column 2 is the stop timestamp. 

    cluster_width : float
        The 'window' within which calls are considered clustered. 
        eg. if the cluster_width is 1 minute, then all calls within 
        1 minute of each other are considered clustered. 
        

    Returns
    -------
    clustered, isolated : np.arrays
        The row indices of timestamps split into 
        clustered and non-clustered. 
    '''
    
   
    # calculate inter-event distances 
    gap_matrix = spatial.distance.cdist(time_points, time_points, calculate_timegap)
    
    # find events that are at least cluster_width distance from all other points
    isolated = np.apply_along_axis(all_values_above_Y,0,gap_matrix,cluster_width)
    isolated_events = np.argwhere(isolated).flatten()
    clustered_events = np.argwhere(~isolated).flatten()
    return clustered_events, isolated_events

def all_values_above_Y(X,Y):
    nonans = X[~np.isnan(X)]
    return np.all(nonans>Y)
    

def calculate_timegap(timestamp1, timestamp2):
    '''
    '''
    if np.array_equal(timestamp1,timestamp2):
        return np.nan
    both = np.row_stack((timestamp1, timestamp2))
    starts = both[:,0]
    # get the earlier one 
    earlier = np.argmin(starts)
    later = np.argmax(starts)
    
    # time-gap
    timegap = both[later,0]-both[earlier,1]
    return timegap
    
    




def generate_time_separated_folds(time_points, min_separation, num_runs=1000, **kwargs):
    '''
    

    Parameters
    ----------
    time_points : np.array
        Nx2 np.array with each row containing the start (col 0) and stop (col 1)
        of an annotation
    min_separation : float>0
        Minimum time gap betwen the end of one annotation and the start of another. 
    num_runs : int, optional 
        Defaults to 1000
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    all_subset_indices = []
    kwargs = {'previous_starts':[]}
    
    for i in range(num_runs):
        index_subset, this_start = find_interval_separated_annotations(time_points, min_separation, **kwargs)
        kwargs['previous_starts'].append(this_start)
        all_subset_indices.append(index_subset)
    # remove all the failed trials
    no_empty_subsets = list(filter(lambda X: len(X)>0,all_subset_indices))

    
    unique_index_subsets = [list(each)for each in list(frozenset(no_empty_subsets))]
    
    return unique_index_subsets

def find_interval_separated_annotations(time_points, min_separation, **kwargs):
    '''
    Parameters
    ----------
    time_points : list
        List with POSIX time stamps, or anything with second level resolution. 
        The time points must be sorted in ascending order.
    min_separation : float>0
        Absolute minimum separation that any two annotations must have in seconds. 
    previous_starts : list, optional 
        A list with the indices of previous start points. These indices are 
        avoided as start points. 
    max_tries : int, optional 
        Maximum trials to perform to get a proper start point. 
        Defaults to 2Xtime_points length.
    
    Returns
    -------
    separated_annotation_indices: set
        Set of integer indices of annotations that are separated by the 
        minimum duration given. 
    current_start
    '''
    previous_starts = kwargs.get('previous_starts', [])
    max_tries = kwargs.get('max_tries', len(time_points)*2)
    # Choose one start point and save the start point. If start point has 
    # been used before re-choose a start point
    
    unique_start_not_found = True
    num_trials = 0
    while np.logical_and(unique_start_not_found, num_trials<max_tries):
        current_start = random.sample(range(len(time_points)), 1)[0]
        #print(current_start)
        current_start_in_previous_starts = len(set([current_start]).intersection(set(previous_starts)))>0
        num_trials += 1
        if not current_start_in_previous_starts:
            unique_start_not_found = False

    if num_trials >= max_tries:
        return [], 0

    valid_time_indices = []    
    valid_time_indices.append(current_start)
    start_point_not_met_again = True
    
    candidate_index = int(current_start) 
    while start_point_not_met_again:
        # Proceed to the right, increase candidate index by one and calculate
        # if minimum distance is met. 
        previous_index = int(candidate_index)
        candidate_index = rollover_counter(candidate_index + 1, len(time_points)-1)
        
        if candidate_index == current_start:
            start_point_not_met_again = False 
        separation = abs(time_points[candidate_index,0] - time_points[previous_index,1])
        
        # If a suitable match is found before the end of the list, then this is the
        # next anchor point
        if separation>=min_separation:
            valid_time_indices.append(candidate_index)
            #print(separation, candidate_index, previous_index)
           # print('actual',time_points[candidate_index], time_points[previous_index])
    
    # check that the last entry and the first are also within min_separation 
    # otherwise remove the last entry
    last_entries_timesetp = abs(time_points[valid_time_indices[-1],1] - time_points[valid_time_indices[0],0])
    last_entry_with_minsep = last_entries_timesetp >= min_separation
    if not last_entry_with_minsep:
        valid_time_indices.pop(-1)
    
    return frozenset(valid_time_indices), current_start
    

def rollover_counter(i,max_i):
    '''
    Can only handle a roll over once. 
    ie. i < 2 X max_i

    Parameters
    ----------
    i : int
        current index
    max_i : int
        max allowable index

    Returns
    -------
    The rollover version of the index
    '''
    if i<=max_i:
        return i 
    else:
        return i-max_i-1