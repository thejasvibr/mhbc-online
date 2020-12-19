#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Data formatting and cleaning convenience functions 
to handle the output from the itsfm package

Removing bad itsfm measurements: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bad measurements/segmenetations from 
in each measurements file are entered in a 'donot_analyse.txt' file. 
The poorly analysed audio file name is written in the txt file 
in a separate row.

@author: tbeleyur
License: MIT LICENSE
"""
import datetime as dt
import time
import glob
import numpy as np 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 

dB = lambda X: 20*np.log10(abs(X))

def num_parts_in_call(call_rows):
    return call_rows.shape[0]

def num_parts_in_calldataset(df):
    '''
    Parameters
    ----------
    audio_file : pd.DataFrame
    
    Returns 
    -------
    num_parts_per_call : pd.DataFrame
            Naudio_files x 1 with the number of call parts per 
            unique audio file analysed
    '''
    each_call = df.groupby('audio_file')
    
    parts_per_call = []
    call_id = []
    for audio_file, each in each_call:
        parts_per_call.append(num_parts_in_call(each))
        call_id.append(audio_file)
        
    
    return pd.DataFrame(data={'audio_file':call_id,
                              'num_parts': parts_per_call},
                            index=range(len(each_call)))

def remove_badquality_points(results_folder, raw_msmts):
    '''
    Removes the rows which match the annotation audio in the 
    donot_analyse file from the raw_msmts pd.DataFrame. 
    
    Parameters
    ----------
    results_folder : str/path 
        path to the  the results folder with a donot_analyse.txt file in it
    raw_msmts : pd.DataFrame
        The DataFrame must have at least one column named 'audio_file'
    
    Returns
    -------
    cleaned_msmts: pd.DataFrame
        Version of raw_msmts without the audio files mentioned in the donot_analyse.txt
        If there are no files to be excluded the `donot_analyse.txt` file must 
        just have `None` in it. 
        Otherwise, each file to be excluded must be entered in a separate row. 
        The `donot_analyse.txt` file should not have a header. 
    '''
   
    filepath = os.path.join(results_folder, 'donot_analyse.txt')
    try:
        to_remove = pd.read_csv(filepath, header=None)
    except:
        raise ValueError(f'tried to read {filepath} -- is it there or formatted properly?')

    if np.logical_and(to_remove.shape==(1,1), to_remove[0][0]=='None'):
        cleaned_msmts = raw_msmts.copy()
    else:
        all_audio_files = raw_msmts
        rows_to_remove = []
        for _, each in to_remove.iterrows():
            rows = np.argwhere(raw_msmts['audio_file'].isin([each[0]]).tolist()).flatten()
            rows_to_remove.append(rows)
        all_rows_to_remove = np.concatenate(rows_to_remove)
        cleaned_msmts = raw_msmts.drop(all_rows_to_remove).reset_index(drop=True)
    return cleaned_msmts

def remove_these_audiofiles(audiofile, df):
    '''
    '''
    rows_to_remove = []
    for each in audiofile:
        rows = np.argwhere(df['audio_file'].isin([each]).tolist()).flatten()
        rows_to_remove.append(rows)
    all_to_remove = np.concatenate(rows_to_remove)
    return df.drop(all_to_remove).reset_index(drop=True)

def get_numbats_from_annotation_id(audio_file_ids, video_annotation_folder):
    '''
    Searches for all matching rows with the audio file id across all video annotation files
    and gets the number of bats observed in that annotation. 
    
    Parameters
    ----------
    audio_file_id : str
    video_annotation_folder : str/path
    
    Returns
    -------
    num_bats : int>0
        The number of bats in video over the annotation.
    '''

    # find and load all annotations from files
    all_annotations = read_all_annotation_files(video_annotation_folder)
  
    # search for each audio_file_id in DF
    num_bats = []
    for each in audio_file_ids:
        row_num = search_for_matching_id(each, all_annotations)
        num_flying_bats = get_num_flying_bats(row_num, all_annotations)
        num_bats.append(num_flying_bats)
    return num_bats

def convert_annotation_time_to_posix(timestamp):
    '''
    Expects the time stamp to be in YYYY-MM-DD hh:mm:ss format. 
    '''
    datetime_obj = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    posix_time = time.mktime(datetime_obj.timetuple())
    return posix_time


def get_annotation_end_time(row_number, annotations):
    '''
    '''
    that_row = annotations.iloc[row_number,:]
    return str(that_row['end_timestamp'])


def get_annotation_start_time(row_number, annotations):
    '''
    '''
    that_row = annotations.iloc[row_number,:]
    return str(that_row['start_timestamp'])


def get_startstoptime_from_annotation_id(audio_file_ids, video_annotation_folder, endtime=True):
    '''
    Searches for all matching rows with the audio file id across all video annotation files
    and gets the number of bats observed in that annotation. 
    
    Parameters
    ----------
    audio_file_id : str
    video_annotation_folder : str/path
    
    Returns
    -------
    end_times : list
        The end time of all the annotations
    '''
    # find and load all annotations from files
    all_annotations = read_all_annotation_files(video_annotation_folder)
  
    if endtime:
        get_annot_time = get_annotation_end_time
    else:
        get_annot_time = get_annotation_start_time
    # search for each audio_file_id in DF
    times = []
    for each in audio_file_ids:
        row_num = search_for_matching_id(each, all_annotations)
        timepoint = get_annot_time(row_num, all_annotations)
        times.append(timepoint)
    return times
    


pandas_read_function = {'*.csv': pd.read_csv,
                       '*.xls': pd.read_excel}
        
def read_all_annotation_files(annotation_folder, file_format='*.csv'):
    '''
    Detects and loads all files of a matching format in the annotation folder into 
    one giant pd.DataFrame
    
    Parameters
    ----------
    annotation_folder : str/path
    file_format : list
        With the formats of the files. The format must be preceded with a *. 
        Defaults to '*.csv'
    
    Returns 
    -------
    all_annotations : pd.DataFrame
        All annotations loaded from detected annotation files in one dataframe.
    '''
    # load all matching format files
    all_matching_files = []
    candidate_path = os.path.join(annotation_folder, file_format)
    this_format_match =  glob.glob(candidate_path)
    all_matching_files += this_format_match
    
    # read all files
    all_annotation_container = []
    for each in all_matching_files:
        try:
            df = pandas_read_function[file_format](each)
        except UnicodeDecodeError:
            df = pandas_read_function[file_format](each, engine = 'python')
        
        all_annotation_container.append(df)
    
    all_annotations = pd.concat(all_annotation_container)
    return all_annotations
    
    
def search_for_matching_id(unique_id, annotations):
    '''
    Searches through a big dataframe with all video annotations. 
    Throws an error either if NO matches are found, or if more than one match is found. 

    Parameters
    ----------
    unique_id : str
    annotations = pd.DataFrame

    Returns 
    -------
    Int64Index 
        The row index where the matching ID is found. 

    '''
    match_found = annotations['annotation_id'].isin([unique_id])
    if sum(match_found) == 0:
        raise ValueError(f'{unique_id} not found among the given annotations')
    elif sum(match_found) >1:
        raise ValueError(f' Multiple matches found ({sum(match_found)}) for {unique_id} among the given annotations')
    return int(np.argwhere(match_found.tolist()))
    
def get_num_flying_bats(row_number, annotations):
    '''
    '''
    that_row = annotations.iloc[row_number,:]
    return int(that_row['no. of flying bats'])

remove_segment_matching_annotaudio_ = lambda X: X[28:]

manual_check_done = lambda X: 'manual_check_done.txt' in os.listdir(X)

def get_folders_with_manual_checks(basepath):
    '''
    Checks all subdirectories in the basepath and outputs only 
    those that have a 'manual_check_done.txt' file in them. 
    
    Parameters
    ----------
    basepath : str/path
    
    Returns
    -------
    manual_checked : list with subdirectories 
    '''
    subdirectories = [x[0] for x in os.walk(basepath)]    
    manual_checked = list(filter(manual_check_done, subdirectories))
    return manual_checked

class TooManyCallParts(ValueError):
    pass
class MultipleAudioFileIds(ValueError):
    pass
class OddCallParts(ValueError):
    pass

def verify_one_call_measurements(call_parts_df):
    '''
    Checks if 1) there are <= 3 call parts and 2) the presence of multiple 
    audio_file ids in the df and 3) that there's at most one CF region.

    The presence of >3 call parts indicates that the segmentation needs some 
    manual inspection. 
    
    The presence of multiple audio files could mean that there's been a mix up 
    of multiple audio files being attributed to the same video annotation. 
    
    Check that there's only one CF segment
    
    Parameters
    ----------
    call_parts_df : pd.DataFrame
        Nparts x Mmeasurements dataframe. There can be upto 3 Nparts, anymore 
        and an error will be raised. The only compulsory column is 'audio_file'
    
    Raises
    ------
    TooManyCallParts
    MultipleAudioFileIds
    '''
    annot_id = call_parts_df['video_annot_id'].unique()
    rows, cols = call_parts_df.shape
    if rows > 3:
        raise TooManyCallParts(f'{rows} call parts detected for annotation id {annot_id}')
    
    audio_files = call_parts_df['audio_file'].unique()
    num_audio_files = len(audio_files)
    if num_audio_files > 1:
        raise MultipleAudioFileIds(f"{num_audio_files} audio files ({audio_files}) attributed to the same annotation id {annot_id}")
    
    call_parts = call_parts_df['region_id']
    num_cf_parts = sum(list(map(lambda X: 'cf'in X, call_parts)))
    if num_cf_parts > 1:
        raise TooManyCallParts(f"{num_cf_parts} CF parts found for {annot_id}. There can only be at most 1.")

def fm_regions_make_sense(call_parts):
    '''
    Checks that the columns with 'fm' are 
    '''
    regions = call_parts['region_id']
    pass

def make_measurements_to_wide(time_sorted, common_columns):
    '''
    Gives each region's measurement a separate column. 
    Also removes any columns with 'Unnamed' in it.
    
    Parameters
    ----------
    time_sorted : pd.DataFrame
        with compulsory column 'TS_partname'
   
    Returns
    -------
    region_wide_formatted : pd.DataFrame
        1 x N columns dataframe
    '''
    unique_columns_raw = set(time_sorted.columns) - set(common_columns)
    unique_columns = list(filter(lambda X: 'Unnamed' not in X, unique_columns_raw))
    
    each_region_data = []
    for indx, region in time_sorted.iterrows():
        new_colnames = [ region['TS_partname'] + colname for colname in unique_columns]
        region_data = {}
        for old_colname, new_colname in zip(unique_columns, new_colnames):
            region_data[new_colname] = region[old_colname] 
        region_msmts = pd.DataFrame(data=region_data, index=[0])
        each_region_data.append(region_msmts)
    region_wide_raw = pd.concat(each_region_data, 1)

    # remove any columns with 'partname' in them 
    region_wide_formatted = region_wide_raw.loc[:,~region_wide_raw.columns.str.contains('partname', case=False)] 
    
    # add the data from the common columns
    for each in common_columns:
        region_wide_formatted[each] = time_sorted[each][0]
        
    return region_wide_formatted


def parse_regionids_into_tianschnitzler_format(call_parts_df):
    '''
    Converts the itsfm measurements of a call into a one-row-per call type df. 
    From a 2/3 row df it becomes a single row df. 
    
    Each column is assigned one of the following prefixes : ifm_, tfm_ or cf_
    
    An FM region before the CF is considered the ifm, while an FM region 
    after the CF is considered a tfm. When both ifm and tfm are before/after
    the CF - an error is raised. 

    In the simple case the mapping between itsfm region_id's and TS labels will be:
    itsfm region_id : ['fm1','cf1','fm2'] --> ['ifm','cf','tfm']

    Parameters
    ----------
    call_parts_df : pd.DataFrame
        N call_parts x M columns
        There can only be 3 call parts in a call and there must be one 'cf' region for a sensible output. 
        The call_parts_df must also have the columns 'start' and 'stop' which refer 
        to the start and stop times of the regions.

    Returns 
    -------
    one_row_for_call : pd.DataFrame
        1 x (NxM) columns
    '''
    time_sorted = call_parts_df.sort_values(by='start').reset_index(drop=True)
    # parse the itsfm region IDs and convert them into TS region labels
    TS_labels = assign_TS_labels(time_sorted)    
    time_sorted['TS_partname'] = TS_labels
    # check which columns have the same values across all rows 
    common_columns = identify_common_columns(time_sorted)
    row_with_common_values = time_sorted.loc[0,common_columns]
    # single row with all measurements done on each TS region.
    row_with_all_region_measurements = make_measurements_to_wide(time_sorted, 
                                                                 common_columns)
    one_row_for_call = row_with_all_region_measurements.join(row_with_common_values)   
    return one_row_for_call


def identify_common_columns(df):
    '''
    Common column names are identified by repetition in the same value across rows. 
    With any numerical value this is highly unlikely -- but may fail !! 
    '''
    repeated_values_in_columns = df.apply(lambda X: X.duplicated(), 0)
    repeats_in_cols = repeated_values_in_columns.apply(lambda X: sum(X)>=1, 0)
    column_names = list(repeats_in_cols[repeats_in_cols].index)
    return column_names

def assign_TS_labels(time_sorted):
    '''
    '''
    cf_row_index = time_sorted['region_id'].apply(lambda X: 'cf' in X, 1).idxmax()
    cf_row = time_sorted.loc[cf_row_index,:]
   
    fm_part_count = {'ifm_':0, 'tfm_':0}
    ts_parts = {}
    ts_region_label = []# assign the Tian-Schnitzler name to the region
    for index, call_part_row in time_sorted.iterrows():
        call_part_before_cf = np.logical_and(call_part_row['start'] < cf_row['start'],
                                            call_part_row['stop'] <= cf_row['stop'])
        call_part_after_cf = np.logical_and(call_part_row['start'] >= cf_row['stop'],
                                            call_part_row['stop'] > cf_row['stop'])
        # if it's the CF row
        if call_part_row.equals(cf_row):
            ts_parts['cf_'] = cf_row
            candidate = 'cf_'
        # if region is ebfore CF -> ifm
        elif call_part_before_cf:
            candidate = 'ifm_'
            ts_parts['ifm_'] = call_part_row
            fm_part_count[candidate] += 1
        # if region is after CF -> tfm
        elif call_part_after_cf:
            candidate = 'tfm_'
            ts_parts['tfm_'] = call_part_row
            fm_part_count[candidate] += 1
        else:
            raise OddCallParts(f'Unable to parse call part {call_part_row}')
        ts_region_label.append(candidate)
    # if any of the parts are present twice this means something's wrong
    call_has_multiple_ifm_and_tfm = list(map(lambda X: X>1, fm_part_count.values()))
    if sum(call_has_multiple_ifm_and_tfm)>2:
        raise OddCallParts(f'>1 FM regions before/after CF!: {time_sorted}')

    call_has_no_fm = list(map(lambda X: X==0, fm_part_count.values()))
    if sum(call_has_no_fm)==2:
        raise OddCallParts(f'No FM parts detected: {time_sorted}')
    return ts_region_label

def make_one_row_per_call(call_part_measurements):
    '''
    Converts a 'long' format dataframe with 
    each call part having its own row, and multiple columns for measurements 
    into a one-row-per-call dataframe. 
    
    All calls are considered to have 'iFM', 'CF' and 'tFM' parts (sensu [1]). 
    There may be some calls which don't have an iFM or tFM, and these
    are then filled with nan's. 
    
    This function parses the fm1,cf1,fm2 type call region numbering from the 
    itsfm package [2], and converts it into the iFM, CF and tFM naming. 

    
    Parameters
    ----------
    call_part_measurements : pd.DataFrame
        Dataframe with many maeasurements from the itsfm package.
        The dataframe has N rows, where N is the sum of all detected call parts 
        across the M calls in the dataset. The Dataframe can have multiple 
        columns, and only 'must' have columns called 'video_annot_id' and 'audio_file'.

    Returns
    -------
    one_row_per_call : pd.DataFrame
        The dataframe will have M rows (when all calls can be well parsed) x (Nregions x measurements per region)
        columns for each row. 

    References
    ----------
    [1] Tian & Schnitzler 1996, Echolocation signals of the Greater Horseshoe
        bat(`Rhinolophus ferrumequinum`)in transfer flight and during landing, J. Acoust. Soc. America

    [2] Beleyur, Thejasvi, 2020, itsfm : identify, track and segment by frequency and its modulation, 
        (itsfm.rtfd.io for documentation and https://github.com/thejasvibr/itsfm for source)
    '''
    one_call_per_rows = []
    for annot_id, measurements in call_part_measurements.groupby(['video_annot_id']):
        # check the shape and basic content of the call part measurements
        verify_one_call_measurements(measurements)
        one_row_msmts = parse_regionids_into_tianschnitzler_format(measurements)
        one_row_msmts['video_annot_id'] = annot_id
        one_call_per_rows.append(one_row_msmts)
    
    one_row_per_call = pd.concat(one_call_per_rows).reset_index(drop=True)
    return one_row_per_call




