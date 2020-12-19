#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Module which reads individual call annotations, saves audio 
with calls in them and performs corrections
Created on Wed Mar 18 18:27:21 2020

@author: tbeleyur
"""
import os 
import pandas as pd 
import scipy.signal as signal 
import soundfile as sf



def extract_single_calls_from_annotations(annotation_df, audio_source,
                                          **kwargs):
    '''Wrapper funciton to execute write_annotation_segment_to_file 
    on each row of a dataframe. 

    Parameters
    -----------
    annotation_df : pd.DataFrame
        Dataframe with fixed column names expected -  see write_annotation_segment_to_file
    audio_soure : str/path
        Location where whole audio files are. 

    Keyword Arguments
    -----------------
    For the other optional keyword arguments see write_annotation_segment_to_file
    '''
    for i, row in annotation_df.iterrows():
        try:
            write_annotation_segment_to_file(row, audio_source, **kwargs)
        except:
            print('Could not save call!')
            print(row['audio_annotation_file'])


def correct_annotation_entries(annotations, corrections):
    '''
    
    Parameters
    ----------
    annotations : pd.DataFrame
        Dataframe with all annotations including at least the following columns:
        audio_annotation_file
    corrections : dictionary.
        Keys refer to the annotation ids in the annotations.
        Each entry itself corresponds to another dictionary with
        keys corresponding to columns names, and entries corresponding to 
        the new values to be assigned. 
    Returns
    --------
    corrected_annotations : pd.DataFrame
        Copy of input annotations df, with corrected values
    
    
    Example
    -------
    df = pd.DataFrame(data={'audio_annotation_file':['file1', 'file2','file3'],
                            'call_start_time':[0,0.5,1.5], 
                            'call_end_time': [0.2, 0.9, 2.5]})
    
    corrections = {
                   'file1' : {'call_start_time':0.1, 'call_end_time':0.3},
                   'file2' : {'call_end_time':1.0}
                   }
    
    annotations_corrected = correct_annotation_entries(df, corrections)
    '''
    corrected_annotations = annotations.copy()

    annotation_ids_to_be_corrected = corrections.keys()
    # index of all rows to be corrected
    rows_to_be_corrected = get_row_indices(annotations, 'audio_annotation_file',
                                           annotation_ids_to_be_corrected)
    
    for row, annotation_id in zip(rows_to_be_corrected,
                                  annotation_ids_to_be_corrected):
        corrections_to_be_made = corrections[annotation_id]
        for column, new_value in corrections_to_be_made.iteritems():
            corrected_annotations.loc[row, column] = new_value

    return corrected_annotations
    

def get_row_indices(df, column, targets):
    '''
    Parameters
    ---------
    df : pd.DataFrame
    column : str
    targets : list
        Values to be searched in the column
    Returns
    -------
    row_indices : list
    '''
    row_indices = []
    for target in targets:
        row = int(np.argwhere(df['audio_annotation_file']== target))
        row_indices.append(row)
    return row_indices

def correct_call_annotation_values(row,columns,new_values):
    '''
    '''
    if len(columns)!=len(new_values):
        raise IndexError('Length of columns and new values not the same!')

    for column, new_value in zip(columns, new_values):
        row[column] = new_value
    return row
        

def write_annotation_segment_to_file(annotation, audio_source, **kwargs):
    '''
    Parameters
    ----------
    annotation_row : pd.DataFrame
        The annotation row is expected to have at least the following 
        column names if single calls are to be extracted:
            audio_annotation_file, call_start_time, call_end_time
        If silent regions are to be extracted, then at least the following columns 
        must be there:
            audio_annotation_file, Silent_period_Start, Silent_period_end
    audio_source : str/path
    file_format : str, optional
        Defaults to '.WAV'
    file_prefix : str, optional
        Defaults to 'segment_'
    save_to : str/path, optional
        Defaults to current working directory
    save_channels : list, optional
        The channels to be saved. Index starting from 0. 
        Defaults to all channels
    get_calls : bool, optional 
        Whether to return audio segments corresponding to the calls or silences. 
        Defaults to True, which returns the calls. 

    Returns
    -------
    audio : np.array
    '''
    save_to = kwargs.get('save_to', '.')
    file_prefix = kwargs.get('file_prefix', 'segment_')
    file_format = kwargs.get('file_format', '.WAV')

    audio, fs = load_audio_from_annotation(annotation, audio_source, **kwargs)
    destination = os.path.join(save_to, file_prefix+annotation['audio_annotation_file']+file_format)
    
    if kwargs.get('save_channels') is None:
        sf.write(destination, audio, fs)
    else:
        sf.write(destination, audio[:,kwargs.get('save_channels')], fs)

def load_audio_from_annotation(annotation, audio_source, **kwargs):
    '''
    Parameters
    ----------
    annotation_row : pd.DataFrame
        The annotation row is expected to have at elast the following 
        column names:
        audio_annotation_file, call_start_time, call_end_time
    audio_source : str/path
        A folder with audio files in it or even subfolders
    file_format : str, optional
        Defaults to '.WAV'
    get_calls : bool, optional
        Whether to get the single call start and stop or the silence. 
        If True, then returns audio corresponding to the call. Otherwise, 
        returns the annotated silent regions. Defaults to True.
        
        

    Returns
    -------
    audio : np.array
    '''
    file_format = kwargs.get('file_format', '.WAV')
    file_name = annotation['audio_annotation_file']
    
    full_path = find_file_in_folder(file_name+file_format, audio_source)
    if len(full_path)>1:
        raise ValueError('Multiple matches found - please give a more specific file names')
    else:
        full_path = full_path[0]
    fs = sf.info(full_path).samplerate
    if kwargs.get('get_calls',True):
        start_sample  = int(float(fs)*annotation['call_start_time'])
        stop_sample  = int(float(fs)*annotation['call_end_time'])
    else:
        start_sample  = int(float(fs)*annotation['Silent_period_Start'])
        stop_sample  = int(float(fs)*annotation['Silent_period_end'])
        

    annotation_audio, _ = get_audio_snippet(full_path, start_sample,
                                                                stop_sample)
    return annotation_audio, fs

def get_audio_snippet(filepath, start_sample, stop_sample):
    '''
    '''
    audio, fs = sf.read(filepath, start=start_sample, 
                                    stop=stop_sample)
    return audio, fs 

def find_file_in_folder(filename, source_folder):
    '''find a file in a folder with many potentially many subfolders
    
    Parameters
    ----------
    filename : str
    source_folder : str/path

    Returns
    --------
    final_file_path : path
        The target file with its full path. 
        If the target file is not found, function returns with None
    '''
    candidate_files = []
    for root, dirs, all_files in os.walk(source_folder):
        for each_file in all_files:
            only_filename = os.path.split(each_file)[-1]

            if filename == only_filename:
                final_file_path = os.path.join(root, each_file)
                candidate_files.append(final_file_path)
                
    num_matches = len(candidate_files)
    print(warn_user.get(num_matches, 'Multiple matches found!!'))
       
    return candidate_files
    
warn_user={0:'No matches found!', 1:'Match found!'}    

to_string = lambda X: str(X)

def convert_entries_to_string(numeric_list):
    '''
    '''
    return map(to_string, numeric_list)




if __name__ == '__main__':
# filename = 'matching_annotaudio_Aditya_2018-08-17_34_142'
# onerow = pd.DataFrame(data={'audio_annotation_file':[filename], 
#                         'Silent_period_Start':[0.152],
#                         'Silent_period_end':[0.163]})
    folder = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/individual_call_analysis/annotation_audio/'
# write_annotation_segment_to_file(onerow, folder, get_calls=False,
#                                 save_channels=0,
#                                 file_prefix='silence_',)
    
    
    csv_folder = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/annotation_audio_analysis/'
    csv_file = 'silent_startstop.csv'
    df = pd.read_csv(csv_folder+ csv_file)
    for i in range(df.shape[0]):
        write_annotation_segment_to_file(df.iloc[i,:], folder, get_calls=False,
                                    save_channels=0,
                                    file_prefix='silence_',
                                    destination=csv_folder)
    
#     folder ='/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/individual_call_analysis/raw_individual_call_annotations/'
#     file_path = os.path.join(folder,'Audio Annotation Datasheet_2018-08-19_23_am.csv')
#     annots = pd.read_csv(file_path)
#     audio_folder = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/individual_call_analysis/annotation_audio/'
    
#     destination = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/individual_call_analysis/raw_individual_call_snippets/2018-08-19_0200-0300/'
    
#     extract_single_calls_from_annotations(annots, audio_folder,
#                                   save_to=destination,
#                                   save_channels=[0])
