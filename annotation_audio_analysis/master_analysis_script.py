#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whole audio analysis control script
===================================
This notebook is an attempt at me simplyfing the number of clicks I need
to perform each time there is a small change in one of the component notebooks
in the whole-audio analysis workflow. This script will run all notebooks.


Original date of creation : 2020-11-29
Author : Thejasvi Beleyur
"""

import nbformat
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

# Verified annotations
# --------------------
# This notebook determines which audio files are correct in the sense
# of having only the target species, and expected calls (single bat annotations
# with no call overlaps, multi-bat annotations with multiple bat calls)


print('Valid annotation notebook')
verified_audio_nb = 'choosing valid annotation audio files.ipynb'
with open(verified_audio_nb) as f:
    verifnb = nbformat.read(f, as_version=4)

# run the verified annots notebook 
wholeaudio_ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
wholeaudio_ep.preprocess(verifnb, {'metadata':{'path':'./'}})
# save 
with open(verified_audio_nb, 'w', encoding='utf-8') as f:
    nbformat.write(verifnb, f)

print('Virtual audio notebook')
# Virtual audio creation 
# ---------------------- 
virtualaudio_path = 'making_virtual_multi_bat_audio.ipynb'
with open(virtualaudio_path,  encoding='utf-8') as vf:
    virtuaudio_nb = nbformat.read(vf, as_version=4)
#run virtual audio notebook
wholeaudio_ep.preprocess(virtuaudio_nb, {'metadata':{'path':'./'}})
# save 
with open(virtualaudio_path, 'w', encoding='utf-8') as f:
    nbformat.write(virtuaudio_nb, f)

print('Split measure notebook')

# Split-measure audio files 
# -------------------------
splitmeasure_path = 'analysis//split-measuring_observed_and_virtual_audio_files.ipynb'
with open(splitmeasure_path,  encoding='utf-8') as spf:
    splitmeasure_nb = nbformat.read(spf, as_version=4)
# run split-measure
wholeaudio_ep.preprocess(splitmeasure_nb, 
                         {'metadata':{'path':'./analysis/'}})
# save
with open(splitmeasure_path,'w',encoding='utf-8') as spf:
    nbformat.write(splitmeasure_nb, spf)

print('Group dominant freqs. notebook')

# Group the dominant frequencies and generate the final split-measure
# -------------------------------------------------------------------
groupdomfreq_path = 'analysis/grouping dominant frequencies.ipynb'
with open(groupdomfreq_path) as gpf:
    groupdomfreq_nb = nbformat.read(gpf, as_version=4)

# run group dominant freqs
wholeaudio_ep.preprocess(groupdomfreq_nb, 
                         {'metadata':{'path':'./analysis/'}})

# save as 
with open(groupdomfreq_path, 'w', encoding='utf-8') as gdf:
    nbformat.write(groupdomfreq_nb, gdf)
    
# Now - also need to manually export the notebooks as html files from
# the command line 