# -*- coding: utf-8 -*-
"""
How to analyse the CF durations in audio with single bat
and multi-bat echolocation calls? 
"""
import glob
import os
import numpy as np 
import soundfile as sf
import scipy.signal as signal 
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

audio_files = glob.glob('test_audio/match*.wav')
for each in audio_files:
    audio, fs = sf.read(each)
    name = os.path.split(each)[-1].split('.')[0]
    if len(audio.shape)>1:
        audio = audio[:,0]
    
    dB = lambda X: 20*np.log10(np.abs(X))
    spec, freqs, t, image = plt.specgram(audio, Fs=fs, NFFT=256);
    plt.close()
    
    above_70khz = np.logical_and(freqs>90000, freqs<120000)
    spec_ultrasonic = spec[above_70khz,:]
    db_specultrasonic = dB(spec_ultrasonic)
    
    #%% 
    # Check to see how many 'peaks' there are over frequency bands
    sum_spec= np.sum(spec_ultrasonic**2.0, 1)
    
    plt.figure()
    plt.plot(sum_spec)
    db_sumspec = dB(sum_spec)
    
    #%% Choose the main frequency band to focus on
    threshold = np.max(db_sumspec) - 3
    main_band = np.argwhere(db_sumspec>=threshold).flatten()
    
    plt.figure()
    plt.imshow(db_specultrasonic, aspect='auto', origin='lower')
    for each in main_band:
        plt.hlines(each, 0, db_specultrasonic.shape[1])
    
    #%% Detect CF portions in the audio
        
    silence_values = [-189, -188, -200,-207] # dB values
    silence_threshold = np.median(silence_values)
    cf_threshold = silence_threshold + 30
    
    db_spec_mainband = db_specultrasonic[main_band,:]
    
    # where are the 'CF bands'
    mask = db_spec_mainband >= cf_threshold
    mask_t = np.sum(mask, 0)>0
    cf_labels, num_cfs = ndimage.label(mask_t)
    objects = ndimage.find_objects(cf_labels)
    
    # remove all detections that are < 5ms
    
    # rought time step calculations -- NEEDS more WORK!!
    delta_t = (audio.shape[0]/fs)/db_specultrasonic.shape[1]
    
    final_detections = []
    for each in objects:
        if (each[0].stop-each[0].start)*delta_t >= 0.005:
            final_detections.append(each)
    
    plt.figure()
    plt.imshow(db_specultrasonic, aspect='auto', origin='lower')
    for each in final_detections:
        plt.hlines(main_band[0], each[0].start, each[0].stop,'k', linewidth=2)
    plt.savefig(name+'.png')







