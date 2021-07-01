# -*- coding: utf-8 -*-
"""
How to analyse the # of calls and their duration over a period?
"""
import numpy as np 
import soundfile as sf
import scipy.signal as signal 
import matplotlib.pyplot as plt


audio, fs = sf.read('test_audio/matching_annotaudio_Aditya_2018-08-16_2324_18_hp_singlebatmixed.WAV')

dB = lambda X: 20*np.log10(np.abs(X))
spec, freqs, t, image = plt.specgram(audio, Fs=fs, NFFT=256);

above_70khz = np.logical_and(freqs>70000, freqs<120000)
spec_ultrasonic = spec[above_70khz,:]
#%%
plt.figure()
plt.imshow(dB(spec_ultrasonic), aspect='auto', origin='lower')

#%% 
# Check to see how many 'peaks' there are over frequency bands
sum_spec= np.sum(spec_ultrasonic**2.0, 1)
plt.figure()
plt.plot(sum_spec)