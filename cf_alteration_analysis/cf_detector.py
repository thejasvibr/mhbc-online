"""
CF detector 
===========
Detects CF components in CF-FM calls. 


Author: Thejasvi Beleyur 
License: MIT License
"""
import matplotlib.pyplot as plt 
import scipy.signal as signal 
import numpy as np 
import soundfile as sf
import scipy.ndimage as ndimage 


def get_sensible_subportion(X,min_sample, max_sample):
    '''
    '''
    start_sensible, end_sensible = min_sample>=0, max_sample<X.size
    both_sensible = np.logical_and(start_sensible, end_sensible)
    start_or_end_sensible = np.logical_or(start_sensible, end_sensible)
    
    window_size=  max_sample-min_sample
    
    if both_sensible:
        return X[min_sample:max_sample]
    if not start_or_end_sensible:
        raise IndexError(f'Both min and max samples do not make sense: {min_sample, max_sample}')
    if not start_sensible:
        return X[:window_size]
    if not end_sensible:
        
        return X[X.size-window_size:]

def choose_loudest_window(X,fs,win_durn=0.5):
    '''
    '''
    if X.size/fs >= win_durn:
        # generate hilbert envelope
        envelope = np.abs(signal.hilbert(X))
        # lowpass the envelope at 0.25xwin_durn
        max_fluctuation = 0.25*win_durn
        max_freq = 1/max_fluctuation
        b,a = signal.butter(1,max_freq/(fs*0.5),'lowpass')
        envelope_lp = signal.lfilter(b,a,envelope)

        # get loudest part and +/- 1/2 win_durn
        peak = np.argmax(envelope_lp)
        half_winsamples = int(fs*win_durn*0.5)
        return get_sensible_subportion(X, peak-half_winsamples, peak+half_winsamples)
        
    else: 
        return X   

def get_band_of_interest(X,fs,**kwargs):
    '''
    Gets main band of interest by taking the peak frequency and +/- X Hz of it
    
    Parameters
    ----------
    X : np.array
    fs : float>0
    band_halfwidth: float, optional 
        Defaults to 1,500 Hz. 

    
    Returns 
    -------
    band_of_interest: tuple
        With (min_freq, max_freq) 
    '''
    
    db_spectrum = dB(np.fft.rfft(X))
    freqs = np.fft.rfftfreq(X.size, 1/fs)
    peak_freq = freqs[np.argmax(db_spectrum)]
    half_width = kwargs.get('band_halfwidth', 1500)
    band_of_interest = (np.max([0,peak_freq-half_width]),
                        np.min([peak_freq+half_width, fs*0.5])
                        )
    return band_of_interest


def cf_detector(X,fs, band_of_interest,**kwargs):
    '''
    Parameters
    ----------
    X : np.array
    fs : float>0
    band_of_interest: array-like
        With [min_freq, max_freq] in Hz. 
    threshold: float, optional
        The threshold in dB 'pixel' value, beyond which a CF region is considered to start.
        Defaults to to -140 dB, which is 110 dB above a 'silent' pixel's value. 
        A 'silent'/non-band pixel with no signal has a typical value of -250 dB. 
    NFFT : int, optional 
        Defaults to 128 samples for FFT size
    noverlap: int, optional
        Defaults to 127 samples - FFT parameter.
    min_cf_duration : float, optional 
        Minimum duration of each CF detection. Defaults to 10ms. 

    Returns 
    -------
    filtered_cf_regions : list with slices
        Where each slice represents one CF detection of at least min_cf_duration 
        length. 
    im : np.array
        The spectrogram matrix
    '''
    # generate spectrogram
    spec,freqs,t,im = plt.specgram(X, Fs=fs, NFFT=kwargs.get('NFFT', 128),
                           noverlap=kwargs.get('noverlap',127));
    min_freq, max_freq = band_of_interest
    row_indices = np.argwhere(np.logical_and(freqs>=min_freq, freqs<=max_freq))
    # take row average of all in band frequencies
    mean_power = np.mean(spec[row_indices,:],0).flatten()
    db_meanpower = dB(mean_power)
    above_threshold = db_meanpower >= kwargs.get('threshold', -140)

    # get regions continuously above threshold
    regions_above, num_regions = ndimage.label(above_threshold)
    regions = ndimage.find_objects(regions_above)
    
    # filter detections by duration. 
    filtered_cf_regions = []
    for detection in regions:
        each = detection[0]
        duration = (each.stop - each.start)/fs
        if duration >= kwargs.get('min_cf_duration', 0.010):
            filtered_cf_regions.append(detection)
    
    return filtered_cf_regions, im

dB = lambda X: 20*np.log10(np.abs(X))