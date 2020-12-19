#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inbuilt functions that perform various measurements on a given audio segment.

All measurement functions need to have one compulsory input, and accept keyword
arguments (they can also be unused).

The output of a measurement function is in the form of a dictionary with 
the key indicating the measurement output and the entry being in the form of 
a list with one or more values. 


@author: tbeleyur
"""
import numpy as np 
import scipy.signal as signal 
import scipy.ndimage as ndimage

def dB(X):
    return 20*np.log10(abs(X))

def rms(audio, **kwargs):
    return {'rms': np.sqrt(np.mean(audio**2.0))}

def peak_amplitude(audio, **kwargs):
    return {'peak_amplitude':  np.max(np.abs(audio))}

def make_smoothened_spectrum(audio, **kwargs):
    '''
    Makes a smoothed power spectrum with power in dB

    Parameters
    ----------
    audio 
    fs 
    spectrum_smoothing_width


    Returns
    -------
    None.

    '''
    fs = kwargs['fs']
    spectrum_smoothing_width = kwargs['spectrum_smoothing_width']  
    
    power_spectrum_audio = 20*np.log10(np.abs(np.fft.rfft(audio)))
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = np.max(np.diff(freqs_audio))
    
    # make the smoothed spectrum
    smoothing_window = int(spectrum_smoothing_width/freq_resolution)
    smoothed_spectrum = np.convolve(power_spectrum_audio,
                                    np.ones(smoothing_window)/smoothing_window, 
                                    'same')
    return smoothed_spectrum
    

def lower_minusXdB_peakfrequency(audio, **kwargs):
    '''
    Returns the lowest frequency that is -X dB of the peak frequency.
    First the smoothened spectrum is generatd, and then the -X dB
    
    Parameters
    ----------
    audio : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    db_range : float, optional
        The X dB range. Defaults to 20 dB
    spectrum_smoothing_width : 

    Returns
    -------
    dictionary with key "minus_XdB_frequency" and entry with a list with one float
    '''
    fs = kwargs['fs']
    db_range = kwargs.get('db_range',20)
    #identify the peak frequency of the audio clip
    smooth_spectrum = make_smoothened_spectrum(audio, **kwargs)
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = np.max(np.diff(freqs_audio))
    
    peak_f = peak_f = freqs_audio[np.argmax(smooth_spectrum)]# peak_frequency(audio, fs=fs)

    below_threshold = smooth_spectrum <= np.max(smooth_spectrum)-db_range
    freqs_below_threshold = freqs_audio[below_threshold]
    # choose only those frequencies that are below the peak frequencies.
    freqs_below_peak = freqs_below_threshold[freqs_below_threshold<peak_f]
    if len(freqs_below_peak) <1:
        return {"minus_XdB_frequency": np.nan}
    else:
        minus_XdB_frequency = np.max(freqs_below_peak)
        return {"minus_XdB_frequency": minus_XdB_frequency}

def dominant_frequencies(audio, **kwargs):
    '''
    Identify multiple dominant frequencies in the audio. This works by considering
    all frequencies within -X dB of the peak frequency. 
    
    The 'dominant frequency' is identified as the central point of a region which 
    is continuously above the threshold. 
    
    The power spectrum is normally quite spikey, making it hard to discern individual 
    peaks with a fixed threshold. To overcome this, the entire spectrum is 
    mean averaged using a running mean filter that is 3 frequency bands long.

    
    Parameters
    ----------
    audio : np.array
    fs : float 
        sampling rate
    spectrum_smoothing_width : float
        The extent of spectral smoothing to perform in Hz. 
        This corresponds to an equivalent number of centre frequencies that will 
        be used to run a smoothing filter over the raw power spectrum. The
        number of center frequencies used to smooth in turn depends on the frequency resolution 
        of the power spectrum itself.         
        See Notes for more.
    inter_peak_difference : float 
        The minimum distance between one peak and the other in the smoothed power spectrum in
        Hz.
    peak_range : float
        The range in which the peaks lie in dB with reference to the maximum spectrum value. 

    Returns
    -------
    dictionary with key "dominant_frequencies" and a  List with dominant frequencies in Hz. 

    Notes
    -----
    The spectrum_smoothing_width is calculated so. If the spectrum frequency resolution is
     3 Hz, and the given spectrum_smoothing_width is set to 300 Hz, then the number
     of center frequencies used for the smoothing is 100. 
     
     The peak_range parameter is important in determining how wide the peak detection
     range is. If the peak_range parameter is very large, then there is a greater chance
     of picking up irrelevant peaks. 
     
    '''
    fs = kwargs['fs']
    inter_peak_difference = kwargs['inter_peak_difference']
    peak_range = kwargs['peak_range']

    
    smoothed_spectrum = make_smoothened_spectrum(audio, **kwargs)
    # get the dominant peaks 
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = np.max(np.diff(freqs_audio))
    inter_peak_distance = int(inter_peak_difference/freq_resolution)
    peak_heights = (np.max(smoothed_spectrum)-peak_range, np.max(smoothed_spectrum))
    peaks, _ = signal.find_peaks(smoothed_spectrum,
                                 distance=inter_peak_distance, 
                                 height=peak_heights)
    dominant_frequencies = freqs_audio[peaks].tolist()
    return {"dominant_frequencies": dominant_frequencies}

def peak_frequency(audio, **kwargs):
    fs = kwargs['fs']
    spectrum_audio = np.abs(np.fft.rfft(audio))
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    peak_ind = np.argmax(spectrum_audio)
    peak_freq = freqs_audio[peak_ind]
    return peak_freq

def get_FM_terminal_frequencies(audio, **kwargs):
    '''
    '''
    
    above, (fm_regions, conserve_fmregions), (freqs, t, spec) = identify_fm_pixel_regions(audio, **kwargs)
    continuous_fm = identify_continuous_fm(conserve_fmregions)
    
    fm_terminal_frequencies = find_terminal_frequency(continuous_fm, above, freqs)
    return {'fm_terminal_freqs':fm_terminal_frequencies}


def identify_fm_pixel_regions(audio, **kwargs):
    '''
    Parameters
    ----------
    audio: np.array
    fm_freqs_condition : function
        Function which accepts an array with the centre frequencies of a 
        spectrogram slice, and outputs a boolean array with True indicating
        the rows in which the FM portions are to be found. 
    overlap
    nperseg
    fs
    
    Returns
    -------
    above
   (fm_regions, conserv_fmregions)
   (freqs, t, spec) 
    '''
    # make spectrogram of audio clip
    overlap_samples = int(kwargs['overlap']*kwargs['nperseg'])
    freqs,t,spec = signal.spectrogram(audio, fs=kwargs['fs'], nperseg=kwargs['nperseg'],
                                 noverlap=overlap_samples)
    
    kwargs['frequencies'] = freqs
    # calculate baseline level in non-signal regions across spectrogram
    baseline_levels = np.apply_along_axis(get_baseline_level, 0, spec, **kwargs)
    
    # Find all pixels that are above db Range of baseline level
    above = identify_pixels_above_baseline(spec, baseline_levels, **kwargs)
    
    # Get regions in time which are above baseline level and below the 
    fm_freqs = kwargs['fm_freqs_condition'](freqs)
    fm_regions = np.sum(above[fm_freqs,:],0)
    
    # output only those regions where there're >2 specgram pixels in a slice above threshold
    conserv_fmregions = fm_regions.copy()
    conserv_fmregions[conserv_fmregions<2] = 0 
    
    return above, (fm_regions, conserv_fmregions), (freqs,t,spec)


def identify_pixels_above_baseline(specgram, baseline_levels,**kwargs):
    '''
    Parameters
    ----------
    specgram : np.array
        Mfreqs x Nslices spectrogram with power in linear units
    baseline_levels: np.array
        Nslices values with the baseline level for each slice, where the baseline level
        is essentially the noise floor, or power in the non-signal band
    db_range : float >0 
        The start of the signal range. Valid 'pixels' in the spectrogram are defined as those
        that are at least >= dB(baseline_level) + db_range 

    Returns
    -------
    above_baseline : np.array
        Mfreqs x Nslices Boolean array where the pixels above the db_range from baseline
        are True. 
    
    '''
    above_baseline = np.zeros(specgram.shape, dtype=np.bool)
    for i, baseline in enumerate(baseline_levels):
        try:
            above_baseline[:,i] = dB(specgram[:,i])>=dB(baseline)+kwargs['db_range']
        except ValueError:
            pass
    return above_baseline

def get_baseline_level(specgram_slice, **kwargs):
    '''
    Calculates the baseline level of a spectrogram slice. 
    
    Parameters
    ----------
    specgram_slice : np.array 
    threshold : 100>float>0, optional 
        Defaults to 95%ile 
    non_signal_freqs_condition : function 
        Function which identifies the frequencies that don't fall into the 
        signal band of interest. The function returns a boolean np.array with 
        True for those  frequencies which are not in the band. 
    frequencies : np.array
        The centre frequencies in a spectrogram slice.

    Returns
    -------
    baseline : float
    '''
    threshold = kwargs['threshold']
    #print('miaow', specgram_slice.shape, kwargs)
    baseline_freqs = kwargs['non_signal_freqs_condition'](kwargs['frequencies'])
    
    baseline = np.percentile(specgram_slice[baseline_freqs], threshold)
    return baseline

def identify_continuous_fm(fm_regions, **kwargs):
    '''
    '''
    regions_bool = fm_regions>0
    object_labels, num_objects = ndimage.label(regions_bool)
    objects = ndimage.find_objects(object_labels)
    return objects

def find_terminal_frequency(fm_spans, above, freqs, **kwargs):
    '''
    Parameters
    ----------
    fm_spans : list
        List with slice objects denoting the start and end columns of the 
        detected fm regions in the spectrogram. 
    above : np.array
        The boolean np.array showing which pixels are above threshold.
    freqs : np.array
        The frequencies that each of the rows correspond to . 
    
    Returns
    -------
    terminal_frequencies : list
        List with terminal frequencies from each detected FM region. 
    '''
    terminal_frequencies = []
    if len(fm_spans)<1:
        return [np.nan]
    
    for each in fm_spans:
        this_fmseg = above[:,each[0]]
        lowest_row = np.min(np.argwhere(this_fmseg)[:,0])
        term_frequency =  freqs[lowest_row]
        terminal_frequencies.append(term_frequency)
    return terminal_frequencies
    



if __name__=='__main__':
    # randomly sample 50ms audio segments from a randomly chosen file in a folder each time.
    import glob
    import os
    import soundfile as sf
    import matplotlib.pyplot as plt
    plt.rcParams['agg.path.chunksize'] = 10000

    # main_audio_path = '../../individual_call_analysis/annotation_audio'
    # rec_hour = '2018-08-16_2300-2400_annotation_audio'
    # all_files = glob.glob(os.path.join(main_audio_path,rec_hour,'*.WAV'))
    # rec_file = np.random.choice(all_files,1)[0]
    
    # #audio_path = os.path.join(main_audio_path, rec_hour, rec_file)
    
    # #multi_bat_path = os.path.join('/home/tbeleyur/Desktop','multi_batwav.wav')
    # raw_audio, fs = sf.read(rec_file)
    # b,a = signal.butter(4, 80000/fs*0.5, 'highpass')
    # start_time = np.random.choice(np.arange(0,sf.info(rec_file).duration, 0.001)-0.05, 1)
    # stop_time = start_time + 0.05
    # start, stop = int(fs*start_time), int(fs*stop_time)
    # audio = signal.filtfilt(b,a, raw_audio[start:stop,0])
    
    # kwargs = {'inter_peak_difference':250, 
    #           'spectrum_smoothing_width': 100,
    #           'peak_range': 14,
    #           'fs':fs,
    #           'db_range':46}
    
    # dom_freqs = dominant_frequencies(audio, **kwargs)

    # power_spectrum_audio = 20*np.log10(np.abs(np.fft.rfft(audio)))
    # smooth = make_smoothened_spectrum(audio, **kwargs)
    # freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    
    # plt.figure()
    # plt.plot(freqs_audio, power_spectrum_audio)
    # plt.plot(freqs_audio, smooth)
    # inds = [int(np.where(each==freqs_audio)[0]) for each in list(dom_freqs.values())[0]]
    # plt.plot(list(dom_freqs.values())[0], power_spectrum_audio[inds],'*')
    
    
    # lower = lower_minusXdB_peakfrequency(audio, **kwargs)
    # plt.vlines(lower['minus_XdB_frequency'], 0, np.max(power_spectrum_audio))
    
    # plt.figure()
    # plt.specgram(audio, Fs=fs)
    # plt.hlines(dom_freqs['dominant_frequencies'], 0,audio.size/fs)
    # plt.hlines(lower['minus_XdB_frequency'], 0,audio.size/fs,'r')
    
    # testing the identify FM terminal frequencies
    audio_segs = glob.glob('../observed_audio_segs/*.WAV')
    rand_index = int(np.random.choice(np.arange(3400),1))
    audio, fs = sf.read(audio_segs[rand_index])
    
    f,t,spec = signal.spectrogram(audio,fs=fs, noverlap=0.5,nperseg=512)
    non_signal_bool = f <70000    
    fm_freq = f < 98000

    term_freqs = get_FM_terminal_frequencies(audio, overlap=0.5, nperseg=512,
                                fs=fs,
                                threshold=95,
                                non_signal_freqs = non_signal_bool,
                                db_range=46,
                                fm_freqs=fm_freq)
    print(term_freqs)
