'''
Making an 'anonymised' multichannel audio file

This module will combine the audio from multiple single 
channel files into a multichannel audio file. 
'''

import glob
import pandas as pd
import soundfile as sf
import numpy as np 



# find all files
filenames = glob.glob('matching_*.wav')


# load all audio from them 
audio_durns = []
all_audio = {}
for i,each in enumerate(filenames):
    audio, fs = sf.read(each)
    try:
        if audio.shape[0]>1:
            audio = audio[:,0]
    except:
        pass
    all_audio[each] = audio
    audio_durns.append(audio.shape[0])

# generate multich array
multichannel_audio = np.zeros((np.max(audio_durns),
                                len(filenames)))

filename_data = pd.DataFrame(data={'file_name':[0]*len(filenames)})

for i,(keys, audiodata) in enumerate(all_audio.items()):
    filename_data['file_name'][i] = keys
    num_samples = audiodata.size
    multichannel_audio[:num_samples,i] = audiodata

filename_data['channel_num'] = np.arange(1,8)

sf.write('call_alteration_testaudio.wav', multichannel_audio, fs)
filename_data.to_csv('channel_to_filename.csv')