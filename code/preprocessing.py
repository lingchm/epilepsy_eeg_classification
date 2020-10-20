######################################################################
# ISYE6740 Project
# EEG epilepsy classification
# preprocessing
#
# @author Jintian Lu, Lingchao Mao
# @date 10/11/2020
######################################################################
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import pyeeg
from scipy.stats import kurtosis
from scipy.signal import argrelextrema
import statistics 


### read in data
# file path here
folder = 'E:\isye6740\project\chb-mit-scalp-eeg-database-1.0.0\chb01'
file = os.path.join(folder, 'chb01_03.edf')

# Method 1 (7s per file)
raw = mne.io.read_raw_edf(file)  
raw = raw.load_data().filter(l_freq=0.25, h_freq=25)    # filterbank
# data = raw.get_data()          # 23 x 921600
# n, m = data.shape
channels = raw.ch_names            # column names
print("done reading", file)

for ch in raw.ch_names:
    print(mne.find_events(raw, stim_channel=ch))

### visualize
'''
# MNE-Python's interactive data browser to get a better visualization
raw.plot()

# select a time frame
start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
temp, times = raw[:, start:stop]
fig, axs = plt.subplots(n)
fig.suptitle('Patient EEG')
plt.xlabel('time (s)')
plt.ylabel('MEG data (T)')
for i in range(n):
    axs[i].plot(times, temp[i].T)
plt.show()
print("done")
'''

### Divide into epochs
epoch_length = 10   # seconds
step_size = 1       # seconds
start_time = 0      # seconds
lst = []
while start_time <= max(raw.times) + 0.01 - epoch_length:  # max(raw.times) = 3600
    start, stop = raw.time_as_index([start_time, start_time + epoch_length])
    temp, times = raw[:, start:stop]
    lst.append(temp)  # 23 x 2560
    start_time += step_size
    print("Section ", str(len(lst)), "; start: ", start, " ; stop: ", stop)
print("Total of ", str(len(lst)), "epochs")
dat = np.asarray(lst)
dat = dat.reshape(len(lst)*len(channels), dat.shape[2])  # 23*3591 = 82593 epochs x 2560 length


### Extract features
def eeg_features(data):
    data = np.asarray(data)
    res  = np.zeros([21])
    Kmax = 5
    Tau  = 4
    DE   = 10
    M    = 10
    R    = 0.3
    Band = [1,5,10,15,20,25]
    Fs   = 256
    power, power_ratio = pyeeg.bin_power(data, Band, Fs)
    res[0] = statistics.mean(data)
    res[1] = kurtosis(data)
    res[2] = statistics.stdev(data)
    res[3] = max(data)
    res[4] = min(data)
    res[5] = len(argrelextrema(data, np.greater)[0])  # number of local extrema or peaks
    res[6] = len(argrelextrema(data, np.less)[0])     # number of local extrema or valleys
    res[7] = np.diff(np.sign(data) != 0).sum()        # number of zero crossings
    res[8] = pyeeg.hfd(data, Kmax)
    res[9] = pyeeg.pfd(data)
    res[10] = pyeeg.hurst(data)
    res[11] = pyeeg.ap_entropy(data, M, R)
    res[12] = pyeeg.spectral_entropy(data, Band, Fs, Power_Ratio = power_ratio)
    res[13] = pyeeg.fisher_info(data,Tau,DE)
    res[14], res[15] = pyeeg.hjorth(data)
    res[16] = power[0]
    res[17] = power[1]
    res[18] = power[2]
    res[19] = power[3]
    res[20] = power[4]
    # res[12] = pyeeg.dfa(data)
    return (res)

res = []
for i in range(dat.shape[1]):
    res.append(eeg_features(dat[i]))
res = np.asarray(res)

