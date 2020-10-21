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
import pyeeg
from scipy.stats import kurtosis, skew
from scipy.signal import argrelextrema, welch
from scipy.integrate import cumtrapz
import statistics 
import time


### Functions

def eeg_features(data):
    data = np.asarray(data)
    res  = np.zeros([22])
    Kmax = 5
    # M    = 10
    # R    = 0.3
    Band = [1,5,10,15,20,25]
    Fs   = 256
    power, power_ratio = pyeeg.bin_power(data, Band, Fs)
    f, P = welch(data, fs=Fs, window='hanning', noverlap=0, nfft=int(256.))       # Signal power spectrum
    area_freq = cumtrapz(P, f, initial=0)
    res[0] = np.sqrt(np.sum(np.power(data, 2)) / data.shape[0])  # amplitude RMS
    res[1] = statistics.stdev(data)**2                  # variance
    res[2] = kurtosis(data)                             # kurtosis
    res[3] = skew(data)                                 # skewness
    res[4] = max(data)                                  # max amplitude
    res[5] = min(data)                                  # min amplitude
    res[6] = len(argrelextrema(data, np.greater)[0])    # number of local extrema or peaks
    res[7] = ((data[:-1] * data[1:]) < 0).sum()         # number of zero crossings
    res[8] = pyeeg.hfd(data, Kmax)                      # Higuchi Fractal Dimension
    res[9] = pyeeg.pfd(data)                            # Petrosian Fractal Dimension
    res[10] = pyeeg.hurst(data)                          # Hurst exponent
    res[11] = pyeeg.spectral_entropy(data, Band, Fs, Power_Ratio=power_ratio) # spectral entropy (1.21s)
    res[12] = area_freq[-1]                                # total power
    res[13] = f[np.where(area_freq >= res[12] / 2)[0][0]]  # median frequency
    res[14] = f[np.argmax(P)]                                 # peak frequency
    res[15], res[16] = pyeeg.hjorth(data)                  # Hjorth mobility and complexity
    res[17] = power_ratio[0]
    res[18] = power_ratio[1]
    res[19] = power_ratio[2]
    res[20] = power_ratio[3]
    res[21] = power_ratio[4]
    # res[22] = pyeeg.samp_entropy(data, M, R)             # sample entropy
    # res[23] = pyeeg.ap_entropy(data, M, R)             # approximate entropy (1.14s)
    return (res)


def eeg_preprocessing(file, seizures, epoch_length = 10, step_size = 1, start_time = 0):
    start = time.time()

    # reading in data 
    raw = mne.io.read_raw_edf(file)  

    # apply filterbank
    raw = raw.load_data().filter(l_freq=0.25, h_freq=25)    
    channels = raw.ch_names                                  # column names

    # Divide into epochs
    res = []
    while start_time <= max(raw.times) + 0.01 - epoch_length:  # max(raw.times) = 3600
        features = []
        start, stop = raw.time_as_index([start_time, start_time + epoch_length])
        temp = raw[:, start:stop][0]

        # start time as ID
        features.append(start_time)

        # features
        for i in range(23):
            features.extend(eeg_features(temp[i]).tolist())

        # seizure flag for y
        for seizure in seizures[filename]:
            if start_time > seizure[0] and start_time < seizure[1]:
                features.append(1)
            elif start_time + epoch_length > seizure[0] and start_time + epoch_length < seizure[1]:
                features.append(1)
            else:
                features.append(0)

        res.append(features)        
        start_time += step_size
        print("Section ", str(len(res)), "; start: ", start, " ; stop: ", stop)

    # print("Total of ", str(len(res)), "epochs")

    # formatting
    feature_names = ["rms", "variance", "kurtosis", "skewness", "max_amp", "min_amp", "n_peaks", "n_crossings", 
        "hfd", "hurst_exp", "spectral_entropy", "total_power", "median_freq", "peak_freq", 
        "hjorth_mobility", "hjorth_complexity", "power_1hz", "power_5hz", "power_10hz", "power_15hz", "power_20hz"]

    column_names = ["start_time"]
    for channel in channels:
        for name in feature_names:
            column_names.append(channel + "_" + name)
    column_names.append("seizure")

    res = pd.DataFrame(res, columns=column_names)

    end = time.time()
    print("Finished preprocessing ", file, f" took {end - start} seconds")
    return res


def eeg_visualize(raw, start_time, end_time):
    n = 2

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



####### Main
# file path here
folder = "E:\isye6740\project\chb-mit-scalp-eeg-database-1.0.0\chb01"
files = ["chb01_03"]
# files = ["chb01_04", "chb01_15","chb01_16","chb01_18","chb01_21","chb01_26"]

seizures =	{
    "chb01_03": [[2996, 3036]], 
    "chb01_04": [[1467, 1494]],
    "chb01_15": [[1732, 1772]], 
    "chb01_16": [[1015, 1066]],
    "chb01_18": [[1720, 1810]],
    "chb01_21": [[327, 420]],
    "chb01_26": [[1862, 1963]]
}

for filename in files:
    file = os.path.join(folder, filename + '.edf')
    res = eeg_preprocessing(file, seizures)
    res.to_csv(os.path.join(folder, filename + '.csv'), index=False) 

print("done")



    


