######################################################################
# ISYE6740 Project
# EEG epilepsy classification
# preprocessing
#
# @author Jintian Lyu, Lingchao Mao
# @date 10/11/2020
######################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
# import plotly.plotly as py
import mne

folder = 'E:\Eeg_Raw\chb01'
file = os.path.join(folder, 'chb01_01.edf')

# Method 1 (7s per file)
raw = mne.io.read_raw_edf(file)  
data = raw.get_data()          # 23 x 921600
n, m = data.shape
channels = raw.ch_names            # column names
print("done reading", file)


'''
# Method 2 (10s per file)
import pyedflib

f = pyedflib.EdfReader(file)
n = f.signals_in_file                       # n columns
signal_labels = f.getSignalLabels()         # column names
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)

print("done")

'''

# MNE-Python's interactive data browser to get a better visualization
# raw.plot()

# to select certain magnetometer channels and plot
# picks = mne.pick_types(raw.info, meg='mag', exclude=[])
# print(picks)

# select a time frame
start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
data, times = raw[:, start:stop]
print(times.shape)
print(times.min(), times.max())

fig, axs = plt.subplots(n)
fig.suptitle('Patient EEG')
plt.xlabel('time (s)')
plt.ylabel('MEG data (T)')
for i in range(n):
    axs[i].plot(times, data[i].T)

plt.show()

print("done")

'''
# viusalize
fig, axs = plt.subplots(2)
fig.suptitle('Patient EEG')
axs[0].plot(raw_data[], y)
axs[1].plot(x, -y)
'''


# Find events
# first column contains the sample index when the event occurred
# second column contains the value of the trigger channel immediately before the event occurred
# third column contains the event-id.
events = mne.find_events(raw, stim_channel='STI 014')
print("Total number of events: ", str(len(events)))
print(events[:5])  # events is a 2d array



