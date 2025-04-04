

"""
# This code does all the preprocessing of the individual MEG files, and then saves the preprocessed
file in the raw data folder. This code is meant for tasks with events denoted.

Based on MNE-Python
"""

#%% Import modules
# Include all relevant modules
import os
import os.path as op
import sys
import numpy as np
import time
import shutil
from scipy.io import savemat
import scipy.io as sio
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mne
plt.ion()

#%% Check string for file of interest
# The inputs in this section will vary depending on the file name
directory = ''      #include the complete directory of the data files
data_read = 'rawdata'
subject_id = 'sub-'   #patient ID
session = 'ses-meg01'
modality = 'meg'
task = 'SEF'
side = 'ul'
run = '01'

if(side == 'None'):
    side = ''
    
task_id = task + side
path = (directory + data_read + '/' + subject_id + '/' + session + '/' + modality + '/' + subject_id + '_' + session +
            '_task-' + task_id + '_run-' + run)
raw_file = (path + '_proc-tsss_meg.fif')

#%% Load file
raw = mne.io.read_raw_fif(raw_file,preload=True)
raw.plot()

#%% Separate MEG data
# Only the MEG, STIM, ECG and EOG channels were included
raw_meg = raw.copy().pick(['meg','stim','ecg','eog'])

#%% Notch filter
freqs = (60,120,180,240,300,360)
raw_meg.notch_filter(freqs=freqs,phase='zero-double')

#%% ECG SSP
if('ECG061' in raw_meg.ch_names):
    ecg_proj,ecg_events = mne.preprocessing.compute_proj_ecg(raw_meg,n_grad=1,n_mag=1,n_eeg=0,reject=None,ch_name='ECG061')
    
#%% EOG SSP
if('EOG062' in raw_meg.ch_names):
    eog_proj1,eog_events1 = mne.preprocessing.compute_proj_eog(raw_meg,n_grad=1,n_mag=1,n_eeg=0,reject=None,ch_name='EOG062')
    
#%% EOG SSP
if('EOG063' in raw_meg.ch_names):
    eog_proj2,eog_events2 = mne.preprocessing.compute_proj_eog(raw_meg,n_grad=1,n_mag=1,n_eeg=0,reject=None,ch_name='EOG063')
    
#%% Add SSPs
if 'ecg_proj' in locals():
    raw_meg.add_proj(ecg_proj)
if 'eog_proj1' in locals():
    raw_meg.add_proj(eog_proj1)
if 'eog_proj2' in locals():
    raw_meg.add_proj(eog_proj2)
    
#%% Apply SSPs
raw_meg.apply_proj()

#%% High pass
raw_meg.filter(l_freq=1,h_freq=None,phase='zero-double')

#%% #%% ICA preprocessing
ica = mne.preprocessing.ICA(n_components=None,random_state=97,method='picard',max_iter=1000)
ica.fit(raw_meg,picks='meg',reject_by_annotation=True)

#%% Determine components to exclude
ica.plot_sources(raw_meg,start=0,stop=10,show_scrollbars=True)
ica.plot_components(inst=raw_meg)         

#%% Exclude components
ica.exclude = []    # include all the components with artifacts
ica_rank = len(ica.exclude)
ica.apply(inst=raw_meg)
raw_meg.plot()

#%% Plot PSD
psd = raw_meg.compute_psd(method='welch',fmin=1,fmax=330)
psd.plot()

#%% Save preprocessed file
preproc_file_name = (subject_id + '_' + session + '_task-' + task_id + '_run-' + run + '_proc-preproc_meg.fif')
raw_meg.copy().save(preproc_file_name,picks='all')
