# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:24:02 2023

@author: srdas
"""

#%% Import modules
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
directory = 'U:/shared/database/meg-ieeg-UNMC/'
data_read = 'rawdata'
subject_id = 'sub-unmc1156'
session = 'ses-meg01'
modality = 'meg'
task = 'spont'
side = 'None'
run = '02'

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
#explained_var_ratio = ica.get_explained_variance_ratio(raw_meg)


#%% Determine components to exclude
ica.plot_sources(raw_meg,start=0,stop=10,show_scrollbars=True)
ica.plot_components(inst=raw_meg)         

#%% Exclude components
ica.exclude = [0,1,2,34]
ica_rank = len(ica.exclude)
ica.apply(inst=raw_meg)
raw_meg.plot()

#%% Plot PSD
psd = raw_meg.compute_psd(method='welch',fmin=1,fmax=330)
psd.plot()

#%% Create new events
events_new = mne.make_fixed_length_events(raw_meg, id=1, start=0,stop=100,duration=0.5,first_samp=True)
raw_with_events = raw_meg.add_events(events_new)

#%% Save raw file with events
preproc_file_name = (subject_id + '_' + session + '_task-' + task_id + '_run-' + run + '_proc-preproc_meg.fif')
preproc_file = ('U:/shared/users/sdas/meg-UNMC_results/' + subject_id + '/spont/preproc_data/' + preproc_file_name)
raw_meg.copy().save(preproc_file, picks='all',tmin=0,tmax=None,proj=True,fmt='double',overwrite=True)
