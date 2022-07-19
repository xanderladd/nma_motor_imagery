# @title Data retrieval
import os, requests
import numpy as np
import matplotlib.pyplot as plt
# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap

# Import some NeuroDSP functions to use with MNE
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.rhythm import compute_lagged_coherence

# Import NeuroDSP plotting functions
from neurodsp.plts import (plot_time_series, plot_power_spectra,
                           plot_bursts, plot_lagged_coherence)
                           
# For converting tal coords to MNI coords
from nimare import utils

def get_all_data(save_to='motor_imagery.npz'):
    fname = save_to
    url = "https://osf.io/ksqv8/download"

    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)
        np.save('motor_imagery.npz',alldat)

    else:
        alldat = np.load(fname, allow_pickle=True)['dat']
    return alldat

def get_subject_data(alldat, subject, session=0):
    return alldat[subject][session]

def get_mne_data():
    # Get the data path for the MNE example data
    raw_fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    # Load the file of example MNE data
    raw = io.read_raw_fif(raw_fname, preload=True, verbose=False)
    # Select EEG channels from the dataset
    raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')
    return raw

def mne_tutorial(raw):
    # Grab the sampling rate from the data
    fs = raw.info['sfreq']
    # Settings for exploring an example channel of data
    ch_label = 'EEG 058'
    t_start = 20000
    t_stop = int(t_start + (10 * fs))
    # Extract an example channel to explore
    sig, times = raw.get_data(mne.pick_channels(raw.ch_names, [ch_label]),
                            start=t_start, stop=t_stop, return_times=True)
    sig = np.squeeze(sig)

    # plot time series
    fig = plt.figure(figsize=(8,3))
    ax = fig.gca()
    # Plot a segment of the extracted time series data
    plot_time_series(times, sig, ax=ax)
    fig.savefig('plots/time_series_plot_tutorial.png',facecolor='white', bbox_inches='tight')
    plt.close(fig)

    # Calculate the power spectrum, using median Welch's & extract a frequency range of interest
    freqs, powers = compute_spectrum(sig, fs, method='welch', avg_type='median')
    freqs, powers = trim_spectrum(freqs, powers, [3, 30])

    # Check where the peak power is
    peak_cf = freqs[np.argmax(powers)]
    print(peak_cf)

    # Plot the power spectra, and note the peak power
    fig = plt.figure()
    ax = fig.gca()
    plot_power_spectra(freqs, powers, ax=ax)
    ax.plot(freqs[np.argmax(powers)], np.max(powers), '.r', ms=12)
    fig.savefig('plots/power_spectra_tutorial.png',facecolor='white', bbox_inches='tight')
    plt.close(fig)

def raw_to_signal(raw, t_start, t_stop, channels=[0]):
    """
    convert from MNE to signal from NeuroDSP for a specific chunk
    """
    if len(channels) > 1: raise NotImplementedError
    channels = np.array(channels, dtype=str)
    # Grab the sampling rate from the data
    # Extract an example channel to explore
    sig, times = raw.get_data(mne.pick_channels(raw.ch_names, channels), start=t_start, stop=t_stop, return_times=True)
    sig = np.squeeze(sig)
    return sig, times

def power_spec_from_signal(sig, sample_freq, spectrum_range=[3,30]):
    # spectrum range frequency is in Hz units
    assert len(spectrum_range) == 2, 'spectrum range must be a 2 elem list'
    # Calculate the power spectrum, using median Welch's & extract a frequency range of interest
    freqs, powers = compute_spectrum(sig, sample_freq, method='welch', avg_type='median')
    freqs, powers = trim_spectrum(freqs, powers, spectrum_range)
    return freqs, powers

def analyze_signal_times(signal, times,  ext=""):
    # plot time series
    fig = plt.figure(figsize=(8,3))
    ax = fig.gca()
    # Plot a segment of the extracted time series data
    plot_time_series(times, signal, ax=ax)
    fig.savefig(f'plots/time_series_plot{ext}.png',facecolor='white', bbox_inches='tight')
    plt.close(fig)

def analyze_freqs_and_powers(freqs, powers, ext=""):
    # Check where the peak power is
    peak_cf = freqs[np.argmax(powers)]
    # Plot the power spectra, and note the peak power
    fig = plt.figure()
    ax = fig.gca()
    plot_power_spectra(freqs, powers, ax=ax)
    ax.plot(freqs[np.argmax(powers)], np.max(powers), '.r', ms=12)
    fig.savefig(f'plots/power_spectra{ext}.png',facecolor='white', bbox_inches='tight')
    plt.close(fig)

def get_events(subject_data):

    # create event array with [onset, duration, trial_type]
    onset = np.concatenate((subject_data['t_off'], subject_data['t_on']))
    onset = np.sort(onset)
    trial_type = np.insert(subject_data['stim_id'], range(1, len(subject_data['stim_id'])+1, 1), 10)
    duration = np.diff(onset, append=onset[-1]+(onset[-1]-onset[-2]))
    events = np.array([onset, duration, trial_type]).T
    events = np.delete(events, -1, 0)

    return events

def get_montage(subject_data, raw):

    # convert tal to
    mni_locs = utils.tal2mni(subject_data['locs'])
    channel_names = raw.ch_names
    loc_dict = dict(zip(channel_names, zip(*mni_locs.T)))
    for channel in channel_names:
        loc_dict[str(channel)] = np.array(loc_dict[str(channel)])
    montage = mne.channels.make_dig_montage(loc_dict, coord_frame='head')

    return montage

def get_raw(subject_data):
    # create mne info data structure
    n_channels = len(subject_data['locs'])
    sampling_freq = subject_data['srate']  # in Hertz
    info = mne.create_info(n_channels, sfreq=sampling_freq, ch_types='ecog')
    # initialise mne raw data struct
    print("\n WARNING: using microvolts instead of volts \n ")
    data_uv = (subject_data['scale_uv'] *  subject_data['V'].astype('float32'))#/10**6
    data = data_uv.T
    raw = mne.io.RawArray(data, info)
    # create event array with [onset, duration, trial_type]
    events = get_events(subject_data)
    # add events as annotations to the raw data
    raw.set_annotations(mne.annotations_from_events(events, sfreq=1000))
    # add electrode locations to raw data
    montage = get_montage(subject_data, raw)
    raw.set_montage(montage)

    return raw

def get_epochs(subject_data, event_ids):

    raw = get_raw(subject_data)
    event = get_events(subject_data)
    epoch = mne.Epochs(raw, event, event_ids, baseline=None, detrend=None, tmin=0, tmax=3)
    
    return epoch

def get_mean_evokeds(epochs):

    evokeds = []

    for event_ids, events in epochs.event_id.items():
        evoked = epochs[event_ids].average()
        evokeds.append(evoked)

    return evokeds



if __name__ == "__main__":

    # NOTE: these are only from the tutorial
    mne_data = get_mne_data()
    mne_tutorial(mne_data)

    # Data from NMA
    ECoG_data =  get_all_data()
    event_ids = dict(rest=10, tongue=11, hand=12)
    subject_data = get_subject_data(ECoG_data, 0, 0)

    # convert to MNE data format
    raw = get_raw(subject_data)
    # pull out window params (arbitary)
    fs = raw.info['sfreq']
    t_start = 20000
    t_stop = int(t_start + (10 * fs)) 
    # convert to NeuroDSP signal
    signal, times = raw_to_signal(raw, t_start, t_stop, channels=[0]) # maybe not pull chunks out here
    analyze_signal_times(signal, times)

    # select PSD 
    hfb_freqs, hfb_powers = power_spec_from_signal(signal, raw.info['sfreq'],spectrum_range=[76,100])
    lfb_freqs, lfb_powers = power_spec_from_signal(signal, raw.info['sfreq'],spectrum_range=[8,32])

    analyze_freqs_and_powers(hfb_freqs, hfb_powers, ext='_hfb')
    analyze_freqs_and_powers(lfb_freqs, lfb_powers, ext='_lfb')


    epochs = get_epochs(subject_data, event_ids)
    evokeds = get_mean_evokeds(epochs)
