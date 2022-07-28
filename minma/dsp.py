# Import key functions to handle data
# from load_data import get_all_data, get_subject_data, get_epochs, raw_to_signal, get_raw, pickle_dataset
# Import some NeuroDSP functions to use with MNE
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.rhythm import compute_lagged_coherence

# Import NeuroDSP plotting functions
from neurodsp.plts import (plot_time_series, plot_power_spectra,
                           plot_bursts, plot_lagged_coherence)

import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import mne
import os


def power_spec_from_signals(sigs, sample_freq, spectrum_range=[3,30]):
    """
    wrapper around power_spec_from_signal to handle multidimensional data
    """
    all_freqs, all_powers = [] , []
    if len(sigs.shape) < 2 or sigs.shape[1] > 1:
        for sig_idx in range(sigs.shape[1]):
            curr_freqs, curr_powers = power_spec_from_signal(sigs[:,sig_idx], sample_freq, spectrum_range=spectrum_range)
            all_freqs.append(curr_freqs), all_powers.append(curr_powers)
        all_freqs, all_powers = np.stack(all_freqs).T, np.stack(all_powers).T
    else:
        all_freqs, all_powers = power_spec_from_signal(sigs[:,0], sample_freq, spectrum_range=spectrum_range)
        all_freqs, all_powers =  all_freqs.reshape(-1,1), all_powers.reshape(-1,1)
    return  all_freqs, all_powers

def power_spec_from_signal(sig, sample_freq, spectrum_range=[3,30]):
    # spectrum range frequency is in Hz units
    assert len(spectrum_range) == 2, 'spectrum range must be a 2 elem list'
    # Calculate the power spectrum, using median Welch's & extract a frequency range of interest
    freqs, powers = compute_spectrum(sig, sample_freq, method='welch', avg_type='median', nperseg=.15)
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
    peak_cf = np.diag(freqs[np.argmax(powers,axis=0)])
    # Plot the power spectra, and note the peak power
    fig = plt.figure()
    ax = fig.gca()
    plot_power_spectra(freqs, powers, ax=ax)
    ax.plot(np.diag(freqs[np.argmax(powers,axis=0)]), np.max(powers, axis=0), '.r', ms=12)
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.x_scale

    fig.savefig(f'plots/power_spectra{ext}.png',facecolor='white', bbox_inches='tight')
    plt.close(fig)

def segment_by_window(epoch,window_size=.150):
    """
    NOTE: window size in ms but epoch stop in S
    """
    # define window stop and start times in ms
    start_times, stop_times = np.arange(epoch.tmin, epoch.tmax - window_size, window_size), np.arange(epoch.tmin+window_size, epoch.tmax, window_size)
    epoch_list = []
    [epoch_list.append(copy.deepcopy(epoch).crop(start, stop)) for start, stop in zip(start_times,stop_times)]
    return epoch_list

def epochs_to_PSD_samples(epochs, tmin=None, tmax=None):
    # print("WARNING: taking the median over PSD freqs instead of integrating")
    hfb_samples, lfb_samples, all_samples, labels = [], [], [], []
    for e_idx in range(len(epochs)): # maybe not pull chunks out here
        # signal, times  = raw_to_signal(e, channels=channels) 
        hfb_samples.append(mne.time_frequency.psd_welch(epochs[e_idx], fmin=76, fmax=100, tmin=tmin, tmax=tmax, verbose=False))
        lfb_samples.append(mne.time_frequency.psd_welch(epochs[e_idx], fmin=8, fmax=32, tmin=tmin, tmax=tmax, verbose=False)) 
        all_samples.append(mne.time_frequency.psd_welch(epochs[e_idx], fmin=0, fmax=150, tmin=tmin, tmax=tmax, verbose=False))
        labels.append(list(epochs[e_idx].event_id.keys())[0])
    return np.array(hfb_samples), np.array(lfb_samples), np.array(all_samples), np.array(labels)

def epoch_windows_to_PSD_samples(epochs, w_size, fft_size):
    # print("WARNING: taking the median over PSD freqs instead of integrating")
    hfb_samples, lfb_samples, all_samples, labels = [], [], [], []
    for e_idx in range(len(epochs)): # maybe not pull chunks out here
        # signal, times  = raw_to_signal(e, channels=channels) 
        for w_end in np.arange(epochs[e_idx].tmin+w_size, epochs[e_idx].tmax, w_size):
            w_start = w_end - w_size
            assert w_start >= 0, "window size cannot be bigger than epoch"
            hfb_samples.append(mne.time_frequency.psd_welch(epochs[e_idx], fmin=76, fmax=100, tmin=w_start, tmax=w_end, n_fft=fft_size, verbose=False ))
            lfb_samples.append(mne.time_frequency.psd_welch(epochs[e_idx], fmin=8, fmax=32, tmin=w_start, tmax=w_end, n_fft=fft_size, verbose=False)) 
            all_samples.append(mne.time_frequency.psd_welch(epochs[e_idx], fmin=0, fmax=150, tmin=w_start, tmax=w_end, n_fft=fft_size, verbose=False))
            labels.append(list(epochs[e_idx].event_id.keys())[0])
            # import pdb; pdb.set_trace()
    return np.array(hfb_samples), np.array(lfb_samples), np.array(all_samples), np.array(labels)

def reduce_samples(samples):
    """
    samples is shape 119 x 2 ... 119 trials x (power, freqs)
    """
    # annoying reshape stuff.. splits power and freqs and then fixes awkward array dims
    psd_samples, freq_samples = samples[:,0],  samples[:,1]
    psd_samples, freq_samples = np.stack(psd_samples)[:,0,:,:], np.stack(freq_samples)
    # do median and integral over freq band
    assert len( np.unique(freq_samples,axis=0)) == 1, 'sampling freq. is not uniform'
    integrated_psd_samples = np.trapz(psd_samples,freq_samples[0])
    # print(freq_samples[0])
    median_psd_samples = np.median(psd_samples,axis=1)
    return integrated_psd_samples, median_psd_samples, freq_samples[0]