# Import key functions to handle data
from load_data import get_all_data, get_subject_data, get_epochs, raw_to_signal
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

# NOTE: can move these to utils at some point
def pickle_dataset(hfb_samples,lfb_samples, labels):
    res = {'hfb_samples': hfb_samples, 'lfb_samples':  lfb_samples, 'labels': labels}
    with open('psd_data.pkl','wb') as f:
        pickle.dump(res, f)


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

# def epoch_to_PSD_samples(epoch, window_size=.150, channels=np.arange(46)):
#     epoch_list = segment_by_window(epoch,window_size=window_size)
#     # print("WARNING: taking the median over PSD freqs instead of integrating")
#     hfb_samples, lfb_samples = [], []
#     for e in epoch_list: # maybe not pull chunks out here
#         signal, times  = raw_to_signal(e, channels=channels) 
#         hfb_freqs, hfb_powers = power_spec_from_signals(signal, e.info['sfreq'],spectrum_range=[76,100])
#         lfb_freqs, lfb_powers = power_spec_from_signals(signal, e.info['sfreq'],spectrum_range=[8,32])
#         hfb_samples.append(np.median(hfb_powers,axis=0))
#         lfb_samples.append(np.median(lfb_powers,axis=0))
#     return hfb_samples, lfb_samples

def epoch_to_PSD_samples(epoch, window_size=.150, channels=np.arange(46)):
    epoch_list = segment_by_window(epoch,window_size=window_size)
    # print("WARNING: taking the median over PSD freqs instead of integrating")
    hfb_samples, lfb_samples = [], []
    for e in epoch_list: # maybe not pull chunks out here
        # signal, times  = raw_to_signal(e, channels=channels) 
        import pdb; pdb.set_trace()
        mne.time_frequency.psd_welch()
        hfb_freqs, hfb_powers = power_spec_from_signals(signal, e.info['sfreq'],spectrum_range=[76,100])
        lfb_freqs, lfb_powers = power_spec_from_signals(signal, e.info['sfreq'],spectrum_range=[8,32])
        hfb_samples.append(np.median(hfb_powers,axis=0))
        lfb_samples.append(np.median(lfb_powers,axis=0))
    return hfb_samples, lfb_samples

if __name__ == "__main__":

    event_ids = dict(rest=10, tongue=11, hand=12)
    ECoG_data =  get_all_data()
    subject_data = get_subject_data(ECoG_data, 0, 0)
    # get MNE epochs
    epochs = get_epochs(subject_data, event_ids, load=True)
    hfb_samples, lfb_samples, labels = [], [], np.array([])
    for epoch_idx in range(len(epochs)):
        hfb, lfb = epoch_to_PSD_samples(epochs[epoch_idx], window_size=2.9)
        hfb_samples += hfb
        lfb_samples += lfb
        labels = np.append(labels,np.repeat(list(epochs[epoch_idx].event_id.keys()), len(hfb)))
    
    hfb_samples = np.vstack(hfb_samples)
    lfb_samples = np.vstack(lfb_samples)
    pickle_dataset(hfb_samples,lfb_samples, labels)


    # raw = epochs[0]
    # # pull out window params (arbitary)
    # fs = raw.info['sfreq']
    # raw.crop(0,1)
    # # single epoch
    # import pdb; pdb.set_trace()
    # signal, times = raw_to_signal(raw, channels=[1]) # maybe not pull chunks out here

    # # analyze_signal_times(signal, times)

    # # select PSD 
    # hfb_freqs, hfb_powers = power_spec_from_signals(signal, raw.info['sfreq'],spectrum_range=[76,100])
    # lfb_freqs, lfb_powers = power_spec_from_signals(signal, raw.info['sfreq'],spectrum_range=[8,32])
    # freqs, powers = power_spec_from_signals(signal, raw.info['sfreq'],spectrum_range=[0,520])

    # analyze_freqs_and_powers(hfb_freqs, hfb_powers, ext='_hfb')
    # analyze_freqs_and_powers(lfb_freqs, lfb_powers, ext='_lfb')
    # analyze_freqs_and_powers(freqs, powers, ext='_all')
    # plt.plot(freqs,powers)



    
    