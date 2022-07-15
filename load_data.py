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


def get_all_data():
    fname = 'motor_imagery.npz'
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

def get_subject_data(alldat, subject, recording=0):
    return alldat[subject][0]

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
    fig.savefig('plots/time_series_plot.png',facecolor='white', bbox_inches='tight')
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
    fig.savefig('plots/power_spectra.png',facecolor='white', bbox_inches='tight')
    plt.close(fig)

def subject_to_mne():
    info = mne.create_info(ch_names=['10 Hz sine', '5 Hz cosine'],
                       ch_types=['misc'] * 2,
                       sfreq=sampling_freq)

    simulated_raw = mne.io.RawArray(data, info)
    simulated_raw.plot(show_scrollbars=False, show_scalebars=False)




if __name__ == "__main__":
    ECoG_data =  get_all_data()
    mne_data = get_mne_data()
    mne_tutorial(mne_data)

