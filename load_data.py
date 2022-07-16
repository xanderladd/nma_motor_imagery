# @title Data retrieval
import os, requests
import numpy as np
import matplotlib.pyplot as plt
# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap


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

def get_subject_data(alldat, subject, session=0):
    return alldat[subject][session]

def get_mne_data():
    # Get the data path for the MNE example data
    raw_fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    # Load the file of example MNE data
    raw = io.read_raw_fif(raw_fname, preload=True, verbose=False)
    # Select EEG channels from the dataset
    raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')
    # Grab the sampling rate from the data
    fs = raw.info['sfreq']
    import pdb; pdb.set_trace()
    # Settings for exploring an example channel of data
    ch_label = 'EEG 058'
    t_start = 20000
    t_stop = int(t_start + (10 * fs))

def subject_to_mne(subject_data):

    # create mne info data structure
    n_channels = len(subject_data['locs'])
    sampling_freq = subject_data['srate']  # in Hertz
    info = mne.create_info(n_channels, sfreq=sampling_freq, ch_types='ecog')

    # initialise mne raw data struct
    data = subject_data['V'].T
    raw = mne.io.RawArray(data, info)

    # create event array with [onset, duration, trial_type]
    onset = np.concatenate((subject_data['t_off'], subject_data['t_on']))
    onset = np.sort(onset)
    trial_type = np.insert(subject_data['stim_id'], range(1, len(subject_data['stim_id'])+1, 1), 10)
    duration = np.diff(onset, append=onset[-1]+(onset[-1]-onset[-2]))
    event = np.array([onset, duration, trial_type]).T
    event = np.delete(event, -1, 0)

    # add events as annotations to the raw data
    subject_data.set_annotations(mne.annotations_from_events(event, sfreq=1000))

    # TODO: add electrode locations to raw data
    return raw



if __name__ == "__main__":
    ECoG_data =  get_all_data()
    mne_data = get_mne_data()

    # convert NMA dataset to MNE raw format
    alldat = get_all_data()
    sub_0_real = get_subject_data(alldat, 0, session=0)
    raw_sub_0_real = subject_to_mne(sub_0_real)