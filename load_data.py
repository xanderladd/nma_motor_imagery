# @title Data retrieval
import os, requests
import numpy as np
import matplotlib.pyplot as plt
# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap

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
    # Grab the sampling rate from the data
    fs = raw.info['sfreq']
    import pdb; pdb.set_trace()
    # Settings for exploring an example channel of data
    ch_label = 'EEG 058'
    t_start = 20000
    t_stop = int(t_start + (10 * fs))

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
    data = subject_data['V'].astype('float32').T
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

    mne_data = get_mne_data()

    # Data from NMA
    ECoG_data =  get_all_data()
    event_ids = dict(rest=10, tongue=11, hand=12)
    subject_data = get_subject_data(ECoG_data, 0, 0)

    # MNE data formats
    raw = get_raw(subject_data)
    epochs = get_epochs(subject_data, event_ids)
    evokeds = get_mean_evokeds(epochs)
