from minma.data import (get_all_data, get_subject_data, get_raw, get_epochs, get_mean_evokeds, get_events)
import numpy as np
from nilearn import plotting
from nimare import utils
import matplotlib.pyplot as plt

def get_mne_data(subject=0, session=0):
    # Data from NMA
    ECoG_data =  get_all_data()
    # get data for 1st subject, session i.e. when real movements were done
    subject_data = get_subject_data(ECoG_data, subject=subject, session=session)

    # convert to MNE data formats

    # array with event info
    events = get_events(subject_data)
    # raw data without any event markings
    raw = get_raw(subject_data)
    # what event ids refer to (info not available inside sub data given, taken from readme text)
    event_ids = dict(rest=10, tongue=11, hand=12)
    # mark raw data and divide into separate events
    epochs = get_epochs(subject_data, event_ids)
    # average over each event type
    evokeds = get_mean_evokeds(epochs)

    return events, event_ids, raw, epochs, evokeds

def make_design_matrix(raw, events, event_ids):

    # get dimensions for design matrix
    n_time_points = raw.get_data().shape[1]
    n_event_types = events.shape[1]

    # create template design matrix
    X = np.zeros((n_time_points, n_event_types))

    # get numeric event ids
    event_ids = list(event_ids.values())

    for event, event_id in zip(range(n_event_types), event_ids):
        # keep the column with trial type info
        trial_type = events[:,2]
        # keep rows with given event
        pick_event_trials = events[trial_type==event_id]
        # get the onsets for trials of a given event
        event_trial_onsets = pick_event_trials[:,0]
        # get the trial duration for the trials of given event
        event_trial_dur = pick_event_trials[:,1]

        # replace zeros in design matrix with ones for whole duration
        # of trial in the column corresponding to that event
        for onset, dur in zip(event_trial_onsets, event_trial_dur):
            np.put(X[:, event], [* range(onset, onset+dur)], [1])

    # event_ids specify which column refer to which event type
    # X is the design matrix 
    return X, event_ids

def fit_GLM(y, X, pad_constant=True):

    if pad_constant:
        n_time_points = X.shape[0]
        constant = np.ones(n_time_points)
        X = np.column_stack([constant, X])

    # fitting GLM
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta
