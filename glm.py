# %%
from load_data import (get_all_data, get_subject_data, get_raw, get_epochs, get_mean_evokeds, get_events)
import numpy as np
from nilearn import plotting
from nimare import utils
import matplotlib.pyplot as plt

# %%
# Data from NMA
ECoG_data =  get_all_data()
# get data for 1st subject, session i.e. when real movements were done
subject_data = get_subject_data(ECoG_data, subject=0, session=0)

# %%
# convert to MNE data formats

# raw data without any event markings
raw = get_raw(subject_data)
# what event ids refer to (info not available inside sub data given, taken from readme text)
event_ids = dict(rest=10, tongue=11, hand=12)
# mark raw data and divide into separate events
epochs = get_epochs(subject_data, event_ids)
# average over each event type
evokeds = get_mean_evokeds(epochs)

#%%
# construct design matrix consisting of 3 columns each for each event type
events = get_events(subject_data)

X = np.zeros((376400, 3))
for event, event_id in zip(range(0, 3), [12, 11, 10]):
    for i in events[events[:,2]==event_id][:,0]:
        np.put(X[:, event], [* range(i, i+3000)], [1])

# %%
# fitting GLM
y = raw.get_data()
theta_all = np.empty((46, 4))
y_0 = y[0, :]
constant = np.ones_like(y_0)
X = np.column_stack([constant, X])
X = np.column_stack([constant, X])

for elect in range(0, 46):
    y_i = y[elect, :]
    # Get the MLE weights for the LG model
    theta = np.linalg.inv(X.T @ X) @ X.T @ y_i
    theta_all[elect, :] = theta

# %%
# plot thetas for each electrode for hand events
plt.figure(figsize=(8, 8))
locs = subject_data['locs']
plotting.plot_markers(theta_all[:, 1], locs)
plt.show()
# %%
