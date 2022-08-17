from minma.data import (get_all_data, get_subject_data, get_raw, get_epochs, get_mean_evokeds, get_events)
import numpy as np
from nilearn import plotting
from nimare import utils
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import preprocessing

from xgboost import XGBClassifier

import pandas as pd

import os

import mne

from minma.viz import postprocess_classif_metrics

def get_mne_data(subject=0, session=0, epoch_with_rest=False):
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
    epochs = get_epochs(subject_data, event_ids, include_rest=epoch_with_rest)
    # average over each event type
    evokeds = get_mean_evokeds(epochs)

    return events, event_ids, raw, epochs, evokeds

def make_design_matrix(raw, events, event_ids, pad_constant=True, trial_by_trial=False):

    # get numeric event ids
    event_ids = np.array(list(event_ids.values()))
    # get dimensions for design matrix
    n_event_types = len(event_ids)

    if trial_by_trial:
        n_time_points = (events[0,1]*2)+1
        X = np.zeros((n_time_points, n_event_types))
        for event, event_id in enumerate(event_ids):
            if event_id == 10:
                np.put(X[:, event], [* range(0, int((n_time_points-1)/2))], [1])
            else:
                np.put(X[:, event], [* range(3000, n_time_points)], [1])
    else: 
        n_time_points = raw.get_data().shape[1]
        # create template design matrix
        X = np.zeros((n_time_points, n_event_types))

        for event, event_id in enumerate(event_ids):
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
    
    if pad_constant:
        n_time_points = X.shape[0]
        constant = np.ones(n_time_points)
        X = np.column_stack([constant, X])

    # event_ids specify which column refer to which event type
    # X is the design matrix 
    return X, event_ids

def fit_GLM(y, X, poisson=False):

    if not poisson:
        # fitting linear gaussian GLM
        # theta = np.linalg.inv(X.T @ X) @ X.T @ y
        model = linear_model.LinearRegression()
        model.fit(X, y)
        theta = model.coef_
    else:
        model = linear_model.GammaRegressor()
        model.fit(X, y)
        theta = model.coef_

    return theta, model

def predict_y(model, X, y):

    # predict y with fitted model
    y_hat = model.predict(X)
    r_square = r2_score(y, y_hat)

    return y_hat, r_square

def plot_prediction(y, y_hat, sub, sess, just_prediction=False, trial=None, show=True, save=False, condition=None, r_square=None):

    n_electrodes = y.shape[0]
    # compare prediction from fitted weights with actual y
    fig = plt.figure(figsize=(20, 10))
    for elect in range(n_electrodes):
        ax = plt.subplot(5, 10, elect+1)
        if not just_prediction:
            plt.plot(y[elect, :], color='blue')
        plt.plot(y_hat[elect, :], color='red')
        plt.title(f'r2={r_square[elect]}', loc='center')

    if trial != None:
        title = f"yhat_sub-{sub}_ses-{sess}_trial-{trial}_cond-{condition}"
    else:
        title = f"yhat_sub-{sub}_ses-{sess}"

    plt.suptitle(title)

    if show:
        plt.show()

    if save:
        fig.savefig(f'plots/{title}.png', bbox_inches='tight')
    
    plt.close()

def plot_theta(locs, theta, event_ids, X_event_ids, sub, sess, trial=None, show=True, save=False, condition=None):

    # plot the weights on an electrode
    if trial != None:
        fig = plt.figure(figsize=(12,4))
        ax = plt.subplot(1, 1, 1)
        theta_col_index = np.where(X_event_ids==event_ids[condition])
        theta_col_index = theta_col_index[0][0] + 1

        print(theta_col_index)

        plotting.plot_markers(theta[:, theta_col_index], locs, axes=ax)
        title = f"brain_sub-{sub}_ses-{sess}_trial-{trial}_cond-{condition}"
        plt.title(title, loc='center')

    else:
        event_ids_rev = {v: k for k, v in event_ids.items()}
        fig = plt.figure(figsize=(12, 14))
        for i, condition in enumerate(X_event_ids):
            ax = plt.subplot(3, 1, i+1)
            plotting.plot_markers(theta[:, i+1], locs, axes=ax)
            title = f"brain_sub-{sub}_ses-{sess}_cond-{event_ids_rev[condition]}"
            plt.title(title, loc='center')

    if show:
        plt.show()

    if save:
        fig.savefig(f'plots/{title}.png', bbox_inches='tight')
    
    plt.close()

def run_glm(sub=0, sess=0, poisson=False, trial_by_trial=False):

    # Data from NMA
    ECoG_data = get_all_data()
    # get data for 1st subject, session i.e. when real movements were done
    subject_data = get_subject_data(ECoG_data, subject=sub, session=sess)
    # get all mne specific data structures
    events, event_ids, raw, epochs, evokeds = get_mne_data(subject=sub, session=sess, epoch_with_rest=True)

    if sess == 0:
        sess = 'real'
    else:
        sess = 'imagery'

    if poisson:
        lnk = 'gamma'
    else:
        lnk = 'gaussian'

    if trial_by_trial:
        freq_filt_epochs = epochs.load_data().copy().filter(70, 115)
        conditions = list(event_ids.keys())
        conditions.remove('rest')
        for condition in conditions:
            power_y = (freq_filt_epochs[condition].get_data()*(10**6))**2
            filt_power_y = mne.filter.filter_data(power_y, 1000, 0, 20)
            n_trials = filt_power_y.shape[0]
            n_electrodes = filt_power_y.shape[1]
            theta = np.empty((n_electrodes, n_trials))
            for trial in range(n_trials):
                print(condition, trial)
                y = filt_power_y[trial, :, :]
                X, X_event_ids = make_design_matrix(raw, events, event_ids, trial_by_trial=True)
                contrast = X_event_ids.copy()
                contrast = np.select([contrast==event_ids[condition], contrast==10],[1, -1], 0)
                contrast = np.insert(contrast, 0, 1)
                X = X * contrast
                n_electrodes = y.shape[0]
                theta_t = np.empty((n_electrodes))
                for elect in range(n_electrodes):
                    y_i = y[elect, :]
                    # Get the MLE weights for the LG model
                    theta_i, model = fit_GLM(y_i, X, poisson)
                    # store fitted parameters for current condition
                    condition_index = np.where(contrast==1)[0][1]
                    theta_t[elect] = theta_i[condition_index]
                theta[:, trial] = theta_t
            data_file = f'data/theta_sub-{sub}_ses-{sess}_cond-{condition}_lnk-{lnk}.csv'
            np.savetxt(data_file, theta, delimiter=",")
    else:
        X, X_event_ids = make_design_matrix(raw, events, event_ids)
        freq_filt_raw = raw.copy().filter(70, 115)
        power_y = (freq_filt_raw.get_data()*(10**6))**2
        filt_power_y = mne.filter.filter_data(power_y, 1000, 0, 20)

        n_electrodes = y.shape[0]
        theta = np.empty((n_electrodes, 4))
        y_hat = np.empty_like(y)

        # fit the GLM and get theta weights for each electrode
        for elect in range(n_electrodes):
            y_i = y[elect, :]
            # Get the MLE weights for the LG model
            theta_i = fit_GLM(y_i, X, poisson)
            # predict y with fitted parameters
            y_hat_i = predict_y(X, theta_i)
            # store fitted parameters
            theta[elect, :] = theta_i
            # store predicted y
            y_hat[elect, :] = y_hat_i

        # compare prediction from fitted weights with actual y
        plot_prediction(y, y_hat, sub, sess, just_prediction=True)

        # plot the weights on an electrode
        plot_theta(subject_data['locs'], theta, event_ids, X_event_ids, sub, sess)

    return 1

def create_Xy(subject, sessions=['real', 'imagery'], only_sessions=False, only_motor=False):

    data_path = 'data'
    files = os.listdir(data_path)
    labels_all = []
    theta_all = []

    for file in files:
        info = file.split('.')[0].split('_')
        sub = int(info[1].split('-')[1])
        sess = info[2].split('-')[1]
        if sub == subject and sess in sessions:
            cond = info[-2]
            if only_sessions:
                label = sess
            elif only_motor:
                label = cond.split('-')[1]
            else:
                label = sess + '_' + cond.split('-')[1]
            theta = pd.read_csv(os.path.join(data_path, file), header=None)
            theta = theta.to_numpy()
            labels = [label for i in range(theta.shape[1])]
        else:
            continue
        theta_all.append(theta)
        labels_all += labels
    theta_all = np.hstack(theta_all)
    X, y = theta_all.T, labels_all
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    y = np.asarray(y)
    return X_scaled, y

def decode(X, y, XGB=False):

    if XGB:
        model = XGBClassifier()
    else:
        model = svm.SVC(class_weight='balanced')

    le = LabelEncoder()
    int_labels = le.fit_transform(y)
    y = int_labels
    lbl_corr = {i:key for i,key in enumerate(le.classes_)}

    # cross validate confusion matr
    y_pred = cross_val_predict(model, X, y, cv=10)

    return y_pred, y, lbl_corr

def tsne(X, y):

    from sklearn.manifold import TSNE
    import seaborn as sns

    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(X)
    
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y)

    plt.title('tSNE over electrode weights')
    plt.legend()
    plt.show()

def pca(X, y):

    from sklearn.decomposition import PCA
    import seaborn as sns

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print(pca.explained_variance_ratio_)

    sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], hue=y)

    plt.title('PCA on GLM weights')
    plt.legend()
    plt.show()    

if __name__ == "__main__":

    # for sub in range(0,7):
    #     for sess in [0, 1]:
    #         run_glm(sub=sub, sess=sess, poisson=False, trial_by_trial=True)

    score_dict = {}
    for sub in range(0,7):
        X, y = create_Xy(sub, ['real', 'imagery'], only_sessions=False)
        y_pred, y, lbl_corr = decode(X, y, XGB=True)
        score_dict['sbj_'+str(sub)] = {'y': y, 'y_pred': y_pred}

    postprocess_classif_metrics(score_dict, 'SVM', lbl_corr, title='4 class acc')

    # sub = 0

    # X, conditions = create_Xy(sub, ['real', 'imagery'], only_sessions=False)
    
    # print(np.where(conditions in ['imagery_hand','imagery_tongue']))
