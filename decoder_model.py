import os 
import numpy as np
from sklearn import linear_model
from load_data import load_psd_dataset, subset_data_paths, append_dataset, update_labels
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

def filter_label(curr_data, label='rest'):
    # inds = np.where(curr_data['labels'] != label)[0]
    # inds = np.where(label in curr_data['labels'])[0]
    inds = np.where(~pd.Series(curr_data['labels']).str.contains(label).values)
    curr_data['labels'] = curr_data['labels'][inds]
    curr_data['integrated_psd'] =  curr_data['integrated_psd'][inds]
    curr_data['median_psd'] =  curr_data['median_psd'][inds]
    return curr_data

def convert_labels_mvmt_rest(dataset):
    inds = np.where(curr_data['labels'] != 'rest')[0]
    curr_data['labels'][inds] = np.repeat('movement', len(inds))
    return curr_data

def get_class_weight(y):
     # compute class weights
    class_weights = {c: 1/len(np.unique(y)) for c in  np.unique(y)}
    class_weights = compute_class_weight(class_weight=class_weights, classes=np.unique(y),y=y )
    class_weights = {c: weight for c, weight in  zip(np.unique(y),class_weights)}
    return class_weights

# todo move this out to some plotting file?
def plot_conf_mats(conf_mats, model, lbl_corr, title):
    conf_mat = np.sum(np.stack(conf_mats), axis=0)
    df_cm = pd.DataFrame(conf_mat, index = [lbl_corr[key] for key in lbl_corr.keys()],
                    columns =  [lbl_corr[key] for key in lbl_corr.keys()])
    fig  = plt.figure(figsize = (10,7))
    ax = fig.gca()
    sns.heatmap(df_cm, annot=True, ax=ax, fmt='g', cmap='Reds')
    plt.title(title)
    fig.savefig(f'plots/{model}_conf')

def postprocess_classif_metrics(results, model, lbl_corr, title=''):
    conf_mats = []
    df = pd.DataFrame(columns=['model', 'subject','precison','recall','F1'])
    for sbj_key in results.keys():
        y, y_pred = results[sbj_key]['y'], results[sbj_key]['y_pred']
        
        conf_mat = confusion_matrix(y, y_pred)
        conf_mats.append(conf_mat)
        if len(np.unique(y)) > 2: #multi class
            row = {'model': model, 'subject': sbj_key, 'precison': precision_score(y, y_pred,average='weighted'), 'recall': recall_score(y, y_pred, average='weighted')  , 'F1': f1_score(y, y_pred, average='weighted') }
        else: # binary cls
            row = {'model': model, 'subject': sbj_key, 'precison': precision_score(y, y_pred), 'recall': recall_score(y, y_pred)  , 'F1': f1_score(y, y_pred) }
        df = df.append(row, ignore_index=True)
    plot_conf_mats(conf_mats, model, lbl_corr, title)
    print(df)

def combine_datasets(subjects=[0], file_keys=['mvmt|imagery','3s','hfb']):
    """
    important NOTE : WIP a function that combines the pickles so we can test things over multiple subjects / conditons etc.
    MISSING: needs breakpoints for where the data was pasted together 
    """
    paths, titles = subset_data_paths(subjects=subjects, file_keys=file_keys)    
    all_data = {}
    for curr_path, curr_title in zip(paths, titles):
        curr_data = load_psd_dataset(curr_title, curr_path)
        # THIS IS INCOMPLETE  -- need to parse "mvmt" or 'imagery' out of the title
        # use split(title, '_') and then grab the 0 value in the list 
        curr_data['labels'] = update_labels(curr_data['labels'],curr_title.split('_')[0]+'_')
        all_data = append_dataset(all_data, curr_data)
    return all_data

if __name__ == "__main__":
    score_dict = {}
    # load sbj 0
    for i in range(7):
        # multiclass option (WIP)
        curr_data = combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb'])
        # single class option
        # base_path = os.path.join('data',f'sbj_{i}')
        # title = 'mvmt_hfb_3s' # a lot of way to change this title to try different tasks
        # curr_data = load_psd_dataset(title, base_path)

        # filter out rest samples for now
        curr_data = filter_label(curr_data,'rest')

        # switch task to decoding movemment
        # curr_data = convert_labels_mvmt_rest(curr_data)
        # dict_keys(['integrated_psd', 'median_psd', 'sampled_freqs', 'labels'])

        # convert string labels into categorial ints
        # print(curr_data['labels'])
        le = LabelEncoder()
        int_labels = le.fit_transform(curr_data['labels'])
        lbl_corr = {i:key for i,key in enumerate( le.classes_)}
    
        X, y = curr_data['integrated_psd'], int_labels

        # model selection
        # model = linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')
        model = svm.SVC(class_weight='balanced')

        # cross validate prediction accuracy
        curr_scores =  cross_val_score(model,X,y,cv=10)
        score_dict['sbj_'+str(i)] = curr_scores
        
        # cross validate confusion matr
        y_pred =  cross_val_predict(model,X,y,cv=10)
        score_dict['sbj_'+str(i)] = {'y': y, 'y_pred': y_pred}

        # not cross val
        # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2, shuffle=True)        
        # model.fit(train_X,train_y)
        # pred_y = model.predict(test_X)
        # # print out acc
        # acc = (test_y == pred_y).sum() /  len(test_y)
        # # results[i] = acc
        # # accs.append(acc)
        # print(f'\n %.3f labels predicted correctly for sbj {i} \n' % acc)
        # print(test_y, pred_y)

    # barplot_scores(score_dict, 'SVM', title='4 class acc')
    postprocess_classif_metrics(score_dict, 'SVM',lbl_corr, title='4 class acc')

   
    
