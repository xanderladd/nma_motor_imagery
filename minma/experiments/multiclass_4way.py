from minma.data import combine_datasets, filter_label
from minma.run_experiments import run_experiments
from minma.viz import barplot_scores, postprocess_classif_metrics
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import os
if __name__ == "__main__":
    os.chdir('../')
    datasets = []
    # multiclass option 
    for i in range(7):
        # curr_data = combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb'], norm_type='preceeding_rest_norm')
        # curr_data = combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb'], norm_type='robust_scaler')
        # curr_data = combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb'])
        curr_data = combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb', 'delay'])

        # filter out rest samples for now
        curr_data = filter_label(curr_data,'rest')
        datasets.append(curr_data)
    
    # models
    # model = linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')
    # models = [ svm.SVC(class_weight='balanced'),
    #             linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')]
    # names = ['svm', 'ridge_reg']
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42, reg_alpha=0.5)
    models = [ svm.SVC(class_weight='balanced'), KNeighborsClassifier(n_neighbors=15), xgb_model]
    names = ['svm', 'knn', 'xgb']
    for model, name in zip(models,names):
        model.name = name

    result_dict, lbl_corr= run_experiments(models, datasets)
    postprocess_classif_metrics(result_dict,lbl_corr, prefix='delay', title='4 class acc')
