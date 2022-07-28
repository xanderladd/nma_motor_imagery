# from minma.data import combine_datasets, filter_label
# from sklearn import linear_model

# import os
# if __name__ == "__main__":
#     os.chdir('../')
#     # curr_data = combine_datasets(subjects=[i], file_keys=['mvmt|imagery','3s','hfb'], norm_type='preceeding_rest_norm')
#     for i in range(7):
#         # single class option
#         base_path = os.path.join('data',f'sbj_{i}')
#         title = 'mvmt_hfb_3s' # a lot of way to change this title to try different tasks
#         curr_data = load_psd_dataset(title, base_path)
#         # filter out rest samples for now
#         curr_data = filter_label(curr_data,'rest')
#         datasets.append(curr_data)
    
#     models = [ svm.SVC(class_weight='balanced'),
#                 linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')]
from minma.data import combine_datasets, filter_label, load_psd_dataset
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
        base_path = os.path.join('data',f'sbj_{i}')
        title = 'imagery_hfb_3s' # a lot of way to change this title to try different tasks
        curr_data = load_psd_dataset(title, base_path)
        # filter out rest samples for now
        curr_data = filter_label(curr_data,'rest')
        datasets.append(curr_data)

    # models
    # model = linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')
    # models = [ svm.SVC(class_weight='balanced'),
    #             linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')]
    # names = ['svm', 'ridge_reg']
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, reg_alpha=0.5)
    models = [ svm.SVC(class_weight='balanced'), KNeighborsClassifier(n_neighbors=15), xgb_model]
    names = ['svm', 'knn', 'xgb']
    for model, name in zip(models,names):
        model.name = name

    result_dict, lbl_corr= run_experiments(models, datasets)
    postprocess_classif_metrics(result_dict,lbl_corr, prefix='hand_v_tongue_imagery', title='4 class acc')
