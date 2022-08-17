import os
from minma.data import load_psd_dataset, subset_data_paths
from minma.utils import encode_labels
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict


def run_experiments(models, datasets, folds=5):
    
    # load sbj 0
    result_dict = {}
    for fold in range(folds):
        for model in models:
            for dataset_idx, dataset in enumerate(datasets):
                # convert string labels into categorial ints
                int_labels, lbl_corr = encode_labels(dataset['labels'])
                X, y = dataset['integrated_psd'], int_labels
                # cross validate confusion matr
                y_pred =  cross_val_predict(model,X,y,cv=2) # double fold
                result_dict[ f'sbj_{str(dataset_idx)}_{model.name}_{fold}'] = {'y': y, 'y_pred': y_pred, 'model':model.name}
    return result_dict, lbl_corr

