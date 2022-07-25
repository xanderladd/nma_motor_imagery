import os 
import numpy as np
from sklearn import linear_model
from load_data import load_psd_dataset, subset_data_paths
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import LabelEncoder
from sklearn import svm

def filter_label(dataset, label='rest'):
    inds = np.where(curr_data['labels'] != label)[0]
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
    

if __name__ == "__main__":
    accs = []
    # load sbj 0
    for i in range(7):
        base_path = os.path.join('data',f'sbj_{i}')
        title = 'mvmt_hfb_3s'
        curr_data = load_psd_dataset(title, base_path)
        # filter out rest samples for now
        curr_data = filter_label(curr_data,'rest')

        # switch task to decoding movemment
        # curr_data = convert_labels_mvmt_rest(curr_data)

        #dict_keys(['integrated_psd', 'median_psd', 'sampled_freqs', 'labels'])
        # convert string labels into categorial ints
        int_labels = LabelEncoder().fit_transform(curr_data['labels'])
    
        X, y = curr_data['integrated_psd'], int_labels
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2, shuffle=True)
        
        # class_weight =  get_class_weight(y)
        
        # model selection
        # model = linear_model.RidgeClassifier(alpha=.5, class_weight=class_weights)
        model = svm.SVC(class_weight='balanced')

        model.fit(train_X,train_y)
        pred_y = model.predict(test_X)

        # print out acc
        acc = (test_y == pred_y).sum() /  len(test_y)
        accs.append(acc)
        print(f'%.3f labels predicted correctly for sbj {i}' % acc)
        # print(test_y, pred_y)
    print(f'\n \n mean acc: {np.mean(accs)}, std acc {np.std(accs)}')