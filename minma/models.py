import os 
import numpy as np
from sklearn import linear_model
from minma.data import load_psd_dataset, subset_data_paths, append_dataset, update_labels
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn import svm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix



# # model selection
# # model = linear_model.RidgeClassifier(alpha=.5, class_weight='balanced')
# model = svm.SVC(class_weight='balanced')
# if __name__ == "__main__":
   
