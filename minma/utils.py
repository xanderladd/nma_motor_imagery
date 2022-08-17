


from sklearn.preprocessing import LabelEncoder, RobustScaler


def encode_labels(labels):
     le = LabelEncoder()
     int_labels = le.fit_transform(labels)
     lbl_corr = {i:key for i,key in enumerate( le.classes_)}
     return int_labels, lbl_corr

def get_class_weight(y):
     # compute class weights
    class_weights = {c: 1/len(np.unique(y)) for c in  np.unique(y)}
    class_weights = compute_class_weight(class_weight=class_weights, classes=np.unique(y),y=y )
    class_weights = {c: weight for c, weight in  zip(np.unique(y),class_weights)}
    return class_weights
