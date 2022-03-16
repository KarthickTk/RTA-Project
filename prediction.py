import numpy as np
import joblib 
from sklearn.ensemble import ExtraTreesClassifier

def ordinal_encoder(input_value, feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats 
    feat_dict = dict(zip(feat_key,feat_val))
    value = feat_dict[input_value]
    return value

def get_prediction(data,model):
    """
    Predict the class of a given data point.

    """
    return model.predict(data)