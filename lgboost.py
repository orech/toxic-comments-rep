import re
import os.path
import argparse
import logging
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import cross_val_score



from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

try:
    import cPickle as pickle
except ImportError:
    import pickle



UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4

"""
Main function
Input: pandas Series and a feature engineering function
Output: pandas Series
"""


def engineer_feature(series, func, normalize=True):
    feature = series.apply(func)

    if normalize:
        feature = pd.Series(z_normalize(feature.values.reshape(-1, 1)).reshape(-1, ))
    feature.name = func.__name__
    return feature


"""
Engineer features
Input: pandas Series and a list of feature engineering functions
Output: pandas DataFrame
"""


def engineer_features(series, funclist, normalize=True):
    features = pd.DataFrame()
    for func in funclist:
        feature = engineer_feature(series, func, normalize)
        features[feature.name] = feature
    return features


"""
Normalizer
Input: NumPy array
Output: NumPy array
"""
scaler = StandardScaler()


def z_normalize(data):
    scaler.fit(data)
    return scaler.transform(data)


"""
Feature functions
"""


def asterix_freq(x):
    return x.count('!') / len(x)


def uppercase_freq(x):
    return len(re.findall(r'[A-Z]', x)) / len(x)


def main():

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    test_predicts_path = "./catboost/catboost_x_test.npy"
    x_test =  np.load(test_predicts_path)
    val_predicts_path = "./catboost/catboost_x_train.npy"
    x_meta = np.load(val_predicts_path)
    train_y = np.load('./data/train.labels.npy')[:x_meta.shape[0]]
    sub = pd.read_csv('../sample_submission.csv')

    train = pd.read_csv('./data/train.csv').fillna(' ')
    test = pd.read_csv('./data/test.csv').fillna(' ')
    INPUT_COLUMN = "comment_text"

    # Engineer features
    feature_functions = [len, asterix_freq, uppercase_freq]
    features = [f.__name__ for f in feature_functions]
    F_train = engineer_features(train[INPUT_COLUMN], feature_functions)
    F_test = engineer_features(test[INPUT_COLUMN], feature_functions)


    # print("X_train shape: {0}".format(F_train[features].as_matrix().shape))
    # print("x_meta shape: {0}".format(x_meta.shape))
    X_train = np.hstack([F_train[features].as_matrix()[:x_meta.shape[0]], x_meta])
    X_test = np.hstack([F_test[features].as_matrix(), x_test])


    stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt",
                                 learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8,
                                 bagging_freq=5, reg_lambda=0.2)

    scores = []
    for i,label in enumerate(target_labels):
        print(label)
        score = cross_val_score(stacker, X_train, train_y[:,i], cv=5, scoring='roc_auc')
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(X_train, train_y[:,i])
        sub[label] = stacker.predict_proba(X_test)[:, 1]
    print("CV score:", np.mean(scores))

    result_path = './stacking'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    submit_path = os.path.join(result_path, "{0}.csv".format('lgboost_folds'))
    sub.to_csv(submit_path, index=False)


if __name__=='__main__':
    main()
