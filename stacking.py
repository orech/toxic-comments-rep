import re
import os.path
import argparse
import logging
import numpy as np
from os import listdir

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold



def main():

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    oof_path = '../oof_all'
    test_predicts_path = '../oof_test'
    oof = []
    test = []

    for file in listdir(oof_path):
        oof.append(np.load(os.path.join(oof_path,file)))

    x_train = np.concatenate(oof,1)


    for file in listdir(test_predicts_path):
        file = os.path.join(test_predicts_path,file)
        df = pd.read_csv(file)
        probs = np.array(df[target_labels].values)
        test.append(probs)
    x_test = np.stack(test,-1)

    y = np.load('./data/train.labels.npy')[:x_train.shape[0]]
    sub_lr = pd.read_csv('../sample_submission.csv')
    sub_lgbm = pd.read_csv('../sample_submission.csv')
    sub = pd.read_csv('../sample_submission.csv')


    x_train_stacked = np.stack(oof,axis=-1)

    final_score = []
    for i in range(10):
        for j, label in enumerate(target_labels):
            final_score.append(roc_auc_score(y[:,j], x_train_stacked[:,j,i]))
    estimators = []
    scores_lr = []
    scores_lgbm = []
    scores =[]
    for i, label in enumerate(target_labels):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        skf.get_n_splits(y[:, i])
        predictions = []
        predictions_lr = []
        predictions_lgbm = []
        for train_idx, valid_idx in skf.split(x_train_stacked[:,i,:], y[:, i]):
            X_train_ = x_train_stacked[train_idx,i,:]
            y_train_ = y[train_idx, i]
            X_valid_ = x_train_stacked[valid_idx,i,:]
            y_valid_ = y[valid_idx, i]

            estimator = XGBClassifier(objective= 'rank:pairwise' ,#'binary:logistic'
                                      eval_metric= 'auc',
                                      n_estimators= 100,
                                      learning_rate= 0.1,
                                      max_depth= 3,
                                      min_child_weight= 10,
                                      gamma=0.5,
                                      subsample= 0.6,
                                      colsample_bytree= 1.0, #0.3-0.5
                                      reg_lambda= 0.0, #0.05
                                      reg_alpha= 0.0,
                                      n_jobs=12)
            model = LogisticRegression(verbose=0)
            stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt",
                                         learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8,
                                         bagging_freq=5, reg_lambda=0.2)


            model.fit(X_train_, y_train_)
            stacker.fit(X_train_, y_train_)
            estimator.fit(X_train_, y_train_)

            y_valid_pred = model.predict_proba(X_valid_)[:, 1]
            score = roc_auc_score(y_valid_, y_valid_pred)
            scores_lr.append(score)
            prediction = model.predict_proba(x_test[:, i, :])[:, 1]
            predictions_lr.append(prediction)

            y_valid_pred = stacker.predict_proba(X_valid_)[:, 1]
            score = roc_auc_score(y_valid_, y_valid_pred)
            scores_lgbm.append(score)
            prediction = stacker.predict_proba(x_test[:, i, :])[:, 1]
            predictions_lgbm.append(prediction)


            y_valid_pred = estimator.predict_proba(X_valid_)[:, 1]
            score = roc_auc_score(y_valid_, y_valid_pred)
            prediction = estimator.predict_proba(x_test[:,i,:])[:, 1]
            predictions.append(prediction)
            estimators.append(estimator)
            scores.append(score)
        sub[label] = np.mean(predictions, axis=0)
        sub_lr[label] = np.mean(predictions_lr, axis=0)
        sub_lgbm[label] = np.mean(predictions_lgbm, axis=0)


    print("Initial CV score:", np.mean(final_score))
    print("Post xgboost score:", np.mean(scores))
    print("Post lgbmboost score:", np.mean(scores_lgbm))
    print("Post lr score:", np.mean(scores_lr))



    result_path = './boosting'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    submit_path = os.path.join(result_path, "{0}.csv".format('xgboost'))
    submit_path_lr = os.path.join(result_path, "{0}.csv".format('lr'))
    submit_path_lgbm = os.path.join(result_path, "{0}.csv".format('lgbm'))
    sub.to_csv(submit_path, index=False)
    sub_lr.to_csv(submit_path_lr, index=False)
    sub_lgbm.to_csv(submit_path_lgbm, index=False)


if __name__=='__main__':
    main()
