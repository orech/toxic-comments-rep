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
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold


class CatBoost(object):
    def __init__(self, target_labels, *args, **kwargs):
        self.target_labels = target_labels
        self.n_classes = len(target_labels)
        self.models = [CatBoostClassifier(*args, **kwargs) for _ in range(self.n_classes)]

    def fit(self, X, y, eval_set=None, use_best_model=True):
        assert np.shape(y)[1] == self.n_classes
        for i, model in enumerate(self.models):
            if eval_set is not None:
                eval_set_i = (eval_set[0], eval_set[1][:, i])
            else:
                eval_set_i = None
            model.fit(X, y[:, i], eval_set=eval_set_i, use_best_model=use_best_model)

    def predict(self, X):
        y = []
        for i, model in enumerate(self.models):
            y.append(model.predict(X))
        return np.array(y)

    def predict_proba(self, X):
        y = []
        for i, model in enumerate(self.models):
            y.append(model.predict_proba(X)[:, 1])
        return np.array(y)

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
    sub_catboost = pd.read_csv('../sample_submission.csv')


    x_train_stacked = np.stack(oof,axis=-1)

    final_score = []
    for i in range(10):
        for j, label in enumerate(target_labels):
            final_score.append(roc_auc_score(y[:,j], x_train_stacked[:,j,i]))
    estimators = []
    scores_lr = []
    scores_lgbm = []
    scores_xgb =[]
    scores_catboost = []
    for i, label in enumerate(target_labels):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        skf.get_n_splits(y[:, i])
        predictions = []
        predictions_lr = []
        predictions_lgbm = []
        predictions_catboost = []
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

            catboost = CatBoostClassifier(
                                  loss_function='Logloss',
                                  iterations=1000,
                                  depth=6,
                                  learning_rate=0.03,
                                  rsm=1
                                  )



            model.fit(X_train_, y_train_)
            stacker.fit(X_train_, y_train_)
            estimator.fit(X_train_, y_train_)
            catboost.fit(X_train_, y_train_, eval_set=(X_valid_, y_valid_), use_best_model=True)

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
            scores_xgb.append(score)

            y_valid_pred = catboost.predict_proba(X_valid_)[:, 1]
            score = roc_auc_score(y_valid_, y_valid_pred)
            scores_catboost.append(score)
            prediction = catboost.predict_proba(x_test[:, i, :])[:, 1]
            predictions_catboost.append(prediction)




        sub[label] = np.mean(predictions, axis=0)
        sub_lr[label] = np.mean(predictions_lr, axis=0)
        sub_lgbm[label] = np.mean(predictions_lgbm, axis=0)
        sub_catboost[label] = np.mean(predictions_catboost, axis=0)


    print("Initial CV score:", np.mean(final_score))
    print("Post xgboost score:", np.mean(scores_xgb))
    print("Post lgbmboost score:", np.mean(scores_lgbm))
    print("Post lr score:", np.mean(scores_lr))
    print("Post catboost score:", np.mean(scores_catboost))



    result_path = './stacking_all'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    submit_path = os.path.join(result_path, "{0}.csv".format('xgboost'))
    submit_path_lr = os.path.join(result_path, "{0}.csv".format('lr'))
    submit_path_lgbm = os.path.join(result_path, "{0}.csv".format('lgbm'))
    submit_path_catboost = os.path.join(result_path, "{0}.csv".format('catboost'))
    sub.to_csv(submit_path, index=False)
    sub_lr.to_csv(submit_path_lr, index=False)
    sub_lgbm.to_csv(submit_path_lgbm, index=False)
    sub_catboost.to_csv(submit_path_catboost, index=False)


if __name__=='__main__':
    main()
