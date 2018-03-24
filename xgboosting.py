import os.path
import numpy as np
from os import listdir

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


from sklearn.metrics import roc_auc_score, roc_curve


def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()


def main():

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    oof_path = '../oof_crawl_wiki'
    test_predicts_path = '../test_crawl_wiki'
    oof = []
    test = []

    for file in listdir(oof_path):
        oof.append(np.load(os.path.join(oof_path,file)))

    x_train = np.concatenate(oof,1)
    for file in listdir(test_predicts_path):
        file = os.path.join(test_predicts_path,file)
        if '.npy' in file:
            print("numpy input")
            test.append(np.load(file))
        else:
            df = pd.read_csv(file)
            probs = np.array(df[target_labels].values)
            test.append(probs)
    x_test = np.stack(test,-1)

    y = np.load('./data/train.labels.npy')[:x_train.shape[0]]
    sub = pd.read_csv('../sample_submission.csv')


    x_train_stacked = np.stack(oof,axis=-1)

    final_score = []
    for i in range(len(listdir(oof_path))):
        final_score.append(multi_roc_auc_score(y, x_train_stacked[:,:,i]))
    print("Initial CV score:", np.mean(final_score))
    estimators = []
    scores =[]
    for i, label in enumerate(target_labels):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        skf.get_n_splits(y[:, i])
        predictions = []
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

            estimator.fit(X_train_, y_train_)
            y_valid_pred = estimator.predict_proba(X_valid_)[:, 1]
            score = roc_auc_score(y_valid_, y_valid_pred)
            prediction = estimator.predict_proba(x_test[:,i,:])[:, 1]
            predictions.append(prediction)
            estimators.append(estimator)
            scores.append(score)
        sub[label] = np.mean(predictions, axis=0)

    print("Post xgboost score:", np.mean(scores))

    result_path = './boosting'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    submit_path = os.path.join(result_path, "{0}.csv".format('xgboost'))
    sub.to_csv(submit_path, index=False)


if __name__=='__main__':
    main()
