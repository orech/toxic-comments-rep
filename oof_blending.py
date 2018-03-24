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
from sklearn.preprocessing import minmax_scale
from scipy.optimize import minimize


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()

def main():


    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    wiki_oof_path = '../wiki_oof'
    crawl_oof_path = '../crawl_oof'
    glove_oof_path = '../glove_oof'
    wiki_test_path = '../wiki_test'
    crawl_test_path = '../crawl_test'
    glove_test_path = '../glove_test'

    oof_wiki = []
    print("Models with pretrained wiki fasttext vectors and generated OOV vectors")
    for file in listdir(wiki_oof_path):
        print(file)
        oof_wiki.append(np.load(os.path.join(wiki_oof_path,file)))

    x_train_wiki = np.concatenate(oof_wiki,1)
    num_of_models_wiki = len(listdir(wiki_oof_path))

    test_wiki = []
    for file in listdir(wiki_test_path):
        file = os.path.join(wiki_test_path,file)
        if '.npy' in file:
            test_wiki.append(np.load(file))
        else:
            df = pd.read_csv(file)
            probs = np.array(df[target_labels].values)
            test_wiki.append(probs)
    x_test_wiki = np.stack(test_wiki,-1)

    y = np.load('./data/train.labels.npy')[:x_train_wiki.shape[0]]
    sub_crawl_wiki = pd.read_csv('../sample_submission.csv')
    sub_crawl_wiki_glove = pd.read_csv('../sample_submission.csv')
    print("Models with pretrained wiki fasttext vectors and generated OOV vectors")

    x_train_stacked_wiki = np.stack(oof_wiki,axis=-1)

    initial_score_wiki = []
    average_oof_predictions_wiki = np.mean(x_train_stacked_wiki, axis=2)

    for i in range(len(listdir(wiki_oof_path))):
        score = multi_roc_auc_score(y, x_train_stacked_wiki[:,:,i])
        initial_score_wiki.append(score)
        print("Score:", multi_roc_auc_score(y, x_train_stacked_wiki[:,:,i]))
    print("Initial CV score:", np.mean(initial_score_wiki))
    print("Post-averaging CV score:", multi_roc_auc_score(y, average_oof_predictions_wiki))



    weighted_blending_predictions_wiki = 0.03 * x_train_stacked_wiki[:,:,5] +\
                                    0.07 * x_train_stacked_wiki[:,:,4] +\
                                    0.15 * x_train_stacked_wiki[:,:,3] +\
                                    0.25 * x_train_stacked_wiki[:,:,2] +\
                                    0.25 * x_train_stacked_wiki[:,:,1] +\
                                    0.25 * x_train_stacked_wiki[:,:,0]

    test_predictions_wiki = 0.03 * x_test_wiki[:, :, 5] + \
                            0.07 * x_test_wiki[:, :, 4] + \
                            0.15 * x_test_wiki[:, :, 3] + \
                            0.25 * x_test_wiki[:, :, 2] + \
                            0.25 * x_test_wiki[:, :, 1] + \
                            0.25 * x_test_wiki[:, :, 0]

    print("Post-weighted-blending CV score:", multi_roc_auc_score(y, weighted_blending_predictions_wiki))




    print("Models with pretrained crawl fasttext vectors")
    oof_crawl = []
    for file in listdir(crawl_oof_path):
        print(file)
        oof_crawl.append(np.load(os.path.join(crawl_oof_path,file)))

    x_train_crawl = np.concatenate(oof_crawl,1)
    num_of_models = len(listdir(crawl_oof_path))
    x_train_stacked_crawl = np.stack(oof_crawl,axis=-1)

    test_crawl = []
    for file in listdir(crawl_test_path):
        file = os.path.join(crawl_test_path, file)
        if '.npy' in file:
            test_crawl.append(np.load(file))
        else:
            df = pd.read_csv(file)
            probs = np.array(df[target_labels].values)
            test_crawl.append(probs)
    x_test_crawl = np.stack(test_crawl, -1)

    initial_score_crawl = []
    averaged_oof_predictions_crawl = np.mean(x_train_stacked_crawl, axis=2)

    for i in range(len(listdir(crawl_oof_path))):
        print("Model_{0}".format(i))
        sc = multi_roc_auc_score(y,x_train_stacked_crawl[:,:,i])
        initial_score_crawl.append(sc)
        print("Score:", sc)
    print("Initial CV score:", np.mean(initial_score_crawl))

    print("Post-blending CV score:", multi_roc_auc_score(y, averaged_oof_predictions_crawl))




    weighted_blending_predictions_crawl = 0.05 * x_train_stacked_crawl[:,:,6] +\
                                    0.10 * x_train_stacked_crawl[:,:,5] +\
                                    0.15 * x_train_stacked_crawl[:,:,4] +\
                                    0.15 * x_train_stacked_crawl[:,:,3] +\
                                    0.15 * x_train_stacked_crawl[:,:,2] + \
                                    0.2 * x_train_stacked_crawl[:, :, 1] + \
                                    0.2 * x_train_stacked_crawl[:, :, 0]

    test_predictions_crawl = 0.05 * x_test_crawl[:, :, 6] + \
                                          0.10 * x_test_crawl[:, :, 5] + \
                                          0.15 * x_test_crawl[:, :, 4] + \
                                          0.15 * x_test_crawl[:, :, 3] + \
                                          0.15 * x_test_crawl[:, :, 2] + \
                                          0.2 * x_test_crawl[:, :, 1] + \
                                          0.2 * x_test_crawl[:, :, 0]


    print("Post-weighted-blending CV score:", multi_roc_auc_score(y, weighted_blending_predictions_crawl))


    final_predictions_crawl_wiki = (weighted_blending_predictions_wiki + weighted_blending_predictions_crawl) / 2

    final_test_predictions_crawl_wiki = (test_predictions_wiki + test_predictions_crawl) / 2

    sub_crawl_wiki[target_labels] = final_test_predictions_crawl_wiki

    print("Post-crawl-wiki-blending CV score:", multi_roc_auc_score(y, final_predictions_crawl_wiki))


    print("Models with pretrained crawl glove vectors")
    oof_glove = []
    for file in listdir(glove_oof_path):
        print(file)
        oof_glove.append(np.load(os.path.join(glove_oof_path, file)))

    x_train_stacked_glove = np.stack(oof_glove, axis=-1)

    test_glove = []
    for file in listdir(glove_test_path):
        file = os.path.join(glove_test_path, file)
        if '.npy' in file:
            test_glove.append(np.load(file))
        else:
            df = pd.read_csv(file)
            probs = np.array(df[target_labels].values)
            test_glove.append(probs)
    x_test_glove = np.stack(test_glove, -1)



    initial_score = []
    final_oof_predictions_glove = np.mean(x_train_stacked_glove, axis=2)

    for i in range(len(listdir(glove_oof_path))):
        print("Model_{0}".format(i))
        sc = multi_roc_auc_score(y, x_train_stacked_glove[:, :, i])
        initial_score.append(sc)
        print("Score:", sc)
    print("Initial CV score:", np.mean(initial_score))

    print("Post-blending CV score:", multi_roc_auc_score(y, final_oof_predictions_glove))

    weighted_blending_predictions_glove = 0.05 * x_train_stacked_glove[:, :, 5] + \
                                          0.05 * x_train_stacked_glove[:, :, 4] + \
                                          0.10 * x_train_stacked_glove[:, :, 3] + \
                                          0.25 * x_train_stacked_glove[:, :, 2] + \
                                          0.25 * x_train_stacked_glove[:, :, 1] + \
                                          0.3 * x_train_stacked_glove[:, :, 0]

    test_predictions_glove = 0.05 * x_test_glove[:, :, 5] + \
                                          0.05 * x_test_glove[:, :, 4] + \
                                          0.10 * x_test_glove[:, :, 3] + \
                                          0.25 * x_test_glove[:, :, 2] + \
                                          0.25 * x_test_glove[:, :, 1] + \
                                          0.3 * x_test_glove[:, :, 0]


    print("Post-weighted-blending CV score:", multi_roc_auc_score(y, weighted_blending_predictions_glove))



    final_predictions_crawl_wiki_glove = 0.45 * weighted_blending_predictions_wiki +  0.45 * weighted_blending_predictions_crawl +  0.1 * weighted_blending_predictions_glove


    final_test_predictions_crawl_wiki_glove = 0.45 * test_predictions_wiki +  0.45 * test_predictions_crawl +  0.1 * test_predictions_glove

    sub_crawl_wiki_glove[target_labels] = final_test_predictions_crawl_wiki_glove

    print("Post-crawl-wiki-glove-blending CV score:", multi_roc_auc_score(y, final_predictions_crawl_wiki_glove))

    avg_of_wiki_crawl = (averaged_oof_predictions_crawl + average_oof_predictions_wiki ) / 2
    avg_of_all_models = ( averaged_oof_predictions_crawl + average_oof_predictions_wiki + final_oof_predictions_glove) / 3

    print("Simple averaging of all models CV score:", multi_roc_auc_score(y, avg_of_all_models))
    print("Simple averaging of wiki crawl CV score:", multi_roc_auc_score(y, avg_of_wiki_crawl))


    result_path = './final_blending'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    submit_path_crawl_wiki = os.path.join(result_path, "{0}.csv".format('crawl_wiki_blend'))
    sub_crawl_wiki.to_csv(submit_path_crawl_wiki, index=False)

    submit_path_crawl_wiki_glove = os.path.join(result_path, "{0}.csv".format('crawl_wiki_glove_blend'))
    sub_crawl_wiki_glove.to_csv(submit_path_crawl_wiki_glove, index=False)

if __name__=='__main__':
    main()
