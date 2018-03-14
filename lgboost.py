import re
import os.path
import argparse
import logging
from six import iteritems
import numpy as np

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale

from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file, show, output_notebook
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure
from bokeh.palettes import brewer
from bokeh.io import export_png
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


output_file('plot')
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

try:
    import cPickle as pickle
except ImportError:
    import pickle

from embed_utils import load_data, Embeds, Logger, clear_embedding_list, read_embedding_list
from data_utils import calc_text_uniq_words, clean_texts, convert_text2seq, get_embedding_matrix, clean_seq, split_data, get_bow, tokenize_sentences, convert_tokens_to_ids
from models import get_cnn, get_lstm, get_concat_model, save_predictions, get_tfidf, get_most_informative_features, get_2BiGRU, get_BiGRU_2dConv_2dMaxPool, get_2BiGRU_BN, get_2BiGRU_GlobMaxPool
from train import train, continue_train, Params, _train_model, train_folds, get_model, train_folds_catboost
from metrics import calc_metrics, get_metrics, print_metrics


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4




def main():

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    test_predicts_path = "./stacking/catboost_x_test.npy"
    x_test =  np.load(test_predicts_path)
    val_predicts_path = "./stacking/catboost_x_train.npy"
    x_meta = np.load(val_predicts_path)
    train_y = np.load('./data/train.labels.npy')[:x_meta.shape[0]]
    sub = pd.read_csv('../sample_submission.csv')
    # print(x_meta[0,:6])
    stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt",
                                 learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8,
                                 bagging_freq=5, reg_lambda=0.2)

    scores = []
    for i,label in enumerate(target_labels):
        print(label)
        score = cross_val_score(stacker, x_meta, train_y[:,i], cv=5, scoring='roc_auc')
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(x_meta, train_y[:,i])
        sub[label] = stacker.predict_proba(x_test)[:, 1]
    print("CV score:", np.mean(scores))

    result_path = './stacking'
    submit_path = os.path.join(result_path, "{0}.csv".format('lgboost_folds'))
    sub.to_csv(submit_path, index=False)


if __name__=='__main__':
    main()
