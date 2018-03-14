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

    test_predicts_path = "./stacking/catboost_x_test.npy"
    x_test =  np.load(test_predicts_path)
    val_predicts_path = "./stacking/catboost_x_train.npy"
    x_meta = np.load(val_predicts_path)
    # print(x_meta[0,:6])




    # ROC AUC analysis based on https://www.kaggle.com/ogrellier/things-you-need-to-be-aware-of-before-stacking
    class_preds = [c_ for c_ in target_labels]
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    oof = pd.DataFrame(data=x_meta[:,:6], columns=target_labels)
    sub = pd.DataFrame(data=x_test[:,:6], columns=target_labels)

    # oof_probas = (1 + oof[target_labels[0]].rank().values) / (oof.shape[0] + 1)
    # oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
    # hist, edges = np.histogram(oof_logit, density=True, bins=50)
    # s = figure(plot_width=600, plot_height=300,
    #            title="Probability logits for %s using rank()" % 'toxic')
    # s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)
    # show(s)



    figures = []
    for i_class, class_name in enumerate(target_labels):
        s = figure(plot_width=600, plot_height=300,
                   title="Probability logits for %s" % class_name)

        for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
            probas = oof[class_preds[i_class]].values[val_idx]
            p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
            hist, edges = np.histogram(p_log, density=True, bins=50)
            s.line(edges[:50], hist, legend="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
            if n_fold == 3:
                break

        oof_probas = oof[class_preds[i_class]].values
        oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
        hist, edges = np.histogram(oof_logit, density=True, bins=50)
        s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)

        sub_probas = sub[class_name].values
        sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
        hist, edges = np.histogram(sub_logit, density=True, bins=50)
        s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
        figures.append(s)

    # put the results in a column and show
    show(column(figures))


    # figures = []
    # for i_class, class_name in enumerate(target_labels):
    #     s = figure(plot_width=600, plot_height=300,
    #                title="Probability logits for %s using rank()" % class_name)
    #
    #     for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
    #         print(n_fold)
    #         probas = (1 + oof[target_labels[i_class]].iloc[val_idx].values) / (len(val_idx) + 1)
    #         p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
    #         hist, edges = np.histogram(p_log, density=True, bins=50)
    #         #print(edges.shape)
    #         s.line(edges[:50], hist, legend="Fold %d" % n_fold,
    #                color=brewer["Set1"][4][n_fold])
    #         if n_fold == 3:
    #             break
    #
    #     oof_probas = (1 + oof[target_labels[i_class]].values) / (oof.shape[0] + 1)
    #     oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
    #     hist, edges = np.histogram(oof_logit, density=True, bins=50)
    #     s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)
    #
    #     sub_probas = (1 + sub[class_name].values) / (sub.shape[0] + 1)
    #     sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
    #     hist, edges = np.histogram(sub_logit, density=True, bins=50)
    #     s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
    #     figures.append(s)
    #
    # # put the results in a column and show
    # show(column(figures))



    # x_meta = oof.rank().values
    # print(x_meta[0, :6])

    train_y = np.load('./data/train.labels.npy')[:x_meta.shape[0]]
    roc_auc_val = []
    val_size = int(len(train_y) / 10)
    for i in range(10):
        roc_auc_val.append(roc_auc_score(train_y[i * val_size : i * val_size + val_size, :6], x_meta[i * val_size : i * val_size + val_size,:6]))
    roc_auc_mean = sum(roc_auc_val)/len(roc_auc_val)
    print('Mean auc: {0}'.format(roc_auc_mean))

    roc_auc_train = roc_auc_score(train_y[:,:6], x_meta[:,:6])
    print('OOF auc: {0}'.format(roc_auc_train))

    # x_train_meta, x_val_meta, y_train_meta, y_val_meta = train_test_split(x_meta, train_y[:x_meta.shape[0]], test_size=0.20, random_state=42)
    # meta_model = CatBoost(target_labels,
    #                       loss_function='Logloss',
    #                       iterations=1000,
    #                       depth=6,
    #                       learning_rate=0.03,
    #                       rsm=1
    #                       )
    # meta_model.fit(x_train_meta, y_train_meta, eval_set=(x_val_meta, y_val_meta), use_best_model=True)
    # #y_hat_meta = meta_model.predict_proba(x_val_meta)
    #
    # #metrics_meta = get_metrics(y_val_meta, y_hat_meta, target_labels)
    # logger.info('Applying models...')
    #
    # final_predictions = np.array(meta_model.predict_proba(x_test)).T
    #
    # # ====Save results====
    # logger.info('Saving results...')
    # test_ids = test_df["id"].values
    # test_ids = test_ids.reshape((len(test_ids), 1))
    #
    # test_predicts = pd.DataFrame(data=final_predictions, columns=target_labels)
    # test_predicts["id"] = test_ids
    # test_predicts = test_predicts[["id"] + target_labels]
    # submit_path = os.path.join(result_path, "{0}.csv".format('catboost_folds'))
    # test_predicts.to_csv(submit_path, index=False)


if __name__=='__main__':
    main()
