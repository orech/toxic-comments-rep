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

from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pandas as pd

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

def get_kwargs(kwargs):
    parser = argparse.ArgumentParser(description='--train=$TRAIN_DATA --test=$TEST_DATA --embeds=$EMBEDS_FILE --embeds_type=$EMBEDS_TYPE --train-clean=$TRAIN_CLEAN --test-clean=$TEST_CLEAN --embeds-clean=$EMBEDS_CLEAN --train-labels=$TRAIN_LABELS --config=$CONFIG --output=$OUTPUT_FILE --logger=$LOG_FILE')
    parser.add_argument('-f', '--train', dest='train', action='store', help='/path/to/trian_file', type=str)
    parser.add_argument('-t', '--test', dest='test', action='store', help='/path/to/test_file', type=str)
    parser.add_argument('-o', '--output', dest='output', action='store', help='/path/to/output_file', type=str)
    parser.add_argument('-e', '--embeds', dest='embeds', action='store', help='/path/to/embeds_file', type=str)
    parser.add_argument('-et', '--embeds_type', dest='embeds_type', action='store', help='fasttext | glove | word2vec', type=str)
    parser.add_argument('-l', '--logger', dest='logger', action='store', help='/path/to/log_file', type=str, default=None)
    parser.add_argument('--warm-start', dest='warm_start', action='store', help='true | false', type=bool, default=False)
    parser.add_argument('--model-warm-start', dest='model_warm_start', action='store', help='CNN | LSTM | CONCAT | LOGREG | CATBOOST, warm start for several models available', type=str, default=[], nargs='+')
    parser.add_argument('--format-embeds', dest='format_embeds', action='store', help='file | json | pickle | binary', type=str, default='file')
    parser.add_argument('--config', dest='config', action='store', help='/path/to/config.BiGRU_Dense.json', type=str, default=None)
    parser.add_argument('--train-clean', dest='train_clean', action='store', help='/path/to/save_train_clean_file', type=str, default='data/train.clean.npy')
    parser.add_argument('--test-clean', dest='test_clean', action='store', help='/path/to/save_test_clean_file', type=str, default='data/test.clean.npy')
    parser.add_argument('--embeds-clean', dest='embeds_clean', action='store', type=str, default=None)
    parser.add_argument('--train-labels', dest='train_labels', action='store', type=str, default=None)
    for key, value in iteritems(parser.parse_args().__dict__):
        kwargs[key] = value


def main(*kargs, **kwargs):

    # ============ Parse global parameters ============
    get_kwargs(kwargs)
    train_fname = kwargs['train']
    test_fname = kwargs['test']
    result_fname = kwargs['output']
    embeds_fname = kwargs['embeds']
    logger_fname = kwargs['logger']
    warm_start = kwargs['warm_start']
    model_warm_start = [model.lower() for model in kwargs['model_warm_start']]
    config = kwargs['config']
    train_clean = kwargs['train_clean']
    train_labels = kwargs['train_labels']
    test_clean = kwargs['test_clean']
    embeds_clean = kwargs['embeds_clean']
    result_path = './catboost/'


    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # cnn_model_file = 'data/cnn.h5'
    # lstm_model_file = 'data/lstm_model.h5'
    # gru_model_file = 'data/gru_model.h5'
    # concat_model_file = 'data/concat.h5'
    # cnn_model_file = 'data/cnn.h5'
    # lr_model_file = 'data/{}_logreg.bin'
    # meta_catboost_model_file = 'data/{}_meta_catboost.bin'

    # ==== Create logger ====
    logger = Logger(logging.getLogger(), logger_fname)

    # ==== Load data ====
    logger.info('Loading data...')
    test_df = load_data(test_fname)
    train_x = np.load(train_clean)
    test_x = np.load(test_clean)
    embedding_matrix = np.load(embeds_clean)
    train_y = np.load(train_labels)


    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    num_classes = len(target_labels)


    # ============= Load params of models =============
    params = Params(config)
    models = params.get('models')
    val_predictions_list = []
    test_predictions_list = []

    # ============ Train models =============
    for model_name in models:
        model_func = get_model(model_name, embedding_matrix, params)
        # =========== Training on folds ============
        batch_size = params.get(model_name).get('batch_size')

        logger.debug('Starting {0} training on folds...'.format(model_name))
        models, val_predictions = train_folds_catboost(train_x, train_y, params.get(model_name).get('num_folds'), batch_size, model_func, params.get(model_name).get('optimizer'), logger=logger)
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_predictions_list.append(val_predictions)

        logger.debug('Predicting results...')
        test_predictions = []
        for fold_id, model in enumerate(models):
            test_predictions.append(model.predict(test_x, batch_size=batch_size))
        final_test_predictions = np.ones(test_predictions[0].shape)
        for fold_predict in test_predictions:
            final_test_predictions *= fold_predict
        final_test_predictions **= (1. / len(test_predictions))
        test_predictions_list.append(final_test_predictions)

    x_test = np.concatenate(test_predictions_list, axis=1)
    test_predicts_path = os.path.join(result_path, "catboost_x_test.npy")
    np.save(test_predicts_path, x_test)
    x_meta = np.concatenate(val_predictions_list, axis=1)
    val_predicts_path = os.path.join(result_path, "catboost_x_train.npy")
    np.save(val_predicts_path, x_meta)

    x_train_meta, x_val_meta, y_train_meta, y_val_meta = train_test_split(x_meta, train_y[:x_meta.shape[0]], test_size=0.20, random_state=42)
    meta_model = CatBoost(target_labels,
                          loss_function='Logloss',
                          iterations=1000,
                          depth=6,
                          learning_rate=0.03,
                          rsm=1
                          )
    meta_model.fit(x_train_meta, y_train_meta, eval_set=(x_val_meta, y_val_meta), use_best_model=True)
    #y_hat_meta = meta_model.predict_proba(x_val_meta)

    #metrics_meta = get_metrics(y_val_meta, y_hat_meta, target_labels)
    logger.info('Applying models...')

    final_predictions = np.array(meta_model.predict_proba(x_test)).T

    # ====Save results====
    logger.info('Saving results...')
    test_ids = test_df["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=final_predictions, columns=target_labels)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + target_labels]
    submit_path = os.path.join(result_path, "{0}.csv".format('catboost_folds'))
    test_predicts.to_csv(submit_path, index=False)


if __name__=='__main__':
    main()
