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

from embed_utils import load_data, Embeds, Logger, WordVecPlot
from data_utils import calc_text_uniq_words, clean_text, convert_text2seq, get_embedding_matrix, clean_seq, split_data, get_bow

NAN_WORD = "_NAN_"

def get_kwargs(kwargs):
    parser = argparse.ArgumentParser(description='-f TRAIN_DATA -t TEST_DATA -e EMBEDS_FILE [-l LOGGER_FILE] [--swear-words SWEAR_FILE] [--wrong-words WRONG_WORDS_FILE] [--warm-start FALSE] [--format-embeds FALSE]')
    parser.add_argument('-f', '--train', dest='train', action='store', help='/path/to/trian_file', type=str)
    parser.add_argument('-t', '--test', dest='test', action='store', help='/path/to/test_file', type=str)
    parser.add_argument('-o', '--output', dest='output', action='store', help='/path/to/output_file', type=str)
    parser.add_argument('-e', '--embeds', dest='embeds', action='store', help='/path/to/embeds_file', type=str)
    parser.add_argument('-et', '--embeds_type', dest='embeds_type', action='store', help='fasttext | glove | word2vec', type=str)
    parser.add_argument('-l', '--logger', dest='logger', action='store', help='/path/to/log_file', type=str, default=None)
    parser.add_argument('--swear-words', dest='swear_words', action='store', help='/path/to/swear_words_file', type=str, default=None)
    parser.add_argument('--wrong-words', dest='wrong_words', action='store', help='/path/to/wrong_words_file', type=str, default=None)
    parser.add_argument('--warm-start', dest='warm_start', action='store', help='true | false', type=bool, default=False)
    parser.add_argument('--model-warm-start', dest='model_warm_start', action='store', help='CNN | LSTM | CONCAT | LOGREG | CATBOOST, warm start for several models available', type=str, default=[], nargs='+')
    parser.add_argument('--format-embeds', dest='format_embeds', action='store', help='file | json | pickle | binary', type=str, default='file')
    parser.add_argument('--config', dest='config', action='store', help='/path/to/config.json', type=str, default=None)
    parser.add_argument('--train-clean', dest='train_clean', action='store', help='/path/to/save_train_clean_file', type=str, default='data/train_clean.csv')
    parser.add_argument('--test-clean', dest='test_clean', action='store', help='/path/to/save_test_clean_file', type=str, default='data/test_clean.csv')
    for key, value in iteritems(parser.parse_args().__dict__):
        kwargs[key] = value


def main(*kargs, **kwargs):
    get_kwargs(kwargs)
    train_fname = kwargs['train']
    test_fname = kwargs['test']
    logger_fname = kwargs['logger']
    swear_words_fname = kwargs['swear_words']
    wrong_words_fname = kwargs['wrong_words']
    train_clean = kwargs['train_clean']
    test_clean = kwargs['test_clean']

    # ====Create logger====
    logger = Logger(logging.getLogger(), logger_fname)

    # ====Load data====
    logger.info('Loading data...')
    train_df = load_data(train_fname)
    test_df = load_data(test_fname)

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # ====Fill NANs====
    logger.info('Calculate sentences lengths...')
    train_df.fillna(NAN_WORD, inplace=True)
    test_df.fillna(NAN_WORD, inplace=True)


    # ====Load additional data====
    logger.info('Loading additional data...')
    swear_words = load_data(swear_words_fname, func=lambda x: set(x.T[0]), header=None)
    wrong_words_dict = load_data(wrong_words_fname, func=lambda x: {val[0] : val[1] for val in x})

    tokinizer = RegexpTokenizer(r'\w+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]

    #====Clean texts====
    logger.info('Cleaning text...')
    train_df['comment_text_clean'] = clean_text(train_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
    test_df['comment_text_clean'] = clean_text(test_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)

    train_df['text_len'] = train_df['comment_text_clean'].apply(lambda words: len(words.split()))
    test_df['text_len'] = test_df['comment_text_clean'].apply(lambda words: len(words.split()))


    # ====Save preprocessed data====
    logger.info('Saving preprocessed data...')
    test_df.to_csv(test_clean, index=False, header=True)
    train_df.to_csv(train_clean, index=False, header=True)


if __name__=='__main__':
    main()
