import re
import sys
import json
import logging

import pandas as pd
from tqdm import tqdm
import numpy as np
import gensim

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from ggplot import *

try:
    import cPickle as pickle
except ImportError:
    import pickle


def load_data(fname, **kwargs):
    func = kwargs.get('func', None)
    if func is not None:
        del kwargs['func']
    df = pd.read_csv(fname, **kwargs)
    if func is not None:
        return func(df.values)
    return df


class Embeds(object):
    def __init__(self, fname, w2v_type='fasttext', format='file'):
        self.embedding_dim = 0
        self.model = {}
        self.embedding_list = []
        self.embedding_index = {}
        if format in ('json', 'pickle'):
            self.load(fname, format)
        elif w2v_type in ('fasttext', 'glove'):
            self.embedding_list, self.embedding_index, self.embedding_dim = self._read_word_vec_from_txt(fname)
        elif w2v_type == 'word2vec':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=format=='binary')
        else:
            self.model = {}

    def __getitem__(self, key):
        try:
            return self.model[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return self.__getitem__[key] is not None

    def _read_word_vec_from_txt(self, fname):
        embedding_word_index = {}
        embedding_list = []

        f = open(fname, encoding='utf8')

        for index, line in enumerate(f):
            if index == 0:
                continue
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                coefs.shape = 300
            except:
                continue
            embedding_list.append(coefs)
            embedding_word_index[word] = len(embedding_word_index)
        f.close()
        embedding_list = np.array(embedding_list)
        embedding_dim = len(embedding_list[0])
        return embedding_list, embedding_word_index, embedding_dim

    def save(self, fname, format='json'):
        if format == 'json':
            with open(fname, 'w') as f:
                json.dump(self.model, f)
        elif format == 'pickle':
            with open(fname, 'wb') as f:
                pickle.dump(self.model, f)
        return self

    def load(self, fname, format='json'):
        if format == 'json':
            with open(fname) as f:
                self.model = json.load(f)
        elif format == 'pickle':
            with open(fname, 'rb') as f:
                self.model = pickle.load(f)
        return self

    def clean_embedding_list(self, words_dict):
        cleared_embedding_list = []
        cleared_embedding_word_dict = {}

        for word in words_dict:
            if word not in self.embedding_index:
                continue
            word_id = self.embedding_index[word]
            row = self.embedding_list[word_id]
            cleared_embedding_list.append(row)
            cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

        self.embedding_list = cleared_embedding_list
        self.embedding_index = cleared_embedding_word_dict


def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []

    f = open(file_path, encoding='utf8')

    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embedding_list.append(coefs)
        embedding_word_dict[word] = len(embedding_word_dict)
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


class Logger(object):
    def __init__(self, logger, fname=None, format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"):
        self.logFormatter = logging.Formatter(format)
        self.rootLogger = logger
        self.rootLogger.setLevel(logging.DEBUG)

        self.consoleHandler = logging.StreamHandler(sys.stdout)
        self.consoleHandler.setFormatter(self.logFormatter)
        self.rootLogger.addHandler(self.consoleHandler)

        if fname is not None:
            self.fileHandler = logging.FileHandler(fname)
            self.fileHandler.setFormatter(self.logFormatter)
            self.rootLogger.addHandler(self.fileHandler)

    def warn(self, message):
        self.rootLogger.warn(message)

    def info(self, message):
        self.rootLogger.info(message)

    def debug(self, message):
        self.rootLogger.debug(message)

# class WordVecPlot(object):
#     def __init__(self, model):
#         self.model = model
#
#     def tsne_plot(self, subset, fname):
#         labels_init = []
#         tokens_init = []
#
#         for word, vec in self.model.items():
#             tokens_init.append(vec)
#             labels_init.append(word)
#
#         tokens = tokens_init[subset[0]: subset[1]]
#         labels = labels_init[subset[0]: subset[1]]
#
#
#         df = {}
#         pca = PCA(n_components=2)
#         new_values = pca.fit_transform(tokens)
#         df['pca-one'] = new_values[:, 0]
#         df['pca-two'] = new_values[:, 1]
#
#
#         x = []
#         y = []
#         for value in new_values:
#             x.append(value[0])
#             y.append(value[1])
#
#         fig, ax = plt.subplots(figsize=(35, 25))
#         ax.scatter(x, y)
#
#         for i in range(len(x)):
#             ax.annotate(labels[i], (x[i], y[i]), textcoords='offset points', xytext=(5,2))
#
#         fig.savefig(fname)
#         plt.close(fig)


