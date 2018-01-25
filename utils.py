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
        self.vec_size = 0
        if format in ('json', 'pickle'):
            self.load(fname, format)
        elif w2v_type in ('fasttext', 'glove'):
            self.model, self.vec_size = self._read_word_vec_from_txt(fname, w2v_type)
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

    def _process_line(self, line, separator):
        line = line.rstrip().split(separator)
        word = line[0]
        vec = line[1:]
        return word, [float(val) for val in vec]

    def _read_word_vec_from_txt(self, fname, w2v_type):
        vec_size = [0]
        with open(fname, 'r') as f:
            if (w2v_type == 'fasttext'):
                tech_line = f.readline()
                dict_size, vec_size = self._process_line(tech_line, ' ')
                print('dict_size = {}'.format(dict_size))
                print('vec_size = {}'.format(vec_size))
            model = {}
            for line in tqdm(f, file=sys.stdout):
                word, vec = self._process_line(line, ' ')
                vec = np.asarray(vec).astype(np.float16)
                model[word] = vec
        return model, int(vec_size[0])

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


class WordVecPlot(object):
    def __init__(self, model):
        self.model = model

    def tsne_plot(self, subset, fname):
        labels_init = []
        tokens_init = []

        for word, vec in self.model.items():
            tokens_init.append(vec)
            labels_init.append(word)

        tokens = tokens_init[subset[0]: subset[1]]
        labels = labels_init[subset[0]: subset[1]]


        df = {}
        pca = PCA(n_components=2)
        new_values = pca.fit_transform(tokens)
        df['pca-one'] = new_values[:, 0]
        df['pca-two'] = new_values[:, 1]


        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        fig, ax = plt.subplots(figsize=(35, 25))
        ax.scatter(x, y)

        for i in range(len(x)):
            ax.annotate(labels[i], (x[i], y[i]), textcoords='offset points', xytext=(5,2))

        fig.savefig(fname)
        plt.close(fig)


