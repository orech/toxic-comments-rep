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

from utils import load_data, Embeds, Logger
from prepare_data import calc_text_uniq_words, clean_text, convert_text2seq, get_embedding_matrix, clean_seq, split_data, get_bow
from models import get_cnn, get_lstm, get_concat_model, save_predictions, get_tfidf, get_most_informative_features
from train import train, continue_train, Params
from metrics import calc_metrics, get_metrics, print_metrics


def get_kwargs(kwargs):
    parser = argparse.ArgumentParser(description='-f TRAIN_FILE -t TEST_FILE -o OUTPUT_FILE -e EMBEDS_FILE [-l LOGGER_FILE] [--swear-words SWEAR_FILE] [--wrong-words WRONG_WORDS_FILE] [--warm-start FALSE] [--format-embeds FALSE]')
    parser.add_argument('-f', '--train', dest='train', action='store', help='/path/to/trian_file', type=str)
    parser.add_argument('-t', '--test', dest='test', action='store', help='/path/to/test_file', type=str)
    parser.add_argument('-o', '--output', dest='output', action='store', help='/path/to/output_file', type=str)
    parser.add_argument('-e', '--embeds', dest='embeds', action='store', help='/path/to/embeds_file', type=str)
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
    result_fname = kwargs['output']
    embeds_fname = kwargs['embeds']
    logger_fname = kwargs['logger']
    swear_words_fname = kwargs['swear_words']
    wrong_words_fname = kwargs['wrong_words']
    warm_start = kwargs['warm_start']
    model_warm_start = [model.lower() for model in kwargs['model_warm_start']]
    format_embeds = kwargs['format_embeds']
    config = kwargs['config']
    train_clean = kwargs['train_clean']
    test_clean = kwargs['test_clean']

    # cnn_model_file = 'data/cnn.h5'
    # lstm_model_file = 'data/lstm.h5'
    # concat_model_file = 'data/concat.h5'
    # cnn_model_file = 'data/cnn.h5'
    # lr_model_file = 'data/{}_logreg.bin'
    # meta_catboost_model_file = 'data/{}_meta_catboost.bin'

    # ====Create logger====
    logger = Logger(logging.getLogger(), logger_fname)

    # ====Load data====
    logger.info('Loading data...')
    train_df = load_data(train_fname)
    test_df = load_data(test_fname)

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    num_classes = len(target_labels)

    # ====Load additional data====
    logger.info('Loading additional data...')
    swear_words = load_data(swear_words_fname, func=lambda x: set(x.T[0]), header=None)
    wrong_words_dict = load_data(wrong_words_fname, func=lambda x: {val[0] : val[1] for val in x})

    tokinizer = RegexpTokenizer(r'\w+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]

    # ====Load word vectors====
    logger.info('Loading embeddings...')
    embeds = Embeds(embeds_fname, 'fasttext', format=format_embeds)
    if embeds.vec_size != 0:
        embed_dim = embeds.vec_size
    else:
        embed_dim = 100

    # ====Clean texts====
    logger.info('Cleaning text...')
    if warm_start:
        logger.info('Use warm start...')
    else:
        train_df['comment_text_clean'] = clean_text(train_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
        test_df['comment_text_clean'] = clean_text(test_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
        train_df.to_csv(train_clean, index=False)
        test_df.to_csv(test_clean, index=False)

    # ====Calculate maximum seq length====
    logger.info('Calc text length...')
    train_df.fillna('unknown', inplace=True)
    test_df.fillna('unknown', inplace=True)
    train_df['text_len'] = train_df['comment_text_clean'].apply(lambda words: len(words.split()))
    test_df['text_len'] = test_df['comment_text_clean'].apply(lambda words: len(words.split()))
    max_seq_len = np.round(train_df['text_len'].mean() + 3*train_df['text_len'].std()).astype(int)
    logger.debug('Max seq length = {}'.format(max_seq_len))

    # ====Prepare data to NN====
    logger.info('Converting texts to sequences...')
    max_words = 100000

    train_df['comment_seq'], test_df['comment_seq'], word_index = convert_text2seq(train_df['comment_text_clean'].tolist(), test_df['comment_text_clean'].tolist(), max_words, max_seq_len, lower=True, char_level=False, uniq=True)
    logger.debug('Dictionary size = {}'.format(len(word_index)))

    logger.info('Preparing embedding matrix...')
    embedding_matrix, words_not_found = get_embedding_matrix(embed_dim, embeds, max_words, word_index)
    logger.debug('Embedding matrix shape = {}'.format(np.shape(embedding_matrix)))
    logger.debug('Number of null word embeddings = {}'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

    logger.info('Deleting unknown words from seq...')
    train_df['comment_seq'] = clean_seq(train_df['comment_seq'], embedding_matrix, max_seq_len)
    test_df['comment_seq'] = clean_seq(test_df['comment_seq'], embedding_matrix, max_seq_len)

    # ====Train/test split data====
    x = np.array(train_df['comment_seq'].tolist())
    y = np.array(train_df[target_labels].values)
    for l in y:
        if np.sum(l) > 0:
            print(l)
    x_train_nn, x_test_nn, y_train_nn, y_test_nn, train_idxs, test_idxs = split_data(x, y, test_size=0.2, shuffle=True, random_state=42)
    test_df_seq = np.array(test_df['comment_seq'].tolist())
    logger.debug('X shape = {}'.format(np.shape(x_train_nn)))
    #
    # # ====Train models====
    #
    # # Load params to the models
    # params = Params(config)
    #
    # # CNN
    # logger.info("training CNN ...")
    # if params.get('cnn').get('warm_start') and os.path.exists(params.get('cnn').get('model_file')):
    #     logger.info('CNN warm starting...')
    #     cnn = load_model(params.get('cnn').get('model_file'))
    #     cnn_hist = None
    # else:
    #     cnn = get_cnn(embedding_matrix,
    #                 num_classes,
    #                 embed_dim,
    #                 max_seq_len,
    #                 num_filters=params.get('cnn').get('num_filters'),
    #                 l2_weight_decay=params.get('cnn').get('l2_weight_decay'),
    #                 dropout_val=params.get('cnn').get('dropout_val'),
    #                 dense_dim=params.get('cnn').get('dense_dim'),
    #                 add_sigmoid=True)
    #     cnn_hist = train(x_train_nn,
    #                      y_train_nn,
    #                      cnn,
    #                      batch_size=params.get('cnn').get('batch_size'),
    #                      num_epochs=params.get('cnn').get('num_epochs'),
    #                      learning_rate=params.get('cnn').get('learning_rate'),
    #                      early_stopping_delta=params.get('cnn').get('early_stopping_delta'),
    #                      early_stopping_epochs=params.get('cnn').get('early_stopping_epochs'),
    #                      use_lr_stratagy=params.get('cnn').get('use_lr_stratagy'),
    #                      lr_drop_koef=params.get('cnn').get('lr_drop_koef'),
    #                      epochs_to_drop=params.get('cnn').get('epochs_to_drop'),
    #                      logger=logger)
    # y_cnn = cnn.predict(x_test_nn)
    # save_predictions(test_df, cnn.predict(test_df_seq), target_labels, 'cnn')
    # metrics_cnn = get_metrics(y_test_nn, y_cnn, target_labels, hist=cnn_hist, plot=False)
    # logger.debug('CNN metrics:\n{}'.format(print_metrics(metrics_cnn)))
    # cnn.save(cnn_model_file)
    #
    # # LSTM
    # logger.info("training LSTM ...")
    # if params.get('lstm').get('warm_start') and os.path.exists(params.get('lstm').get('model_file')):
    #     logger.info('LSTM warm starting...')
    #     lstm = load_model(params.get('lstm').get('model_file'))
    #     lstm_hist = None
    # else:
    #     lstm = get_lstm(embedding_matrix,
    #                     num_classes,
    #                     embed_dim,
    #                     max_seq_len,
    #                     l2_weight_decay=params.get('lstm').get('l2_weight_decay'),
    #                     lstm_dim=params.get('lstm').get('lstm_dim'),
    #                     dropout_val=params.get('lstm').get('dropout_val'),
    #                     dense_dim=params.get('lstm').get('dense_dim'),
    #                     add_sigmoid=True)
    #     lstm_hist = train(x_train_nn,
    #                       y_train_nn,
    #                       lstm,
    #                       batch_size=params.get('lstm').get('batch_size'),
    #                       num_epochs=params.get('lstm').get('num_epochs'),
    #                       learning_rate=params.get('lstm').get('learning_rate'),
    #                       early_stopping_delta=params.get('lstm').get('early_stopping_delta'),
    #                       early_stopping_epochs=params.get('lstm').get('early_stopping_epochs'),
    #                       use_lr_stratagy=params.get('lstm').get('use_lr_stratagy'),
    #                       lr_drop_koef=params.get('lstm').get('lr_drop_koef'),
    #                       epochs_to_drop=params.get('lstm').get('epochs_to_drop'),
    #                       logger=logger)
    # y_lstm = lstm.predict(x_test_nn)
    # save_predictions(test_df, lstm.predict(test_df_seq), target_labels, 'lstm')
    # metrics_lstm = get_metrics(y_test_nn, y_lstm, target_labels, hist=lstm_hist, plot=False)
    # logger.debug('LSTM metrics:\n{}'.format(print_metrics(metrics_lstm)))
    # lstm.save(lstm_model_file)
    #
    # # CONCAT
    # logger.info("training Concat NN (LSTM + CNN) ...")
    # if params.get('concat').get('warm_start') and os.path.exists(params.get('concat').get('model_file')):
    #     logger.info('Concat NN warm starting...')
    #     concat = load_model(params.get('concat').get('model_file'))
    #     concat_hist = None
    # else:
    #     concat = get_concat_model(embedding_matrix,
    #                               num_classes,
    #                               embed_dim,
    #                               max_seq_len,
    #                               num_filters=params.get('concat').get('num_filters'),
    #                               l2_weight_decay=params.get('concat').get('l2_weight_decay'),
    #                               lstm_dim=params.get('concat').get('lstm_dim'),
    #                               dropout_val=params.get('concat').get('dropout_val'),
    #                               dense_dim=params.get('concat').get('dense_dim'),
    #                               add_sigmoid=True)
    #     concat_hist = train([x_train_nn, x_train_nn],
    #                         y_train_nn,
    #                         concat,
    #                         batch_size=params.get('concat').get('batch_size'),
    #                         num_epochs=params.get('concat').get('num_epochs'),
    #                         learning_rate=params.get('concat').get('learning_rate'),
    #                         early_stopping_delta=params.get('concat').get('early_stopping_delta'),
    #                         early_stopping_epochs=params.get('concat').get('early_stopping_epochs'),
    #                         use_lr_stratagy=params.get('concat').get('use_lr_stratagy'),
    #                         lr_drop_koef=params.get('concat').get('lr_drop_koef'),
    #                         epochs_to_drop=params.get('concat').get('epochs_to_drop'),
    #                         logger=logger)
    # y_concat = concat.predict([x_test_nn, x_test_nn])
    # save_predictions(test_df, concat.predict([test_df_seq, test_df_seq]), target_labels, 'concat')
    # metrics_concat = get_metrics(y_test_nn, y_concat, target_labels, hist=concat_hist, plot=False)
    # logger.debug('Concat_NN metrics:\n{}'.format(print_metrics(metrics_concat)))
    # concat.save(concat_model_file)
    #
    # # TFIDF + LogReg
    # logger.info('training LogReg over tfidf...')
    # train_tfidf, val_tfidf, test_tfidf, word_tfidf, char_tfidf = get_tfidf(train_df['comment_text_clean'].values[train_idxs],
    #                                                                        train_df['comment_text_clean'].values[test_idxs],
    #                                                                        test_df['comment_text_clean'].values)
    #
    # models_lr = []
    # metrics_lr = {}
    # y_tfidf = []
    # for i, label in enumerate(target_labels):
    #     model = LogisticRegression(C=4.0, solver='sag', max_iter=1000, n_jobs=16)
    #     model.fit(train_tfidf, y_train_nn[:, i])
    #     y_tfidf.append(model.predict_proba(val_tfidf)[:,1])
    #     test_df['tfidf_{}'.format(label)] = model.predict_proba(test_tfidf)[:,1]
    #     metrics_lr[label] = calc_metrics(y_test_nn[:, i], y_tfidf[-1])
    #     models_lr.append(model)
    #     joblib.dump(model, lr_model_file.format(label))
    # metrics_lr['Avg logloss'] = np.mean([metric['Logloss'] for label, metric in metrics_lr.items()])
    # logger.debug('LogReg(TFIDF) metrics:\n{}'.format(metrics_lr))
    #
    # # Bow for catboost
    # if params.get('catboost').get('add_bow'):
    #     top_pos_words = []
    #     top_neg_words = []
    #     for i in range(num_classes):
    #         top_pos_words.append([])
    #         top_neg_words.append([])
    #         top_pos_words[-1], top_neg_words[-1] = get_most_informative_features([word_tfidf, char_tfidf], models_lr[i], n=params.get('catboost').get('bow_top'))
    #
    #     top_pos_words = list(set(np.concatenate([[val for score, val in top] for top in top_pos_words])))
    #     top_neg_words = list(set(np.concatenate([[val for score, val in top] for top in top_neg_words])))
    #     top = list(set(np.concatenate([top_pos_words, top_neg_words])))
    #     train_bow = get_bow(train_df['comment_text_clean'].values[train_idxs], top)
    #     val_bow = get_bow(train_df['comment_text_clean'].values[test_idxs], top)
    #     test_bow = get_bow(test_df['comment_text_clean'].values, top)
    #     logger.debug('Count bow words = {}'.format(len(top)))
    #
    # # Meta catboost
    # logger.info('training catboost as metamodel...')
    # train_df['text_unique_len'] = train_df['comment_text_clean'].apply(calc_text_uniq_words)
    # test_df['text_unique_len'] = test_df['comment_text_clean'].apply(calc_text_uniq_words)
    #
    # train_df['text_unique_koef'] = train_df['text_unique_len'] / train_df['text_len']
    # test_df['text_unique_koef'] = test_df['text_unique_len'] / test_df['text_len']
    #
    # text_len_features = train_df[['text_len', 'text_unique_len', 'text_unique_koef']].values[test_idxs]
    #
    # x_train_catboost = []
    # y_train_catboost = y_test_nn
    # features = [text_len_features, y_cnn, y_lstm, y_concat, np.array(y_tfidf).T]
    # if params.get('catboost').get('add_bow'):
    #     features.append(val_bow)
    # for feature in zip(*features):
    #     x_train_catboost.append(np.concatenate(feature))
    #
    # models_cb = []
    # metrics_cb = {}
    # x_train_cb, x_val_cb, y_train_cb, y_val_cb = train_test_split(x_train_catboost, y_train_catboost, test_size=0.20, random_state=42)
    # for i, label in enumerate(target_labels):
    #     model = CatBoostClassifier(loss_function='Logloss',
    #                                iterations=params.get('catboost').get('iterations'),
    #                                depth=params.get('catboost').get('depth'),
    #                                rsm=params.get('catboost').get('rsm'),
    #                                learning_rate=params.get('catboost').get('learning_rate'),
    #                                device_config=params.get('catboost').get('device_config'))
    #     model.fit(x_train_cb, y_train_cb[:, i], plot=True, eval_set=(x_val_cb, y_val_cb[:, i]), use_best_model=True)
    #     y_hat_cb = model.predict_proba(x_val_cb)
    #     metrics_cb[label] = calc_metrics(y_val_cb[:, i], y_hat_cb[:, 1])
    #     models_cb.append(model)
    #     joblib.dump(model, meta_catboost_model_file.format(label))
    # metrics_cb['Avg logloss'] = np.mean([metric['Logloss'] for label,metric in metrics_cb.items()])
    # logger.debug('CatBoost metrics:\n{}'.format(metrics_cb))
    #
    # # ====Predict====
    # logger.info('Applying models...')
    # text_len_features = test_df[['text_len', 'text_unique_len', 'text_unique_koef']].values
    # y_cnn_test = test_df[['cnn_{}'.format(label) for label in target_labels]].values
    # y_lstm_test = test_df[['lstm_{}'.format(label) for label in target_labels]].values
    # y_concat_test = test_df[['concat_{}'.format(label) for label in target_labels]].values
    # y_tfidf_test = test_df[['tfidf_{}'.format(label) for label in target_labels]].values
    # x_test_cb = []
    # features = [text_len_features, y_cnn_test, y_lstm_test, y_concat_test, y_tfidf_test]
    # if params.get('catboost').get('add_bow'):
    #     features.append(test_bow)
    # for feature in tqdm(zip(*features)):
    #     x_test_cb.append(np.concatenate(feature))
    #
    # for label, model in zip(target_labels, models_cb):
    #     pred = model.predict_proba(x_test_cb)
    #     test_df[label] = np.array(list(pred))[:, 1]
    #
    # # ====Save results====
    # logger.info('Saving results...')
    # test_df[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(result_fname, index=False, header=True)


if __name__=='__main__':
    main()
