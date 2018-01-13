import re
import argparse
import logging
from six import iteritems
import numpy as np

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

from utils import load_data, read_swear_words, read_wrong_words, Embeds, Logger
from prepare_data import calc_text_uniq_words, clean_text, convert_text2seq, get_embedding_matrix, split_data, get_bow
from models import get_cnn, get_lstm, get_concat_model, save_predictions, get_tfidf, get_most_informative_features
from train import train, continue_train
from metrics import calc_metrics, get_metrics


def get_kwargs(kwargs):
    parser = argparse.ArgumentParser(description='-f TRAIN_FILE -t TEST_FILE -o OUTPUT_FILE -e EMBEDS_FILE [-l LOGGER_FILE] [--swear-words SWEAR_FILE] [--wrong-words WRONG_WORDS_FILE] [--warm-start FALSE] [--pickle-embeds FALSE]')
    parser.add_argument('-f', '--train', dest='train', action='store', help='/path/to/trian_file', type=str)
    parser.add_argument('-t', '--test', dest='test', action='store', help='/path/to/test_file', type=str)
    parser.add_argument('-o', '--output', dest='output', action='store', help='/path/to/output_file', type=str)
    parser.add_argument('-e', '--embeds', dest='embeds', action='store', help='/path/to/embeds_file', type=str)
    parser.add_argument('-l', '--logger', dest='logger', action='store', help='/path/to/log_file', type=str, default=None)
    parser.add_argument('--swear-words', dest='swear_words', action='store', help='/path/to/swear_words_file', type=str, default=None)
    parser.add_argument('--wrong-words', dest='wrong_words', action='store', help='/path/to/wrong_words_file', type=str, default=None)
    parser.add_argument('--warm-start', dest='warm_start', action='store', help='true | false', type=bool, default=False)
    parser.add_argument('--format-embeds', dest='format_embeds', action='store', help='file | json | pickle | binary', type=str, default='file')
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
    format_embeds = kwargs['format_embeds']

    cnn_model_file = 'data/cnn.h5'
    lstm_model_file = 'data/lstm.h5'
    concat_model_file = 'data/concat.h5'
    cnn_model_file = 'data/cnn.h5'
    lr_model_file = 'data/{}_logreg.bin'
    meta_catboost_model_file = 'data/{}_meta_catboost.bin'

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
    swear_words = read_swear_words(swear_words_fname)
    wrong_words_dict = read_wrong_words(wrong_words_fname)

    tokinizer = RegexpTokenizer(r'\w+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]

    # ====Load word vectors====
    logger.info('Loading embeddings...')
    embed_dim = 300
    embeds = Embeds(embeds_fname, 'fasttext', format=format_embeds)

    # ====Clean texts====
    logger.info('Cleaning text...')
    if warm_start:
        logger.info('Use warm start...')
    else:
        train_df['comment_text_clear'] = clean_text(train_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
        test_df['comment_text_clear'] = clean_text(test_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
        train_df.to_csv(train_clear, index=False)
        test_df.to_csv(test_clear, index=False)

    # ====Calculate maximum seq length====
    logger.info('Calc text length...')
    train_df.fillna('unknown', inplace=True)
    test_df.fillna('unknown', inplace=True)
    train_df['text_len'] = train_df['comment_text_clear'].apply(lambda words: len(words.split()))
    test_df['text_len'] = test_df['comment_text_clear'].apply(lambda words: len(words.split()))
    max_seq_len = np.round(train_df['text_len'].mean() + 3*train_df['text_len'].std()).astype(int)
    logger.debug('Max seq length = {}'.format(max_seq_len))

    # ====Prepare data to NN====
    logger.info('Converting texts to sequences...')
    max_words = 100000

    train_df['comment_seq'], test_df['comment_seq'], word_index = convert_text2seq(train_df['comment_text_clear'].tolist(), test_df['comment_text_clear'].tolist(), max_words, max_seq_len, lower=True, char_level=False)
    logger.debug('Dictionary size = {}'.format(len(word_index)))
    logger.info('Preparing embedding matrix...')
    embedding_matrix, words_not_found = get_embedding_matrix(embed_dim, embeds, max_words, word_index)
    logger.debug('Embedding matrix shape = {}'.format(np.shape(embedding_matrix)))
    logger.debug('Number of null word embeddings = {}'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

    # ====Train/test split data====
    x = np.array(train_df['comment_seq'].tolist())
    y = np.array(train_df[target_labels].values)
    x_train_nn, x_test_nn, y_train_nn, y_test_nn, train_idxs, test_idxs = split_data(x, y, test_size=0.2, shuffle=True, random_state=42)
    test_df_seq = np.array(test_df['comment_seq'].tolist())

    # ====Train models====

    # CNN
    logger.info("training CNN ...")
    cnn = get_cnn(embedding_matrix, num_classes, embed_dim, max_seq_len, num_filters=64, l2_weight_decay=0.0001, dropout_val=0.5, dense_dim=32, add_sigmoid=True)
    cnn_hist = train(x_train_nn, y_train_nn, cnn, batch_size=256, num_epochs=100, learning_rate=0.005, early_stopping_delta=0.0001, early_stopping_epochs=3, use_lr_stratagy=True, lr_drop_koef=0.66, epochs_to_drop=2, logger=logger)
    y_cnn = cnn.predict(x_test_nn)
    save_predictions(test_df, cnn.predict(test_df_seq), target_labels, 'cnn')
    metrics_cnn = get_metrics(y_test_nn, y_cnn, target_labels, hist=cnn_hist, plot=False)
    logger.debug('CNN metrics:\n{}'.format(metrics_cnn))
    cnn.save(cnn_model_file)

    # LSTM
    logger.info("training LSTM ...")
    lstm = get_lstm(embedding_matrix, num_classes, embed_dim, max_seq_len, l2_weight_decay=0.0001, lstm_dim=50, dropout_val=0.3, dense_dim=32, add_sigmoid=True)
    lstm_hist = train(x_train_nn, y_train_nn, lstm, batch_size=256, num_epochs=100, learning_rate=0.005, early_stopping_delta=0.0001, early_stopping_epochs=3, use_lr_stratagy=True, lr_drop_koef=0.66, epochs_to_drop=2, logger=logger)
    y_lstm = lstm.predict(x_test_nn)
    save_predictions(test_df, lstm.predict(test_df_seq), target_labels, 'lstm')
    metrics_lstm = get_metrics(y_test_nn, y_lstm, target_labels, hist=lstm_hist, plot=False)
    logger.debug('LSTM metrics:\n{}'.format(metrics_lstm))
    lstm.save(lstm_model_file)

    # CONCAT
    logger.info("training Concat NN (LSTM + CNN) ...")
    concat = get_concat_model(embedding_matrix, num_classes, embed_dim, max_seq_len, num_filters=64, l2_weight_decay=0.0001, lstm_dim=50, dropout_val=0.5, dense_dim=32, add_sigmoid=True)
    concat_hist = train([x_train_nn, x_train_nn], y_train_nn, concat, batch_size=256, num_epochs=100, learning_rate=0.005, early_stopping_delta=0.0001, early_stopping_epochs=4, use_lr_stratagy=True, lr_drop_koef=0.66, epochs_to_drop=3, logger=logger)
    y_concat = concat.predict([x_test_nn, x_test_nn])
    save_predictions(test_df, concat.predict([test_df_seq, test_df_seq]), target_labels, 'concat')
    metrics_concat = get_metrics(y_test_nn, y_concat, target_labels, hist=concat_hist, plot=False)
    logger.debug('Concat_NN metrics:\n{}'.format(metrics_concat))
    concat.save(concat_model_file)

    # TFIDF + LogReg
    logger.info('training LogReg over tfidf...')
    train_tfidf, val_tfidf, test_tfidf, word_tfidf, char_tfidf = get_tfidf(train_df['comment_text_clear'].values[train_idxs],
                                                    train_df['comment_text_clear'].values[test_idxs],
                                                    test_df['comment_text_clear'].values)

    models_lr = []
    metrics_lr = {}
    y_tfidf = []
    for i, label in enumerate(target_labels):
        model = NbSvmClassifier(C=4.0, solver='sag', max_iter=1000)
        model.fit(train_tfidf, y_train_nn[:, i])
        y_tfidf.append(model.predict_proba(val_tfidf)[:,1])
        test_df['tfidf_{}'.format(label)] = model.predict_proba(test_tfidf)[:,1]
        metrics_lr[label] = calc_metrics(y_test_nn[:, i], y_tfidf[-1])
        models_lr.append(model)
        joblib.dump(model, lr_model_file.format(label))
    metrics_lr['Avg logloss'] = np.mean([metric[0] for label, metric in metrics_lr.items()])
    logger.debug('LogReg(TFIDF) metrics:\n{}'.format(metrics_lr))

    # Bow for catboost
    top_pos_words = []
    top_neg_words = []
    for i in range(num_classes):
        top_pos_words.append([])
        top_neg_words.append([])
        top_pos_words[-1], top_neg_words[-1] = get_most_informative_features([word_tfidf, char_tfidf], models_lr[i], n=100)

    top_pos_words = list(set(np.concatenate([[val for score, val in top] for top in top_pos_words])))
    top_neg_words = list(set(np.concatenate([[val for score, val in top] for top in top_neg_words])))
    top = list(set(np.concatenate([top_pos_words, top_neg_words])))
    train_bow = get_bow(train_df['comment_text_clear'].values[train_idxs], top)
    val_bow = get_bow(train_df['comment_text_clear'].values[test_idxs], top)
    test_bow = get_bow(test_df['comment_text_clear'].values, top)
    logger.debug('Count bow words = {}'.format(len(top)))

    # Meta catboost
    logger.info('training catboost as metamodel...')
    train_df['text_unique_len'] = train_df['comment_text_clear'].apply(calc_text_uniq_words)
    test_df['text_unique_len'] = test_df['comment_text_clear'].apply(calc_text_uniq_words)

    train_df['text_unique_koef'] = train_df['text_unique_len'] / train_df['text_len']
    test_df['text_unique_koef'] = test_df['text_unique_len'] / test_df['text_len']

    text_len_features = train_df[['text_len', 'text_unique_len', 'text_unique_koef']].values[test_idxs]

    x_train_catboost = []
    y_train_catboost = y_test_nn
    for len_f, y_hat_cnn, y_hat_lstm, y_hat_concat, y_hat_tfidf, bow in zip(text_len_features, y_cnn, y_lstm, y_concat, np.array(y_tfidf).T, val_bow):
        x_train_catboost.append(np.concatenate([len_f, y_hat_cnn, y_hat_lstm, y_hat_concat, y_hat_tfidf, bow]))

    models_cb = []
    metrics_cb = {}
    x_train_cb, x_val_cb, y_train_cb, y_val_cb = train_test_split(x_train_catboost, y_train_catboost, test_size=0.20, random_state=42)
    for i, label in enumerate(target_labels):
        model = CatBoostClassifier(loss_function='Logloss', iterations=1000, depth=6, rsm=1, learning_rate=0.01)
        model.fit(x_train_cb, y_train_cb[:, i], plot=True, eval_set=(x_val_cb, y_val_cb[:, i]), use_best_model=True)
        y_hat_cb = model.predict_proba(x_val_cb)
        metrics_cb[label] = calc_metrics(y_val_cb[:, i], y_hat_cb[:, 1])
        models_cb.append(model)
        joblib.dump(model, meta_catboost_model_file.format(label))
    metrics_cb['Avg logloss'] = np.mean([metric[0] for label,metric in metrics_cb.items()])
    logger.debug('CatBoost metrics:\n{}'.format(metrics_cb))

    # ====Predict====
    logger.info('Applying models...')
    text_len_features = test_df[['text_len', 'text_unique_len', 'text_unique_koef']].values
    y_cnn_test = test_df[['cnn_{}'.format(label) for label in target_labels]].values
    y_lstm_test = test_df[['lstm_{}'.format(label) for label in target_labels]].values
    y_concat_test = test_df[['concat_{}'.format(label) for label in target_labels]].values
    y_tfidf_test = test_df[['tfidf_{}'.format(label) for label in target_labels]].values
    x_test_cb = []
    for len_f, y_hat_cnn, y_hat_lstm, y_hat_concat, y_hat_tfidf, bow in tqdm(zip(text_len_features, y_cnn_test, y_lstm_test, y_concat_test, y_tfidf_test, test_bow)):
        x_test_cb.append(np.concatenate([len_f, y_hat_cnn, y_hat_lstm, y_hat_concat, y_hat_tfidf, bow]))

    for label, model in zip(target_labels, models_cb):
        pred = model.predict_proba(x_test_cb)
        test_df[label] = np.array(list(pred))[:, 1]

    # ====Save results====
    logger.info('Saving results...')
    test_df[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(result_fname, index=False, header=True)


if __name__=='__main__':
    main()
