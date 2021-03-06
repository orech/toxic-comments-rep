import re
import os.path
import argparse
import logging
from six import iteritems
import numpy as np
from alphabet_detector import AlphabetDetector

from embed_utils import load_data, Embeds, Logger, read_embedding_list, clear_embedding_list
from data_utils import calc_text_uniq_words, clean_texts, convert_text2seq, get_embedding_matrix, clean_seq, split_data, get_bow, tokenize_sentences, convert_tokens_to_ids, tokenize_sentences_adv

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
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
    parser.add_argument('--config', dest='config', action='store', help='/path/to/config.BiGRU_Dense.json', type=str, default=None)
    parser.add_argument('--train-clean', dest='train_clean', action='store', help='/path/to/save_train_clean_file', type=str, default='data/train_clean.npy')
    parser.add_argument('--test-clean', dest='test_clean', action='store', help='/path/to/save_test_clean_file', type=str, default='data/results/test_clean.npy')
    parser.add_argument('--embeds-clean', dest='embeds_clean', action='store', help='/path/to/save_embeds_clean_file', type=str, default='data/results/embeds_clean.npy')
    parser.add_argument('--oov-embeds', dest='oov_embeds', action='store', help='/path/to/oov_embeds_file', type=str, default='')
    parser.add_argument('--embeds-type', dest='embeds_type', action='store', type=str, default='ft_comm_crawl')
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
    embeds_clean = kwargs['embeds_clean']
    embeds_fname = kwargs['embeds']
    embeds_type = kwargs['embeds_type']
    oov_embeds_file = kwargs['oov_embeds']
    train_labels = 'data/train.labels.npy'


    # ==== Create logger ====
    logger = Logger(logging.getLogger(), logger_fname)

    # ==== Load data ====
    logger.info('Loading data...')
    train_df = load_data(train_fname)
    test_df = load_data(test_fname)

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # ==== Tokenize comment texts ====
    logger.info('Replacing nans and tokenizing texts...')
    list_sentences_train = train_df['comment_text'].fillna(NAN_WORD).values
    list_sentences_test = test_df['comment_text'].fillna(NAN_WORD).values

    # train_tokens, word_dict = tokenize_sentences(list_sentences_train, {})
    # test_tokens, word_dict = tokenize_sentences(list_sentences_test, word_dict)

    train_tokens, word_dict = tokenize_sentences_adv(list_sentences_train, {})
    test_tokens, word_dict = tokenize_sentences_adv(list_sentences_test, word_dict)

    word_dict[UNKNOWN_WORD] = len(word_dict)


    # # ==== Load additional data ====
    # logger.info('Loading additional data...')
    # swear_words = load_data(swear_words_fname, func=lambda x: set(x.T[0]), header=None)
    # wrong_words_dict = load_data(wrong_words_fname, func=lambda x: {val[0] : val[1] for val in x})


    # ==== Load embedding vectors and clean them ====
    logger.info('Loading embeddings...')
    embedding_list, embedding_word_dict = read_embedding_list(embeds_fname)
    embedding_size = len(embedding_list[0])

    if oov_embeds_file != '':
        logger.info('Loading embeddings for oov words...')
        embedding_list, embedding_word_dict = read_embedding_list(oov_embeds_file, embedding_word_dict, embedding_list)
        embedding_size = len(embedding_list[0])

    logger.info('Cleaning embedding list...')
    embedding_list, embedding_word_dict, oov_words = clear_embedding_list(embedding_list, embedding_word_dict, word_dict)

    # ======== Clean oov words and save them =========
    oov_cleaned = []
    ad = AlphabetDetector()
    with open('data/oov_words_{0}.txt'.format(embeds_type), 'wt+') as oov_file:
        for w in oov_words:
            if ad.only_alphabet_chars(w, "LATIN") and re.match(r'^[A-Za-z]+$', w) and (len(w) <= 15):
                oov_cleaned.append(w)
                oov_file.write(w+'\n')
    oov_file.close()


    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append(np.asarray([0.] * embedding_size))
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append(np.asarray([-1.] * embedding_size))

    embedding_matrix = np.array(embedding_list)

    # ==== Convert word tokens into sequences of word ids  ====
    logger.info('Converting tokens to word ids...')
    id_to_word = dict((id, word) for word, id in word_dict.items())
    train_token_ids = convert_tokens_to_ids(tokenized_sentences=train_tokens,
                                                    words_list=id_to_word,
                                                    embedding_word_dict=embedding_word_dict,
                                                    sentences_length=500)

    test_token_ids = convert_tokens_to_ids(tokenized_sentences=test_tokens,
                                                    words_list=id_to_word,
                                                    embedding_word_dict=embedding_word_dict,
                                                    sentences_length=500)

    # ==== Prepare train/test data for NN ====
    x = np.array(train_token_ids)
    y = np.array(train_df[target_labels].values)
    x_test = np.array(test_token_ids)

    # ==== Saving the results ====
    logger.info("Saving results...")
    np.save(train_clean, x)
    np.save(train_labels, y)
    np.save(test_clean, x_test)
    np.save(embeds_clean, embedding_matrix)


if __name__=='__main__':
    main()
