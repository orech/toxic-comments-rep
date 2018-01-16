import re
import random
import numpy as np
from tqdm import tqdm

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def calc_text_uniq_words(text):
    unique_words = set()
    for word in text.split():
        unique_words.add(word)
    return len(unique_words)


# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46371
def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)

def substitute_repeats(text, ntimes=3):
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    return text


# Split word and digits
def split_text_and_digits(text, regexps):
    for regexp in regexps:
        result = regexp.match(text)
        if result is not None:
            return ' '.join(result.groups())
    return text


def read_wrong_words(fname):
    wrong_word_dict = {}
    with open(fname) as f:
        for line in f:
            line = line.rstrip()
            line = re.sub(' +', ' ', line)
            line = line.split()
            if len(line) < 2:
                continue
            wrong_word_dict[line[0]] = ' '.join(line[1:])
    return wrong_word_dict


def combine_swear_words(text, swear_words):
    i = 0
    n = len(text)
    result = []
    while i < n - 1:
        word = text[i]
        next_word = text[i+1]
        if len(word) == 1 or len(next_word) == 1:
            if not (word.isdigit() or next_word.isdigit()):
                combine_word = '{}{}'.format(word, next_word)
                if combine_word in swear_words:
                    i += 1
                    word = combine_word
        result.append(word)
        i += 1
    return result


def clean_text(df, tokinizer, wrong_words_dict, swear_words, regexps, autocorrect=True, swear_combine=True):
    df.fillna("__NA__", inplace=True)
    texts = df.tolist()
    result = []
    for text in tqdm(texts):
        tokens = tokinizer.tokenize(text.lower())
        tokens = [split_text_and_digits(token, regexps) for token in tokens]
        tokens = [substitute_repeats(token, 3) for token in tokens]
        if swear_combine:
            tokens = combine_swear_words(tokens, swear_words)
        text = ' '.join(tokens)
        if autocorrect:
            for wrong, right in wrong_words_dict.items():
                text = text.replace(wrong, right)
        result.append(text)
    return result


def uniq_words_in_text(text):
    return ' '.join(list(set(text.split())))


def convert_text2seq(train_texts, test_texts, max_words, max_seq_len, lower=True, char_level=False, uniq=False):
    tokenizer = Tokenizer(num_words=max_words, lower=lower, char_level=char_level)
    texts = train_texts + test_texts
    if uniq:
        texts = [uniq_words_in_text(text) for text in texts]
    tokenizer.fit_on_texts(texts)
    word_seq_train = tokenizer.texts_to_sequences(train_texts)
    word_seq_test = tokenizer.texts_to_sequences(test_texts)
    word_index = tokenizer.word_index
    word_seq_train = list(sequence.pad_sequences(word_seq_train, maxlen=max_seq_len))
    word_seq_test = list(sequence.pad_sequences(word_seq_test, maxlen=max_seq_len))
    return word_seq_train, word_seq_test, word_index


def delete_unknown_words(seq, embedding_sums, max_len):
    seq = [idx for idx in seq if embedding_sums[idx] != 0]
    seq = list(np.zeros(max_len - len(seq))) + seq
    return np.array(seq)


def clean_seq(seqs, embedding_matrix, max_len):
    embedding_sums = np.sum(embedding_matrix, axis=1)
    seqs = [delete_unknown_words(seq, embedding_sums, max_len) for seq in seqs]
    return seqs


def get_embedding_matrix(embed_dim, embeds, max_words, word_index):
    words_not_found = []
    nb_words = min(max_words, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeds[word]
        if embedding_vector is not None and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    return embedding_matrix, words_not_found


def split_data_idx(n, test_size=0.2, shuffle=True, random_state=0):
    train_size = 1 - test_size
    idxs = np.arange(n)
    if shuffle:
        random.seed(random_state)
        random.shuffle(idxs)
    return idxs[:int(train_size*n)], idxs[int(train_size*n):]


def split_data(x, y, test_size=0.2, shuffle=True, random_state=0):
    n = len(x)
    train_idxs, test_idxs = split_data_idx(n, test_size, shuffle, random_state)
    return np.array(x[train_idxs]), np.array(x[test_idxs]), y[train_idxs], y[test_idxs], train_idxs, test_idxs


def get_bow(texts, words):
    result = np.zeros((len(texts), len(words)))
    print(np.shape(result))
    for i, text in tqdm(enumerate(texts)):
        for j, word in enumerate(words):
            try:
                if word in text:
                    result[i][j] = 1
            except UnicodeDecodeError:
                pass
    return result
