from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Merge
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D


def get_cnn(embedding_matrix, num_classes, embed_dim, max_seq_len, num_filters=64, l2_weight_decay=0.0001, dropout_val=0.5, dense_dim=32, add_sigmoid=True):
    model = Sequential()
    model.add(Embedding(len(embedding_matrix), embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_val))
    model.add(Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay)))
    if add_sigmoid:
        model.add(Dense(num_classes, activation='sigmoid'))
    return model


def get_lstm(embedding_matrix, num_classes, embed_dim, max_seq_len, l2_weight_decay=0.0001, lstm_dim=50, dropout_val=0.3, dense_dim=32, add_sigmoid=True):
    model = Sequential()
    model.add(Embedding(len(embedding_matrix), embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Bidirectional(LSTM(lstm_dim, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_val))
    model.add(Dense(lstm_dim, activation="relu"))
    model.add(Dropout(dropout_val))
    model.add(Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay)))
    if add_sigmoid:
        model.add(Dense(num_classes, activation="sigmoid"))
    return model



def get_concat_model(embedding_matrix, num_classes, embed_dim, max_seq_len, num_filters=64, l2_weight_decay=0.0001, lstm_dim=50, dropout_val=0.5, dense_dim=32, add_sigmoid=True):
    model_lstm = get_lstm(embedding_matrix, num_classes, embed_dim, max_seq_len, l2_weight_decay, lstm_dim, dropout_val, dense_dim, add_sigmoid=False)
    model_cnn = get_cnn(embedding_matrix, num_classes, embed_dim, max_seq_len, num_filters, l2_weight_decay, dropout_val, dense_dim, add_sigmoid=False)
    model = Sequential()
    model.add(Merge([model_lstm, model_cnn], mode='concat'))
    model.add(Dropout(dropout_val))
    model.add(Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay)))
    if add_sigmoid:
        model.add(Dense(num_classes, activation="sigmoid"))
    return model


def get_tfidf(x_train, x_val, x_test, max_features=50000):
    word_tfidf = TfidfVectorizer(max_features=max_features, analyzer='word', lowercase=True, ngram_range=(1, 3), token_pattern='[a-zA-Z0-9]')
    char_tfidf = TfidfVectorizer(max_features=max_features, analyzer='char', lowercase=True, ngram_range=(1, 5), token_pattern='[a-zA-Z0-9]')

    train_tfidf_word = word_tfidf.fit_transform(x_train)
    val_tfidf_word = word_tfidf.transform(x_val)
    test_tfidf_word = word_tfidf.transform(x_test)

    train_tfidf_char = char_tfidf.fit_transform(x_train)
    val_tfidf_char = char_tfidf.transform(x_val)
    test_tfidf_char = char_tfidf.transform(x_test)

    train_tfidf = sparse.hstack([train_tfidf_word, train_tfidf_char])
    val_tfidf = sparse.hstack([val_tfidf_word, val_tfidf_char])
    test_tfidf = sparse.hstack([test_tfidf_word, test_tfidf_char])

    return train_tfidf, val_tfidf, test_tfidf, word_tfidf, char_tfidf


def get_most_informative_features(vectorizers, clf, n=20):
    feature_names = []
    for vectorizer in vectorizers:
        feature_names.extend(vectorizer.get_feature_names())
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    return coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1]


def save_predictions(df, predictions, target_labels, additional_name=None):
    for i, label in enumerate(target_labels):
        if additional_name is not None:
            label = '{}_{}'.format(additional_name, label)
        df[label] = predictions[:, i]

# From https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1, solver='sag', max_iter=100):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.solver = solver
        self.max_iter = max_iter

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, max_iter=self.max_iter, solver=self.solver, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
