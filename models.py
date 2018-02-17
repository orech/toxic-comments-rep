from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Merge, Conv2D, MaxPooling2D, BatchNormalization, Lambda
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input, GlobalMaxPooling2D, Concatenate
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Merge, Conv2D, MaxPooling2D, BatchNormalization, Lambda, GlobalAveragePooling1D, Concatenate
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input, GlobalMaxPooling2D

from keras.layers import Bidirectional, Dropout, CuDNNGRU, CuDNNLSTM, Reshape
from keras.models import Model
from keras.backend import cast

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())

        This implementation is taken from the kernel : https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043/code
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


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


def get_2BiGRU(embedding_matrix, num_classes, sequence_length, recurrent_units, dense_size, dropout_rate=0.5):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu", kernel_initializer='glorot_uniform')(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_BiGRU_Dense(embedding_matrix, num_classes, sequence_length, recurrent_units, dense_sizes, dropout_rate=0.3):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_sizes[0], activation="relu", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_sizes[1], activation="relu", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_BiGRU_Attention(embedding_matrix, num_classes, sequence_length, recurrent_units, dense_size, dropout_rate=0.5):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Attention(sequence_length)(x)
    x = Dense(dense_size, activation="relu", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_2BiGRU_BN(embedding_matrix, num_classes, sequence_length, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = BatchNormalization()(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dense(dense_size, activation="relu", kernel_initializer='glorot_uniform')(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_2BiGRU_GlobMaxPool(embedding_matrix, num_classes, sequence_length, recurrent_units, dense_size, dropout_rate=0.5):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = Dropout(dropout_rate)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(dense_size, activation="relu", kernel_initializer='glorot_uniform')(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_BiGRU_Max_Avg_Pool(embedding_matrix, num_classes, sequence_length, recurrent_units, dense_size, dropout_rate=0.5):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x_1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x_2 = GlobalMaxPooling1D()(x_1)
    x_3 = GlobalAveragePooling1D()(x_1)
    x = Concatenate()([x_1, x_2, x_3])
    x = Dense(dense_size, activation="relu", kernel_initializer='glorot_uniform')(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_BiGRU_2dConv_2dMaxPool(embedding_matrix, num_classes, sequence_length):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(input_layer)
    drop_embedding = Dropout(0.3)(embedding_layer)
    x = Bidirectional(CuDNNLSTM(300, return_sequences=True))(drop_embedding)
    x = Dropout(0.2)(x)
    x = Reshape((500, 600, 1))(x)
    x = Conv2D(filters=100, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', data_format='channels_last')(x)
    x = GlobalMaxPooling2D()(x)
    output_layer = Dense(num_classes, activation="sigmoid")(x)
    drop_output = Dropout(0.3)(output_layer)
    model = Model(inputs=input_layer, outputs=drop_output)
    return model

def get_pyramidCNN(embedding_matrix, num_classes, sequence_length, dropout_rate, num_of_filters, filter_size, num_of_blocks, dense_size=128, l2_weight_decay=0.0001):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)

    region_embedding = Conv1D(num_of_filters, filter_size)(embedding_layer)


    pre_activation_conv0_1 = Lambda(lambda x: K.relu(x))(region_embedding)
    drop0_1 = Dropout(dropout_rate)(pre_activation_conv0_1)
    conv0_1 = Conv1D(num_of_filters, filter_size, padding='same')(drop0_1)

    pre_activation_conv0_2 = Lambda(lambda x: K.relu(x))(conv0_1)
    drop0_2 = Dropout(dropout_rate)(pre_activation_conv0_2)
    conv0_2 = Conv1D(num_of_filters, filter_size, padding='same')(drop0_2)

    shortcut0 = Lambda(lambda x: x[0] + x[1])([conv0_2, region_embedding])
    res = shortcut0

    for i in range(num_of_blocks):
        pooled = MaxPooling1D(pool_size=3, strides=2)(res)

        pre_activation_conv1 = Lambda(lambda x: K.relu(x))(pooled)
        drop1 = Dropout(dropout_rate)(pre_activation_conv1)
        conv1 = Conv1D(num_of_filters, filter_size, padding='same')(drop1)

        pre_activation_conv2 = Lambda(lambda x: K.relu(x))(conv1)
        drop2 = Dropout(dropout_rate)(pre_activation_conv2)
        conv2 = Conv1D(num_of_filters, filter_size, padding='same')(drop2)

        shortcut = Lambda(lambda x: x[0] + x[1])([conv2, pooled])

        res = shortcut

    globalPooled = GlobalMaxPooling1D()(res)
    drop = Dropout(dropout_rate)(globalPooled)
    #dense = Dense(dense_size,activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(drop)
    output_layer = Dense(num_classes, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_weight_decay))(drop)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get__original_pyramidCNN(embedding_matrix, num_classes, sequence_length, dropout_rate, num_of_filters, filter_size, num_of_blocks, l2_weight_decay=0.0001):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)

    region_embedding = Conv1D(num_of_filters, filter_size)(embedding_layer)


    pre_activation_conv0_1 = Lambda(lambda x: K.relu(x))(region_embedding)
    conv0_1 = Conv1D(num_of_filters, filter_size, padding='same')(pre_activation_conv0_1)

    pre_activation_conv0_2 = Lambda(lambda x: K.relu(x))(conv0_1)
    conv0_2 = Conv1D(num_of_filters, filter_size, padding='same')(pre_activation_conv0_2)

    shortcut0 = Lambda(lambda x: x[0] + x[1])([conv0_2, region_embedding])
    res = shortcut0

    for i in range(num_of_blocks):
        pooled = MaxPooling1D(pool_size=3, strides=2)(res)

        pre_activation_conv1 = Lambda(lambda x: K.relu(x))(pooled)
        conv1 = Conv1D(num_of_filters, filter_size, padding='same')(pre_activation_conv1)

        pre_activation_conv2 = Lambda(lambda x: K.relu(x))(conv1)
        conv2 = Conv1D(num_of_filters, filter_size, padding='same')(pre_activation_conv2)

        shortcut = Lambda(lambda x: x[0] + x[1])([conv2, pooled])

        res = shortcut

    globalPooled = GlobalMaxPooling1D()(res)
    drop = Dropout(dropout_rate)(globalPooled)
    output_layer = Dense(num_classes, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_weight_decay))(drop)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model






def get_simpleCNN(embedding_matrix, num_classes, sequence_length, dropout_rate, num_of_filters, filter_sizes):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)

    pooled_outputs = []
    for i,filter_size in enumerate(filter_sizes):
        conv = Conv1D(num_of_filters, filter_size, activation='relu')(embedding_layer)
        pooled = GlobalMaxPooling1D()(conv)
        pooled_outputs.append(pooled)
    concat = Concatenate(1)(pooled_outputs)
    drop = Dropout(dropout_rate)(concat)
    output_layer = Dense(num_classes, activation="sigmoid")(drop)
    model = Model(inputs=input_layer, outputs=output_layer)
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
