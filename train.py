import json
from math import pow, floor

from keras import optimizers, callbacks, backend, losses
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from models import get_2BiGRU, get_2BiGRU_BN, get_2BiGRU_GlobMaxPool, get_BiGRU_2dConv_2dMaxPool, get_cnn, get_lstm, get_concat_model, get_tfidf, get_BiGRU_Attention, get_BiLSTM_Attention, get_BiSRU_Attention, get_BiGRU_Dense, get_BiGRU_Max_Avg_Pool_concat, get_2BiGRU_rec_dropout, get_pyramidCNN, get_simpleCNN, get__original_pyramidCNN, get_pyramid_gated_CNN, get_2BiSRU_rec_dropout_GlobMaxPool, get_simpleCNN_conv2d, get_diSAN, get_2BiLSTM_rec_dropout, get_2BiSRU_spat_dropout, get_2BiGRU_spat_dropout, get_2BiLSTM_spat_dropout, get_capsuleNet
import numpy as np


def step_decay(initial_lr, lr_drop_koef, epochs_to_drop, epoch):
    return initial_lr * pow(lr_drop_koef, floor((1 + epoch) / epochs_to_drop))

def get_model(model_name, embedding_matrix, params):
    print(model_name)
    if model_name == '2BiGRU':
        get_model_func = lambda: get_2BiGRU(embedding_matrix=embedding_matrix,
                                        num_classes=6,
                                        sequence_length=params.get(model_name).get('sequence_length'),
                                        dense_size=params.get(model_name).get('dense_dim'),
                                        recurrent_units=params.get(model_name).get('recurrent_units'))
    elif model_name == '2BiGRU_2dConv_2dMaxPool':
        get_model_func = lambda: get_BiGRU_2dConv_2dMaxPool(embedding_matrix=embedding_matrix,
                                                        num_classes=6,
                                                        sequence_length=params.get(model_name).get('sequence_length'))

    elif model_name == '2BiGRU_BN':
        get_model_func = lambda: get_2BiGRU_BN(embedding_matrix=embedding_matrix,
                                           num_classes=6,
                                           sequence_length=params.get(model_name).get('sequence_length'),
                                           recurrent_units=params.get(model_name).get('recurrent_units'),
                                           dense_size=params.get(model_name).get('dense_dim'))
    elif model_name == '2BiGRU_GlobMaxPool':
        get_model_func = lambda: get_2BiGRU_GlobMaxPool(embedding_matrix=embedding_matrix,
                                                    num_classes=6,
                                                    sequence_length=params.get(model_name).get('sequence_length'),
                                                    recurrent_units=params.get(model_name).get('recurrent_units'),
                                                    dense_size=params.get(model_name).get('dense_dim'),
                                                    dropout_rate=params.get(model_name).get('dropout'))
    elif model_name == 'BiGRU_attention':
        get_model_func = lambda: get_BiGRU_Attention(embedding_matrix=embedding_matrix,
                                        num_classes=6,
                                        sequence_length=params.get(model_name).get('sequence_length'),
                                        dense_size=params.get(model_name).get('dense_dim'),
                                        recurrent_units=params.get(model_name).get('recurrent_units'),
                                        dropout_rate=params.get(model_name).get('dropout'),
                                        spatial_dropout_rate=params.get(model_name).get('spatial_dropout'))
    elif model_name == 'BiLSTM_attention':
        get_model_func = lambda: get_BiLSTM_Attention(embedding_matrix=embedding_matrix,
                                                     num_classes=6,
                                                     sequence_length=params.get(model_name).get('sequence_length'),
                                                        dense_size=params.get(model_name).get('dense_dim'),
                                                     recurrent_units=params.get(model_name).get('recurrent_units'),
                                                     dropout_rate=params.get(model_name).get('dropout'),
                                                     spatial_dropout_rate=params.get(model_name).get('spatial_dropout'))
    elif model_name == 'BiSRU_attention':
        get_model_func = lambda: get_BiSRU_Attention(embedding_matrix=embedding_matrix,
                                                     num_classes=6,
                                                     sequence_length=params.get(model_name).get('sequence_length'),
                                                     dense_size=params.get(model_name).get('dense_dim'),
                                                     recurrent_units=params.get(model_name).get('recurrent_units'),
                                                     dropout_rate=params.get(model_name).get('dropout'),
                                                     spatial_dropout_rate=params.get(model_name).get('spatial_dropout'))

    elif model_name == 'BiGRU_Dense':
        get_model_func = lambda: get_BiGRU_Dense(embedding_matrix=embedding_matrix,
                                        num_classes=6,
                                        sequence_length=params.get(model_name).get('sequence_length'),
                                        dense_sizes=params.get(model_name).get('dense_dims'),
                                        recurrent_units=params.get(model_name).get('recurrent_units'))
    elif model_name == 'pyramidCNN':
        get_model_func = lambda: get_pyramidCNN(embedding_matrix=embedding_matrix,
                                        num_classes=6,
                                        sequence_length=params.get(model_name).get('sequence_length'),
                                        num_of_filters=params.get(model_name).get('num_of_filters'),
                                        filter_size=params.get(model_name).get('filter_size'),
                                        num_of_blocks=params.get(model_name).get('num_of_blocks'),
                                        embedding_dropout=params.get(model_name).get('embedding_dropout'),
                                        conv_dropout=params.get(model_name).get('conv_dropout'),
                                        dense_dropout=params.get(model_name).get('dense_dropout'),
                                        use_bn=params.get(model_name).get('use_bn'),
                                        l2_weight_decay=0.0001)
    elif model_name == 'original_pyramidCNN':
        get_model_func = lambda: get__original_pyramidCNN(embedding_matrix=embedding_matrix,
                                              num_classes=6,
                                              sequence_length=params.get(model_name).get('sequence_length'),
                                              dropout_rate=params.get(model_name).get('dropout_rate'),
                                              num_of_filters=params.get(model_name).get('num_of_filters'),
                                              filter_size=params.get(model_name).get('filter_size'),
                                              num_of_blocks=params.get(model_name).get('num_of_blocks'),
                                              l2_weight_decay=0.0001)
    elif model_name == 'simpleCNN':
        get_model_func = lambda: get_simpleCNN(embedding_matrix=embedding_matrix,
                                              num_classes=6,
                                              sequence_length=params.get(model_name).get('sequence_length'),
                                              dropout_rate=params.get(model_name).get('dropout_rate'),
                                              num_of_filters=params.get(model_name).get('num_of_filters'),
                                              filter_sizes=params.get(model_name).get('filter_sizes'))
    elif model_name == 'get_simpleCNN_conv2d':
        get_model_func = lambda: get_simpleCNN_conv2d(embedding_matrix=embedding_matrix,
                                              num_classes=6,
                                              sequence_length=params.get(model_name).get('sequence_length'),
                                              dropout_rate=params.get(model_name).get('dropout_rate'),
                                              num_of_filters=params.get(model_name).get('num_of_filters'),
                                              filter_sizes=params.get(model_name).get('filter_sizes'))

    elif model_name == "BiGRU_max_avg_pool_concat":
        get_model_func = lambda: get_BiGRU_Max_Avg_Pool_concat(embedding_matrix=embedding_matrix,
                                                    num_classes=6,
                                                    sequence_length=params.get(model_name).get('sequence_length'),
                                                    recurrent_units=params.get(model_name).get('recurrent_units'),
                                                    dense_size=params.get(model_name).get('dense_dims'),
                                                    dropout_rate=params.get(model_name).get('dropout'))
    elif model_name == "get_2BiGRU_rec_dropout":
        get_model_func = lambda: get_2BiGRU_rec_dropout(embedding_matrix=embedding_matrix,
                                                                  num_classes=6,
                                                                  sequence_length=params.get(model_name).get('sequence_length'),
                                                                  recurrent_units=params.get(model_name).get('recurrent_units'),
                                                                  dense_size=params.get(model_name).get('dense_dim'),
                                                                  dropout_rate=params.get(model_name).get('dropout'))
    elif model_name == 'pyramid_gated_CNN':
        get_model_func = lambda: get_pyramid_gated_CNN(embedding_matrix=embedding_matrix,
                                              num_classes=6,
                                              sequence_length=params.get(model_name).get('sequence_length'),
                                              dropout_rate=params.get(model_name).get('dropout_rate'),
                                              num_of_filters=params.get(model_name).get('num_of_filters'),
                                              filter_size=params.get(model_name).get('filter_size'),
                                              num_of_blocks=params.get(model_name).get('num_of_blocks'),
                                              dense_size=params.get(model_name).get('dense_size'),
                                              l2_weight_decay=0.0001)
    elif model_name == '2BiSRU_rec_dropout':
        get_model_func = lambda: get_2BiSRU_rec_dropout_GlobMaxPool(embedding_matrix=embedding_matrix,
                                                    num_classes=6,
                                                    sequence_length=params.get(model_name).get('sequence_length'),
                                                    recurrent_units=params.get(model_name).get('recurrent_units'),
                                                    dense_size=params.get(model_name).get('dense_dim'),
                                                    dropout_rate=params.get(model_name).get('dropout'))

    elif model_name == '2BiLSTM_rec_dropout':
        get_model_func = lambda: get_2BiLSTM_rec_dropout(embedding_matrix=embedding_matrix,
                                                    num_classes=6,
                                                    sequence_length=params.get(model_name).get('sequence_length'),
                                                    recurrent_units=params.get(model_name).get('recurrent_units'),
                                                    dense_size=params.get(model_name).get('dense_dim'),
                                                    dropout_rate=params.get(model_name).get('dropout'),
                                                    recurrent_dropout_rate=params.get(model_name).get('recurrent_dropout'))

    elif model_name == '2BiSRU_spat_dropout':
        get_model_func = lambda: get_2BiSRU_spat_dropout(embedding_matrix=embedding_matrix,
                                                            num_classes=6,
                                                            sequence_length=params.get(model_name).get(
                                                                'sequence_length'),
                                                            recurrent_units=params.get(model_name).get(
                                                                'recurrent_units'),
                                                            dense_size=params.get(model_name).get('dense_dim'),
                                                            dropout_rate=params.get(model_name).get('dropout'),
                                                            spatial_dropout_rate=params.get(model_name).get('spatial_dropout'))
    elif model_name == '2BiGRU_spat_dropout':
        get_model_func = lambda: get_2BiGRU_spat_dropout(embedding_matrix=embedding_matrix,
                                                                     num_classes=6,
                                                                     sequence_length=params.get(model_name).get(
                                                                         'sequence_length'),
                                                                     recurrent_units=params.get(model_name).get(
                                                                         'recurrent_units'),
                                                                     dense_size=params.get(model_name).get('dense_dim'),
                                                                     dropout_rate=params.get(model_name).get('dropout'),
                                                                     spatial_dropout_rate=params.get(model_name).get(
                                                                         'spatial_dropout'))
    elif model_name == '2BiLSTM_spat_dropout':
        get_model_func = lambda: get_2BiLSTM_spat_dropout(embedding_matrix=embedding_matrix,
                                                                     num_classes=6,
                                                                     sequence_length=params.get(model_name).get(
                                                                         'sequence_length'),
                                                                     recurrent_units=params.get(model_name).get(
                                                                         'recurrent_units'),
                                                                     dense_size=params.get(model_name).get('dense_dim'),
                                                                     dropout_rate=params.get(model_name).get('dropout'),
                                                                     spatial_dropout_rate=params.get(model_name).get(
                                                                         'spatial_dropout'))
    elif model_name == 'capsuleNet':
        get_model_func = lambda: get_capsuleNet(embedding_matrix=embedding_matrix,
                                                num_classes=6,
                                                sequence_length=params.get(model_name).get('sequence_length'),
                                                recurrent_units=params.get(model_name).get('recurrent_units'),
                                                num_capsule=params.get(model_name).get('num_capsule'),
                                                dim_capsule=params.get(model_name).get('dim_capsule'),
                                                routings=params.get(model_name).get('routings'),
                                                dropout_rate=params.get(model_name).get('dropout'),
                                                spatial_dropout_rate=params.get(model_name).get('spatial_dropout'))


    # is still in development
    elif model_name == 'disan':
        get_model_func = lambda: get_diSAN(embedding_matrix=embedding_matrix,
                                           num_classes=6,
                                           sequence_length=500)

    else:
    # ============= BiGRU =============
        get_model_func = lambda: get_2BiGRU(embedding_matrix=embedding_matrix,
                                        num_classes=6,
                                        sequence_length=params.get(model_name).get('sequence_length'),
                                        dense_size=params.get(model_name).get('dense_dim'),
                                        recurrent_units=params.get(model_name).get('recurrent_units'))

    return get_model_func


class LossHistory(Callback):
    def __init__(self, initial_lr, lr_drop_koef, epochs_to_drop):
        self.initial_lr = initial_lr
        self.lr_drop_koef = lr_drop_koef
        self.epochs_to_drop = epochs_to_drop

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(self.initial_lr, self.lr_drop_koef, self.epochs_to_drop, len(self.losses)))


def define_callbacks(early_stopping_delta, early_stopping_epochs, use_lr_stratagy=True, initial_lr=0.005, lr_drop_koef=0.66, epochs_to_drop=5):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta, patience=early_stopping_epochs, verbose=1)
    callbacks_list = [early_stopping]
    if use_lr_stratagy:
        epochs_to_drop = float(epochs_to_drop)
        loss_history = LossHistory(initial_lr, lr_drop_koef, epochs_to_drop)
        lrate = LearningRateScheduler(lambda epoch: step_decay(initial_lr, lr_drop_koef, epochs_to_drop, epoch))
        callbacks_list.append(loss_history)
        callbacks_list.append(lrate)
    return callbacks_list

# ================= KL-divergence loss ======================
def abs_kullback_leibler(y_true, y_pred):
  y_true = backend.clip(y_true, backend.epsilon(), None)
  y_pred = backend.clip(y_pred, backend.epsilon(), None)
  kl = backend.sum(backend.abs((y_true - y_pred) * (backend.log(y_true / y_pred))), axis=-1)
  kl_2 = losses.kullback_leiber_divergence(y_true, y_pred)

  return kl


def _train_model(model, batch_size, train_x, train_y, val_x, val_y, opt, logger):
  best_loss = -1
  best_roc_auc = -1
  best_weights = None
  best_epoch = 0
  current_epoch = 0

  # ============== Define callbacks ==============
  history = callbacks.History()
  terminate_on_nan = callbacks.TerminateOnNaN()
  callbacks_list = [terminate_on_nan, history]

  # ============= Initialize optimizer =============
  if opt == 'adam':
      optimizer = optimizers.Adam()
      logger.info('Initialize Adam optimizer')
  elif opt == 'nadam':
      optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
      logger.info('Initialize Nadam optimizer')
  elif opt == 'rmsprop':
      optimizer = optimizers.RMSprop(clipvalue=1, clipnorm=1)
      logger.info('Initialize RMSprop optimizer')
  elif opt == 'adagrad':
    optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    logger.info('Initialize AdaGrad optimizer')
  else:
      optimizer = optimizers.RMSprop(clipvalue=1, clipnorm=1)
      logger.info('Initialize RMSprop optimizer')

  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  if logger is not None:
    model.summary(print_fn=lambda line: logger.debug(line))
  else:
    model.summary()

  # ============= Iterate through epochs =============
  while True:
    model.fit(train_x,
              train_y,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              shuffle=True,
              callbacks=callbacks_list)
    y_pred = model.predict(val_x, batch_size=batch_size)

    y_pred = np.float64(y_pred)
    val_y = np.float64(val_y)

    total_loss = 0
    for j in range(6):
      loss = log_loss(val_y[:, j], y_pred[:, j])
      total_loss += loss

    total_loss /= 6.

    roc_auc = roc_auc_score(val_y, y_pred)
    logger.debug('Epoch {0} : loss {1}, roc_auc {2}, best_loss {3}'.format(current_epoch, total_loss, roc_auc, best_loss))

    current_epoch += 1
    if total_loss < best_loss or best_loss == -1:
      best_loss = total_loss
      best_weights = model.get_weights()
      best_epoch = current_epoch
    else:
      if current_epoch - best_epoch == 5:
        break

    # if roc_auc > best_roc_auc or best_roc_auc == -1:
    #   best_roc_auc = roc_auc
    #   best_weights = model.get_weights()
    #   best_epoch = current_epoch
    # else:
    #   if current_epoch - best_epoch == 5:
    #     break

  model.set_weights(best_weights)
  return model

def train_folds(X, y, fold_count, batch_size, get_model_func, optimizer, logger):
    fold_size = len(X) // fold_count
    models = []
    val_predictions = []
    for fold_id in range(0, fold_count):
      fold_start = fold_size * fold_id
      fold_end = fold_start + fold_size

      if fold_id == fold_size - 1:
        fold_end = len(X)

      train_x = np.concatenate([X[:fold_start], X[fold_end:]])
      train_y = np.concatenate([y[:fold_start], y[fold_end:]])

      val_x = X[fold_start:fold_end]
      val_y = y[fold_start:fold_end]

      model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y, optimizer, logger)
      val_predictions.append(model.predict(val_x))
      models.append(model)

    return models, val_predictions


def train_folds_catboost(X, y, fold_count, batch_size, get_model_func, optimizer, logger):
    fold_size = len(X) // fold_count
    models = []
    val_predictions = []
    for fold_id in range(0, fold_count):
      fold_start = fold_size * fold_id
      fold_end = fold_start + fold_size

      if fold_id == fold_size - 1:
        fold_end = len(X)

      train_x = np.concatenate([X[:fold_start], X[fold_end:]])
      train_y = np.concatenate([y[:fold_start], y[fold_end:]])

      val_x = X[fold_start:fold_end]
      val_y = y[fold_start:fold_end]

      model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y, optimizer, logger)
      val_predictions.append(model.predict(val_x))
      models.append(model)

    return models, val_predictions


def train(x_train, y_train, model, batch_size, num_epochs, learning_rate=0.001, early_stopping_delta=0.0, early_stopping_epochs=10, use_lr_stratagy=True, lr_drop_koef=0.66, epochs_to_drop=5, logger=None):
    adam = optimizers.Adam(lr=learning_rate)
    rmsprop = optimizers.RMSprop(clipvalue=1, clipnorm=1)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    if logger is not None:
        model.summary(print_fn=lambda line: logger.debug(line))
    else:
        model.summary()

    callbacks_list = define_callbacks(early_stopping_delta,
                                    early_stopping_epochs,
                                    use_lr_stratagy=use_lr_stratagy,
                                    initial_lr=learning_rate,
                                    lr_drop_koef=lr_drop_koef,
                                    epochs_to_drop=epochs_to_drop)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=callbacks_list,
                     validation_split=0.1,
                     shuffle=True,
                     verbose=1)
    return hist


def continue_train(x_train, y_train, model, batch_size, num_epochs, learning_rate_decay, learning_rate=0.001, early_stopping_delta=0.0, early_stopping_iters=10, use_lr_stratagy=True, lr_drop_koef=0.66, epochs_to_drop=5):
    callbacks_list = define_callbacks(early_stopping_delta,
                                    early_stopping_iters,
                                    use_lr_stratagy=use_lr_stratagy,
                                    initial_lr=learning_rate,
                                    lr_drop_koef=lr_drop_koef,
                                    epochs_to_drop=epochs_to_drop)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=callbacks_list,
                     validation_split=0.1,
                     shuffle=True,
                     verbose=1)
    return hist


class Params(object):
    def __init__(self, config=None):
        self._params = self._common_init()
        config_params = self._load_from_file(config)
        self._update_params(config_params)

    def _load_from_file(self, fname):
        if fname is None:
            return {}
        with open(fname) as f:
            return json.loads(f.read())

    def _common_init(self):
        common_params = {
                    'warm_start': False,
                    'model_file': None,
                    'batch_size': 256,
                    'num_epochs': 10,
                    'learning_rate': 0.001,
                    'early_stopping_delta': 0.001,
                    'early_stopping_epochs': 2,
                    'use_lr_stratagy': True,
                    'lr_drop_koef': 0.5,
                    'epochs_to_drop': 1,
                    'l2_weight_decay':0.0001,
                    'dropout_val': 0.5,
                    'dense_dim': 32}

        params = {'cnn': common_params,
                  'lstm': common_params,
                  'concat': common_params,
                  'gru': common_params}

        params['cnn']['num_filters'] = 64
        params['lstm']['lstm_dim'] = 50
        params['concat']['num_filters'] = 64
        params['concat']['lstm_dim'] = 50

        params['catboost'] = {
                    'add_bow': False,
                    'bow_top': 100,
                    'iterations': 1000,
                    'depth': 6,
                    'rsm': 1,
                    'learning_rate': 0.01,
                    'device_config': None}
        return params

    def _update_params(self, params):
        if params is not None and params:
            for key in params.keys():
                if isinstance(params[key], dict):
                    self._params.setdefault(key, {})
                    self._params[key].update(params[key])
                else:
                    self._params.setdefault(key, None)
                    self._params[key] = params[key]

    def get(self, key):
        return self._params.get(key, None)
