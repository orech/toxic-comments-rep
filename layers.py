# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class DiSAN(Layer):
  def __init__(self, output_dim, hidden_dim, return_mask=False, **kwargs):
    self.output_dim = output_dim
    self.supports_masking = True
    self.return_mask = return_mask
    self.hidden_dim = hidden_dim
    super(DiSAN, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.W_h = self.add_weight(name='W_h', shape=(input_shape[1], self.hidden_dim), initializer='uniform', trainable=True)
    self.b_h = self.add_weight(name='b_h', shape=())
    super(DiSAN, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, x):
    return K.dot(x, self.kernel)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    Implementation of taken from DeepMoji project: https://github.com/bfelbo/DeepMoji
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


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