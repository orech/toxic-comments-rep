# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
from keras.engine.topology import Layer
from functools import reduce
from operator import mul
import tensorflow as tf
import numpy as np
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.layers import Recurrent
from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


class disan(Layer):
  def __init__(self, keep_prob, is_train, wd, **kwargs):
      self.keep_prob = keep_prob
      self.is_train = tf.cast(is_train, tf.bool)
      self.wd = wd

      super(disan, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    bs = input_shape[0]
    sl = input_shape[1]
    vec = input_shape[2]

    self.W_h = self.add_weight(name='W_h', shape=(vec, vec), initializer='uniform', trainable=True) # sl x vec
    self.b_h = self.add_weight(name='b_h', shape=(vec,), initializer='uniform', trainable=True) # vec

    self.W_1 = self.add_weight(name='W_1', shape=(vec, vec), initializer='uniform', trainable=True)
    self.W_2 = self.add_weight(name='W_2', shape=(vec, vec), initializer='uniform', trainable=True)
    self.b = self.add_weight(name='b', shape=(vec,), initializer='uniform', trainable=True)

    self.W_f1 = self.add_weight(name='W_f1', shape=(vec, vec), initializer='uniform', trainable=True)
    self.W_f2 = self.add_weight(name='W_f2', shape=(vec, vec), initializer='uniform', trainable=True)
    self.b_f = self.add_weight(name='b_f', shape=(vec,), initializer='uniform', trainable=True)

    self.W_fg1 = self.add_weight(name='W_fg1', shape=(vec, vec), initializer='uniform', trainable=True)
    self.W_fg2 = self.add_weight(name='W_fg2', shape=(vec, vec), initializer='uniform', trainable=True)
    self.b_fg1 = self.add_weight(name='b_fg1', shape=(vec,), initializer='uniform', trainable=True)
    self.b_fg2 = self.add_weight(name='b_fg2', shape=(vec,), initializer='uniform', trainable=True)
    self.b_fg3 = self.add_weight(name='b_fg3', shape=(vec,), initializer='uniform', trainable=True)

    self.W_md1 = self.add_weight(name='W_md1', shape=(2*vec, 2*vec), initializer='uniform', trainable=True)
    self.W_md2 = self.add_weight(name='W_md2', shape=(2*vec, 2*vec), initializer='uniform', trainable=True)
    self.b_md1 = self.add_weight(name='b_md1', shape=(2*vec,), initializer='uniform', trainable=True)
    self.b_md2 = self.add_weight(name='b_md2', shape=(2*vec,), initializer='uniform', trainable=True)

    # mask generation
    sl_indices = tf.range(sl, dtype=tf.int32)
    sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)

    self.undirected_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
    self.f_direct_mask = tf.greater(sl_row, sl_col)
    self.b_direct_mask = tf.greater(sl_col, sl_row)

    super(disan, self).build(input_shape)  # Be sure to call this somewhere!


  # ---------------   DiSAN Interface  ----------------
  def call(self, inputs, rep_mask, scope=None,
            keep_prob=1., activation='elu',
            tensor_dict=None, name=''):
    with tf.variable_scope(scope or 'DiSAN'):
      with tf.variable_scope('ct_attn'):
        fw_res = self.directional_attention_with_dense(
          inputs, rep_mask, 'forward', 'dir_attn_fw',
          keep_prob, self.is_train, self.wd, activation,
          tensor_dict=tensor_dict, name=name + '_fw_attn')
        # bw_res = self.directional_attention_with_dense(
        #   inputs, rep_mask, 'backward', 'dir_attn_bw',
        #   keep_prob, self.is_train, self.wd, activation,
        #   tensor_dict=tensor_dict, name=name + '_bw_attn')

        # seq_rep = tf.concat([fw_res, bw_res], -1)

      # with tf.variable_scope('sent_enc_attn'):
      #   sent_rep = self.multi_dimensional_attention(
      #     seq_rep, rep_mask, 'multi_dimensional_attention',
      #     keep_prob, self.is_train, self.wd, activation,
      #     tensor_dict=tensor_dict, name=name + '_attn')
        return fw_res

  # --------------- supporting networks ----------------
  def directional_attention_with_dense(self, inputs, rep_mask=None, direction=None, scope=None,
                                       keep_prob=1., is_train=None, wd=0., activation='relu',
                                       tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
      return scale * tf.nn.tanh(1. / scale * x)


    bs, sl, vec = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
    ivec = inputs.get_shape()[2]


    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
      if direction is None:
        direct_mask = self.undirected_mask
      elif direction == 'forward':
        direct_mask = self.f_direct_mask
      else:
        direct_mask = self.b_direct_mask

      direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
      rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
      attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

      # non-linear
      rep_map = self.bn_dense_layer(inputs, ivec, self.W_h, self.b_h, 0., 'bn_dense_map', activation, False, wd, keep_prob, is_train)
      rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
      rep_map_dp = self.dropout(rep_map, keep_prob, is_train)

      # attention
      with tf.variable_scope('attention'):  # bs,sl,sl,vec
        dependent = self.linear(rep_map_dp, ivec, self.W_f1, scope='linear_dependent')  # bs,sl,vec
        dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
        head = self.linear(rep_map_dp, ivec, self.W_f2, scope='linear_head')  # bs,sl,vec
        head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

        logits = scaled_tanh(dependent_etd + head_etd + self.b_f, 5.0)  # bs,sl,sl,vec

        logits_masked = self.exp_mask_for_high_rank(logits, attn_mask)
        attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
        attn_score = self.mask_for_high_rank(attn_score, attn_mask)

        attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

      with tf.variable_scope('output'):
        # input gate
        fusion_gate = tf.nn.sigmoid(
          self.linear(rep_map, ivec, self.W_fg1, self.b_fg1, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
          self.linear(attn_result, ivec, self.W_fg2, self.b_fg2, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
          self.b_fg3)
        output = fusion_gate * rep_map + (1 - fusion_gate) * attn_result
        output = self.mask_for_high_rank(output, rep_mask)

      # save attn
      if tensor_dict is not None and name is not None:
        tensor_dict[name + '_dependent'] = dependent
        tensor_dict[name + '_head'] = head
        tensor_dict[name] = attn_score
        tensor_dict[name + '_gate'] = fusion_gate
      return output

  def multi_dimensional_attention(self, rep_tensor, rep_mask, scope=None,
                                  keep_prob=1., is_train=None, wd=0., activation='relu',
                                  tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
      map1 = self.bn_dense_layer(rep_tensor, ivec, self.W_md1, self.b_md1, 0., 'bn_dense_map1', activation, False, wd, keep_prob, is_train)
      map2 = self.bn_dense_layer(map1, ivec, self.W_md2, self.b_md2, 0., 'bn_dense_map2', 'linear', False, wd, keep_prob, is_train)
      map2_masked = self.exp_mask_for_high_rank(map2, rep_mask, name='exp_mask_md')

      soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
      attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

      # save attn
      if tensor_dict is not None and name is not None:
        tensor_dict[name] = soft

      return attn_output

  def bn_dense_layer(self, input_tensor, hn, W, bias=None, bias_start=0.0, scope=None,
                     activation='relu', enable_bn=True,
                     wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
      is_train = False

    # activation
    if activation == 'linear':
      activation_func = tf.identity
    elif activation == 'relu':
      activation_func = tf.nn.relu
    elif activation == 'elu':
      activation_func = tf.nn.elu
    # elif activation == 'selu':
    #   activation_func = selu
    else:
      raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
      linear_map = self.linear(input_tensor, hn, W, bias, bias_start, 'linear_map',
                          False, wd, keep_prob, is_train)
      # if enable_bn:
      #   linear_map = tf.contrib.layers.batch_norm(
      #     linear_map, center=True, scale=True, is_training=is_train, scope='bn')
      return activation_func(linear_map)

  def dropout(self, x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
      assert is_train is not None
      if keep_prob < 1.0:
        d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        out = tf.cond(is_train, lambda: d, lambda: x)
        return out
      return x

  def linear(self, args, output_size, W, bias=None, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
             is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
      raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
      args = [args]

    flat_args = [self.flatten(arg, 1) for arg in args]  # for dense layer [(-1, d)]
    # if input_keep_prob < 1.0:
    #   assert is_train is not None
    #   flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg) for arg in flat_args]
    flat_out = self._linear(flat_args, W, bias, bias_start=bias_start, scope=scope)  # dense
    out = self.reconstruct(flat_out, args[0], 1)  # ()
    if squeeze:
      out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])

    if wd:
      self.add_reg_without_bias()

    return out

  def _linear(self, xs, W, bias=None, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
      x = tf.concat(xs, -1)
      if bias is not None:
        out = tf.matmul(x, W) + bias
      else:
        out = tf.matmul(x, W)
      return out

  def flatten(self, tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

  def reconstruct(self, tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list()  # original shape
    tensor_shape = tensor.get_shape().as_list()  # current shape
    ref_stop = len(ref_shape) - keep  # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]  #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]  #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

  def mask_for_high_rank(self, val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name='mask_for_high_rank')

  def exp_mask_for_high_rank(self, val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER, name='exp_mask_for_high_rank')

  def selu(self, x):
    with tf.name_scope('elu') as scope:
      alpha = 1.6732632423543772848170429916717
      scale = 1.0507009873554804934193349852946
      return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

  def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
      if len(var.get_shape().as_list()) <= 1: continue
      tf.add_to_collection('reg_vars', var)
      counter += 1

    return counter

  def compute_output_shape(self, input_shape):
      return (input_shape[0], input_shape[2])


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    Implementation of this layer is taken from DeepMoji project: https://github.com/bfelbo/DeepMoji
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


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
  """Apply `y . w + b` for every temporal slice y of x.

  # Arguments
      x: input tensor.
      w: weight matrix.
      b: optional bias vector.
      dropout: wether to apply dropout (same dropout mask
          for every temporal slice of the input).
      input_dim: integer; optional dimensionality of the input.
      output_dim: integer; optional dimensionality of the output.
      timesteps: integer; optional number of timesteps.
      training: training phase tensor or boolean.
  # Implementation is taken from https://github.com/titu1994/keras-SRU

  # Returns
      Output tensor.
  """
  if not input_dim:
    input_dim = K.shape(x)[2]
  if not timesteps:
    timesteps = K.shape(x)[1]
  if not output_dim:
    output_dim = K.int_shape(w)[1]

  if dropout is not None and 0. < dropout < 1.:
    # apply the same dropout pattern at every timestep
    ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
    dropout_matrix = K.dropout(ones, dropout)
    expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
    x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

  # collapse time dimension and batch dimension together
  x = K.reshape(x, (-1, input_dim))
  x = K.dot(x, w)
  if b is not None:
    x = K.bias_add(x, b)
  # reshape to 3D tensor
  if K.backend() == 'tensorflow':
    x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
    x.set_shape([None, None, output_dim])
  else:
    x = K.reshape(x, (-1, timesteps, output_dim))
  return x

class SRU(Recurrent):
  """Simple Recurrent Unit - https://arxiv.org/pdf/1709.02755.pdf.
  This implementation is taken from https://github.com/titu1994/keras-SRU

  # Arguments
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use
          (see [activations](../activations.md)).
          If you pass None, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step
          (see [activations](../activations.md)).
      use_bias: Boolean, whether the layer uses a bias vector.
      project_input: Add a projection vector to the input
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs.
          (see [initializers](../initializers.md)).
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state.
          (see [initializers](../initializers.md)).
      bias_initializer: Initializer for the bias vector
          (see [initializers](../initializers.md)).
      unit_forget_bias: Boolean.
          If True, add 1 to the bias of the forget gate at initialization.
          Setting it to true will also force `bias_initializer="zeros"`.
          This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix
          (see [regularizer](../regularizers.md)).
      recurrent_regularizer: Regularizer function applied to
          the `recurrent_kernel` weights matrix
          (see [regularizer](../regularizers.md)).
      bias_regularizer: Regularizer function applied to the bias vector
          (see [regularizer](../regularizers.md)).
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation").
          (see [regularizer](../regularizers.md)).
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix
          (see [constraints](../constraints.md)).
      recurrent_constraint: Constraint function applied to
          the `recurrent_kernel` weights matrix
          (see [constraints](../constraints.md)).
      bias_constraint: Constraint function applied to the bias vector
          (see [constraints](../constraints.md)).
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.
      implementation: one of {0, 1, or 2}.
          If set to 0, the SRU will use
          an implementation that uses fewer, larger matrix products,
          thus running faster on CPU but consuming more memory.
          If set to 1, the SRU will use more matrix products,
          but smaller ones, thus running slower
          (may actually be faster on GPU) while consuming less memory.
          If set to 2, the SRU will combine the input gate,
          the forget gate and the output gate into a single matrix,
          enabling more time-efficient parallelization on the GPU.
          Note: SRU dropout must be shared for all gates,
          resulting in a slightly reduced regularization.

  # References
      - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
      - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
      - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
      - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
      - [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)
  """

  @interfaces.legacy_recurrent_support
  def __init__(self, units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               project_input=False,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               return_sequences=False,
               **kwargs):
    super(SRU, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias
    self.project_input = project_input

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_spec = [InputSpec(shape=(None, self.units)),
                       InputSpec(shape=(None, self.units))]

    self.implementation = implementation
    self.return_sequences = return_sequences

  def build(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_dim = input_shape[2]
    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))  # (timesteps, batchsize, inputdim)

    self.states = [None, None]
    if self.stateful:
      self.reset_states()

    if self.project_input:
      self.kernel_dim = 4
    elif self.input_dim != self.units:
      self.kernel_dim = 4
    else:
      self.kernel_dim = 3

    self.kernel = self.add_weight(shape=(self.input_dim, self.units * self.kernel_dim),
                                  name='kernel',
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)

    if self.use_bias:
      if self.unit_forget_bias:
        def bias_initializer(shape, *args, **kwargs):
          return K.concatenate([
            self.bias_initializer((self.units,), *args, **kwargs),
            initializers.Ones()((self.units,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer

      self.bias = self.add_weight(shape=(self.units * 2,),
                                  name='bias',
                                  initializer=bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None

    self.kernel_w = self.kernel[:, :self.units]
    self.kernel_f = self.kernel[:, self.units: self.units * 2]
    self.kernel_r = self.kernel[:, self.units * 2: self.units * 3]

    if self.kernel_dim == 4:
      self.kernel_p = self.kernel[:, self.units * 3: self.units * 4]
    else:
      self.kernel_p = None

    if self.use_bias:
      self.bias_f = self.bias[:self.units]
      self.bias_r = self.bias[self.units: self.units * 2]
    else:
      self.bias_f = None
      self.bias_r = None
    self.built = True

  def preprocess_input(self, inputs, training=None):
    if self.implementation == 0:
      input_shape = K.int_shape(inputs)
      input_dim = input_shape[2]
      timesteps = input_shape[1]

      x_w = _time_distributed_dense(inputs, self.kernel_w, None,
                                    self.dropout, input_dim, self.units,
                                    timesteps, training=training)
      x_f = _time_distributed_dense(inputs, self.kernel_f, self.bias_f,
                                    self.dropout, input_dim, self.units,
                                    timesteps, training=training)
      x_r = _time_distributed_dense(inputs, self.kernel_r, self.bias_r,
                                    self.dropout, input_dim, self.units,
                                    timesteps, training=training)

      x_f = self.recurrent_activation(x_f)
      x_r = self.recurrent_activation(x_r)

      if self.kernel_dim == 4:
        x_p = _time_distributed_dense(inputs, self.kernel_p, None,
                                      self.dropout, input_dim, self.units,
                                      timesteps, training=training)

        return K.concatenate([x_w, x_f, x_r, x_p], axis=2)
      else:
        return K.concatenate([x_w, x_f, x_r], axis=2)
    else:
      return inputs

  def get_constants(self, inputs, training=None):
    constants = []
    if self.implementation != 0 and 0 < self.dropout < 1:
      input_shape = K.int_shape(inputs)  # (timesteps, batchsize, inputdim)
      input_dim = input_shape[-1]
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, int(input_dim)))

      def dropped_inputs():
        return K.dropout(ones, self.dropout)

      dp_mask = [K.in_train_phase(dropped_inputs,
                                  ones,
                                  training=training) for _ in range(3)]
      constants.append(dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(3)])

    if 0 < self.recurrent_dropout < 1:
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, self.units * self.kernel_dim))

      def dropped_inputs():
        return K.dropout(ones, self.recurrent_dropout)

      rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                      ones,
                                      training=training) for _ in range(self.kernel_dim)]
      constants.append(rec_dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(self.kernel_dim)])
    return constants

  def step(self, inputs, states):
    h_tm1 = states[0]  # not used
    c_tm1 = states[1]
    dp_mask = states[2]
    rec_dp_mask = states[3]

    if self.implementation == 2:
      z = K.dot(inputs * dp_mask[0], self.kernel)
      z = z * rec_dp_mask[0]

      z0 = z[:, :self.units]

      if self.use_bias:
        z_bias = K.bias_add(z[:, self.units: self.units * 3], self.bias)
        z_bias = self.recurrent_activation(z_bias)
        z1 = z_bias[:, :self.units]
        z2 = z_bias[:, self.units: 2 * self.units]
      else:
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]

      if self.kernel_dim == 4:
        z3 = z[:, 3 * self.units: 4 * self.units]
      else:
        z3 = None

      f = z1
      r = z2

      c = f * c_tm1 + (1 - f) * z0
      if self.kernel_dim == 4:
        h = r * self.activation(c) + (1 - r) * z3
      else:
        h = r * self.activation(c) + (1 - r) * inputs
    else:
      if self.implementation == 0:
        x_w = inputs[:, :self.units]
        x_f = inputs[:, self.units: 2 * self.units]
        x_r = inputs[:, 2 * self.units: 3 * self.units]
        if self.kernel_dim == 4:
          x_w_x = inputs[:, 3 * self.units: 4 * self.units]
        else:
          x_w_x = None
      elif self.implementation == 1:
        x_w = K.dot(inputs * dp_mask[0], self.kernel_w)
        x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
        x_r = K.dot(inputs * dp_mask[2], self.kernel_r) + self.bias_r

        x_f = self.recurrent_activation(x_f)
        x_r = self.recurrent_activation(x_r)

        if self.kernel_dim == 4:
          x_w_x = K.dot(inputs * dp_mask[0], self.kernel_p)
        else:
          x_w_x = None
      else:
        raise ValueError('Unknown `implementation` mode.')

      w = x_w * rec_dp_mask[0]
      f = x_f
      r = x_r

      c = f * c_tm1 + (1 - f) * w
      if self.kernel_dim == 4:
        h = r * self.activation(c) + (1 - r) * x_w_x
      else:
        h = r * self.activation(c) + (1 - r) * inputs

    if 0 < self.dropout + self.recurrent_dropout:
      h._uses_learning_phase = True

    return h, [h, c]

  def get_config(self):
    config = {'units': self.units,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'unit_forget_bias': self.unit_forget_bias,
              'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'activity_regularizer': regularizers.serialize(self.activity_regularizer),
              'kernel_constraint': constraints.serialize(self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout}
    base_config = super(SRU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def squash(x, axis=-1):
  s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
  scale = K.sqrt(s_squared_norm + K.epsilon())
  return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
  def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
               activation='default', **kwargs):
    super(Capsule, self).__init__(**kwargs)
    self.num_capsule = num_capsule
    self.dim_capsule = dim_capsule
    self.routings = routings
    self.kernel_size = kernel_size
    self.share_weights = share_weights
    if activation == 'default':
      self.activation = squash
    else:
      self.activation = Activation(activation)

  def build(self, input_shape):
    super(Capsule, self).build(input_shape)
    input_dim_capsule = input_shape[-1]
    if self.share_weights:
      self.W = self.add_weight(name='capsule_kernel',
                               shape=(1, input_dim_capsule,
                                      self.num_capsule * self.dim_capsule),
                               # shape=self.kernel_size,
                               initializer='glorot_uniform',
                               trainable=True)
    else:
      input_num_capsule = input_shape[-2]
      self.W = self.add_weight(name='capsule_kernel',
                               shape=(input_num_capsule,
                                      input_dim_capsule,
                                      self.num_capsule * self.dim_capsule),
                               initializer='glorot_uniform',
                               trainable=True)

  def call(self, u_vecs):
    if self.share_weights:
      u_hat_vecs = K.conv1d(u_vecs, self.W)
    else:
      u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

    batch_size = K.shape(u_vecs)[0]
    input_num_capsule = K.shape(u_vecs)[1]
    u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                        self.num_capsule, self.dim_capsule))
    u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
    # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

    b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
    for i in range(self.routings):
      b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
      c = K.softmax(b)
      c = K.permute_dimensions(c, (0, 2, 1))
      b = K.permute_dimensions(b, (0, 2, 1))
      outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
      if i < self.routings - 1:
        b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

    return outputs

  def compute_output_shape(self, input_shape):
    return (None, self.num_capsule, self.dim_capsule)