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
from functools import reduce
from operator import mul
import tensorflow as tf

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


# class DiSAN(Layer):
#   def __init__(self, keep_prob, is_train, wd, **kwargs):
#     self.keep_prob = keep_prob
#     self.is_train = tf.cast(is_train, tf.bool)
#     self.wd = wd
#     super(DiSAN, self).__init__(**kwargs)
#
#   def build(self, input_shape):
#     # Create a trainable weight variable for this layer.
#     self.W_h = self.add_weight(name='W_h', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True) # sl x vec
#     self.b_h = self.add_weight(name='b_h', shape=(input_shape[2],), initializer='uniform', trainable=True) # vec
#
#     self.W_1 = self.add_weight(name='W_1', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
#     self.W_2 = self.add_weight(name='W_2', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
#     self.b = self.add_weight(name='b', shape=(input_shape[2],), initializer='uniform', trainable=True)
#
#     self.W_f1 = self.add_weight(name='W_f1', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
#     self.W_f2 = self.add_weight(name='W_f2', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
#     self.b_f = self.add_weight(name='b_f', shape=(input_shape[2],), initializer='uniform', trainable=True)
#
#     self.W_fg1 = self.add_weight(name='W_fg1', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
#     self.W_fg2 = self.add_weight(name='W_fg2', shape=(input_shape[1], input_shape[2]), initializer='uniform', trainable=True)
#     self.b_fg = self.add_weight(name='b_fg', shape=(input_shape[2],), initializer='uniform', trainable=True)
#
#     super(DiSAN, self).build(input_shape)  # Be sure to call this somewhere!
#
#   def call(self, rep_tensor, tensor_dict=None):
#     # rep_mask is a tensor [bs x sl]
#     fw_res = self.directional_attention_with_dense(rep_tensor=rep_tensor,
#                                                    direction='forward',
#                                                    keep_prob=self.keep_prob,
#                                                    is_train=self.is_train,
#                                                    wd=self.wd,
#                                                    tensor_dict=tensor_dict,
#                                                    name='_fw_attn')
#     bw_res = self.directional_attention_with_dense(rep_tensor=rep_tensor,
#                                                    direction='backward',
#                                                    keep_prob=self.keep_prob,
#                                                    is_train=self.is_train,
#                                                    wd=self.wd,
#                                                    tensor_dict=tensor_dict,
#                                                    name='_bw_attn')
#
#     seq_rep = tf.concat([fw_res, bw_res], -1)
#
#     sent_rep = self.multi_dimensional_attention(
#       seq_rep, self.rep_mask, 'multi_dimensional_attention',
#       self.keep_prob, self.is_train, self.wd,
#       tensor_dict=tensor_dict, name='_attn')
#
#     return sent_rep
#
#
#   def flatten(self, tensor, keep):
#     fixed_shape = tensor.get_shape().as_list()
#     start = len(fixed_shape) - keep
#     left = reduce(mul, [fixed_shape[i] or K.shape(tensor)[i] for i in range(start)])
#     out_shape = [left] + [fixed_shape[i] or K.shape(tensor)[i] for i in range(start, len(fixed_shape))]
#     flat = K.reshape(tensor, out_shape)
#     return flat
#
#   def add_reg_without_bias(self, scope=None):
#     scope = scope or K.get_variable_scope().name
#     variables = K.get_collection(K.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
#     counter = 0
#     for var in variables:
#       if len(var.get_shape().as_list()) <= 1: continue
#       K.add_to_collection('reg_vars', var)
#       counter += 1
#
#   def reconstruct(self, tensor, ref, keep, dim_reduced_keep=None):
#     dim_reduced_keep = dim_reduced_keep or keep
#
#     ref_shape = ref.get_shape().as_list()  # original shape
#     tensor_shape = tensor.get_shape().as_list()  # current shape
#     ref_stop = len(ref_shape) - keep  # flatten dims list
#     tensor_start = len(tensor_shape) - dim_reduced_keep  # start
#     pre_shape = [ref_shape[i] or K.shape(ref)[i] for i in range(ref_stop)]  #
#     keep_shape = [tensor_shape[i] or K.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]  #
#     target_shape = pre_shape + keep_shape
#     out = K.reshape(tensor, target_shape)
#     return out
#
#   def directional_attention_with_dense(self, rep_tensor, direction=None, keep_prob=1., is_train=None, wd=0., tensor_dict=None, name=None):
#     # ============= functions =============
#     def scaled_tanh(self, x, scale=5.):
#       return scale * tf.nn.tanh(1. / scale * x)
#
#     def mask_for_high_rank(val, val_mask, name=None):
#       val_mask = K.expand_dims(val_mask, -1)
#       return K.multiply(val, K.cast(val_mask, K.float32), name=name or 'mask_for_high_rank')
#
#
#     def exp_mask_for_high_rank(val, val_mask, name=None):
#       val_mask = K.expand_dims(val_mask, -1)
#       return K.add(val, (1 - K.cast(val_mask, K.float32)) * VERY_NEGATIVE_NUMBER,
#                     name=name or 'exp_mask_for_high_rank')
#
#
#
#     # ============================================
#
#
#     bs, sl, vec = K.shape(rep_tensor)[0], K.shape(rep_tensor)[1], K.shape(rep_tensor)[2]
#     ivec = rep_tensor.get_shape()[2]
#     mask = tf.ones(shape=[sl, sl])
#     rep_mask = tf.cast(mask, tf.bool)
#     # mask generation
#     sl_indices = tf.range(sl, dtype=tf.int32)
#     sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)  # n x n tensor (sl x sl)
#     if direction is None:
#       direct_mask = K.cast(K.diag(- K.ones([sl], K.int32)) + 1, K.bool)
#     else:
#       if direction == 'forward':
#         direct_mask = K.greater(sl_row, sl_col)
#       else:
#         direct_mask = K.greater(sl_col, sl_row)
#     direct_mask_tile = K.tile(K.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
#     rep_mask_tile = K.tile(K.expand_dims(rep_mask, 0), [bs, 1, 1])  # bs,sl,sl
#     attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl
#
#     # non-linear
#     linear_map = self.linear(self.W_h, rep_tensor, ivec, True, 0., 'linear_map', False, wd, keep_prob, is_train, self.b_h)
#     # if enable_bn:
#     #   linear_map = tf.contrib.layers.batch_norm(linear_map, center=True, scale=True, is_training=is_train, scope='bn')
#     rep_map = tf.nn.relu(linear_map)
#
#     rep_map_tile = K.tile(K.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
#     # dropout
#     rep_map_dp = self.dropout(rep_map, keep_prob, is_train)
#
#     # attention
#     dependent = self.linear(self.W_1, rep_map_dp, ivec, False, scope='linear_dependent', b=None)  # bs,sl,vec
#     dependent_etd = K.expand_dims(dependent, 1)  # bs,1,sl,vec
#     head = self.linear(self.W_2, rep_map_dp, ivec, False, scope='linear_head', b=None)  # bs,sl,vec
#     head_etd = K.expand_dims(head, 2)  # bs,sl,1,vec
#
#     logits = scaled_tanh(dependent_etd + head_etd + self.b, 5.0)  # bs,sl,sl,vec
#
#     logits_masked = exp_mask_for_high_rank(logits, attn_mask)
#     attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
#     attn_score = mask_for_high_rank(attn_score, attn_mask)
#
#     attn_result = K.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec
#
#     # output
#     # input gate
#     fusion_gate = tf.nn.sigmoid(
#       self.linear(self.W_f1, rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train, b=None) +
#       self.linear(self.W_f2, attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train,
#                   b=None) + self.b_f)
#     output = fusion_gate * rep_map + (1 - fusion_gate) * attn_result
#     output = mask_for_high_rank(output, rep_mask)
#
#     # save attn
#     if tensor_dict is not None and name is not None:
#       tensor_dict[name + '_dependent'] = dependent
#       tensor_dict[name + '_head'] = head
#       tensor_dict[name] = attn_score
#       tensor_dict[name + '_gate'] = fusion_gate
#     return output
#
#
#   def linear(self, W, args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
#                is_train=None, b=None):
#       if args is None or (isinstance(args, (tuple, list)) and not args):
#         raise ValueError("`args` must be specified")
#       if not isinstance(args, (tuple, list)):
#         args = [args]
#
#       flat_args = [self.flatten(arg, 1) for arg in args]  # for dense layer [(-1, d)]
#       # if input_keep_prob < 1.0:
#       #   assert is_train is not None
#       #   flat_args = [tf.cond(is_train, lambda: K.nn.dropout(arg, input_keep_prob), lambda: arg) for arg in flat_args]
#       flat_out = self._linear(flat_args, bias, W=W, b=b, scope=scope)  # dense
#       out = self.reconstruct(flat_out, args[0], 1)  # ()
#       if squeeze:
#         out = K.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
#
#       if wd:
#         self.add_reg_without_bias()
#
#       return out
#
#   def _linear(self, xs, bias, W, b=None, scope=None):
#     x = tf.concat(xs, -1)
#     input_size = x.get_shape()[-1]
#
#     if bias is not None:
#       out = tf.matmul(W, x) + b
#     else:
#       out = tf.matmul(W, x)
#     return out
#
#
#   def compute_output_shape(self, input_shape):
#     return (input_shape[0], 2*input_shape[2])
#
#   def dropout(self, x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
#     with tf.name_scope(name or "dropout"):
#       assert is_train is not None
#       if keep_prob < 1.0:
#         d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
#         out = tf.cond(is_train, lambda: d, lambda: x)
#         return out
#       return x
#
#   def multi_dimensional_attention(self, rep_tensor, rep_mask, scope=None,
#                                   keep_prob=1., is_train=None, wd=0., activation='elu',
#                                   tensor_dict=None, name=None):
#     bs, sl, vec = K.shape(rep_tensor)[0], K.shape(rep_tensor)[1], K.shape(rep_tensor)[2]
#     ivec = rep_tensor.get_shape()[2]
#     linear_map = self.linear(self.W_fg1, rep_tensor, ivec, True, 0., 'linear_map', False, wd, keep_prob, is_train,
#                              self.b_fg1)
#     map1 = tf.nn.relu(linear_map)
#
#     linear_map = self.linear(self.W_fg2, rep_tensor, ivec, True, 0., 'linear_map', False, wd, keep_prob, is_train,
#                              self.b_fg2)
#     map2 = tf.nn.relu(linear_map)
#
#     map2_masked = self.exp_mask_for_high_rank(map2, rep_mask)
#
#     soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
#     attn_output = K.reduce_sum(soft * rep_tensor, 1)  # bs, vec
#
#     # save attn
#     if tensor_dict is not None and name is not None:
#       tensor_dict[name] = soft
#
#     return attn_output

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

    # inputs = tf.transpose(inputs, [0, 2, 1])

    bs, sl, vec = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
    ivec = inputs.get_shape()[2]


    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
      # mask generation
      sl_indices = tf.range(sl, dtype=tf.int32)
      sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
      if direction is None:
        direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
      else:
        if direction == 'forward':
          direct_mask = tf.greater(sl_row, sl_col)
        else:
          direct_mask = tf.greater(sl_col, sl_row)
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
      #   output = self.mask_for_high_rank(output, rep_mask)
      #
      # # save attn
      # if tensor_dict is not None and name is not None:
      #   tensor_dict[name + '_dependent'] = dependent
      #   tensor_dict[name + '_head'] = head
      #   tensor_dict[name] = attn_score
      #   tensor_dict[name + '_gate'] = fusion_gate
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
      if enable_bn:
        linear_map = tf.contrib.layers.batch_norm(
          linear_map, center=True, scale=True, is_training=is_train, scope='bn')
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
    if input_keep_prob < 1.0:
      assert is_train is not None
      flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                   # for dense layer [(-1, d)]
                   for arg in flat_args]
    flat_out = self._linear(flat_args, output_size, W, bias, bias_start=bias_start, scope=scope)  # dense
    out = self.reconstruct(flat_out, args[0], 1)  # ()
    if squeeze:
      out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])

    if wd:
      self.add_reg_without_bias()

    return out

  def _linear(self, xs, output_size, W, bias=None, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
      x = tf.concat(xs, -1)
      input_size = x.get_shape()[-1]
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