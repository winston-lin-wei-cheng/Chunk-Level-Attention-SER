#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros


head_dim = 50 

class ScaledDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], head_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(ScaledDotProductAttention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (head_dim**0.5)
        QK = K.softmax(QK)
        Z = K.batch_dot(QK,WV)
        return Z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], head_dim)

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape
