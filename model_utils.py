#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import SimpleRNN, Lambda, Input, Add, TimeDistributed, Concatenate, Dot
import random
from keras import backend as K
from transformer import ScaledDotProductAttention, LayerNormalization
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99


def crop(dimension, start, end):
    def func(x):
        if dimension == 0: # crop by batch
            return x[start: end]
        if dimension == 1: # crop by time-step
            return x[:, start: end]
        if dimension == 2: # crop by feature
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def reshape():
    def func(x):
        feat_num = 130
        C = 11
        return K.reshape(x, (1, C, feat_num))
    return Lambda(func)

def repeat():  
    def func(x):
        C = 11
        return K.repeat_elements(x, C, 0)
    return Lambda(func)

def mean():  
    def func(x):
        return K.mean(x, axis=1, keepdims=False)
    return Lambda(func)

def atten_gated(feat_num, C):
    inputs = Input((C, feat_num))
    a = TimeDistributed(Dense(1))(inputs)
    a_weighted = Activation('sigmoid')(a)
    model = Model(inputs=inputs, outputs=a_weighted)
    return model

def atten_rnn(feat_num, C):
    inputs = Input((C, feat_num)) 
    encode = SimpleRNN(units=feat_num, activation='tanh', return_sequences=True)(inputs)
    score_first_part = Dense(feat_num, use_bias=False)(encode)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(feat_num,))(encode)
    score = Dot(axes=(2, 1))([score_first_part, h_t])
    attention_weights = Activation('softmax')(score)
    context_vector = Dot(axes=(1, 1))([encode, attention_weights])
    pre_activation = Concatenate(axis=1)([context_vector, h_t])
    attention_vector = Dense(feat_num, use_bias=False, activation='tanh')(pre_activation)
    model = Model(inputs=inputs, outputs=attention_vector)
    return model

def atten_selfMH(feat_num, C):
    inputs = Input((C, feat_num)) 
    head_1 = ScaledDotProductAttention()(inputs)   
    head_2 = ScaledDotProductAttention()(inputs)
    head_3 = ScaledDotProductAttention()(inputs)
    multi_head = Concatenate(axis=2)([head_1, head_2, head_3])
    multi_head = Dense(feat_num, activation='relu')(multi_head)
    residule_out = Add()([inputs, multi_head])
    residule_out = LayerNormalization()(residule_out)
    attention_vector = mean()(residule_out)
    model = Model(inputs=inputs, outputs=attention_vector)
    return model

def output_net(feat_num):
    inputs = Input((feat_num,))
    outputs = Dense(units=feat_num, activation='relu')(inputs)
    outputs = Dense(units=1, activation='linear')(outputs)    
    model = Model(inputs=inputs, outputs=outputs)
    return model
