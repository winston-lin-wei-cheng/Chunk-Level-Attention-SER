#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
import os
from keras import optimizers
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input, Multiply, Concatenate
import random
import matplotlib.pyplot as plt
from dataloader import DataGenerator_LLD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from model_utils import crop, reshape, mean, repeat
from model_utils import atten_gated, atten_rnn, atten_selfMH, output_net
from transformer import ScaledDotProductAttention, LayerNormalization
from utils import cc_coef
import time
import argparse
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
###############################################################################


# Attention on LSTM chunk output => Weighted Mean of the gated-Attention model
def UttrAtten_GatedVec(atten):
    time_step = 62    # same as the number of frames within a chunk (i.e., m)
    feat_num = 130    # number of LLDs features
    chunk_num = 11    # number of chunks splitted for a sentence (i.e., C)
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    # Uttr Attention Layer
    batch_atten_out = []
    for uttr_idx in range(0, batch_size*chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx+chunk_num
        encode_crop = crop(0, _start, _end)(encode)
        encode_crop = reshape()(encode_crop)
        atten_weights = atten(encode_crop)
        atten_out = Multiply()([encode_crop, atten_weights])
        atten_out = mean()(atten_out)
        batch_atten_out.append(atten_out)
    # Output-Layer
    concat_atten_out= Concatenate(axis=0)(batch_atten_out)
    outputs = output_net(feat_num)(concat_atten_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Attention on LSTM chunk output => RNN-Attention/MultiHead(MH)-Self Attention 
def UttrAtten_AttenVec(atten):
    time_step = 62    # same as the number of frames within a chunk (i.e., m)
    feat_num = 130    # number of LLDs features
    chunk_num = 11    # number of chunks splitted for a sentence (i.e., C)
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    # Uttr Attention Layer
    batch_atten_out = [] 
    for uttr_idx in range(0, batch_size*chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx+chunk_num
        encode_crop = crop(0, _start, _end)(encode)
        encode_crop = reshape()(encode_crop)
        atten_out = atten(encode_crop)
        batch_atten_out.append(atten_out)
    # Output-Layer
    concat_atten_out= Concatenate(axis=0)(batch_atten_out)
    outputs = output_net(feat_num)(concat_atten_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)    
    return model

# Attention on LSTM chunk output => directly average without Attention
def UttrAtten_NonAtten():
    time_step = 62    # same as the number of frames within a chunk (i.e., m)
    feat_num = 130    # number of LLDs features
    chunk_num = 11    # number of chunks splitted for a sentence (i.e., C)
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    # Uttr Attention Layer
    batch_out = []
    for uttr_idx in range(0, batch_size*chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx+chunk_num
        encode_crop = crop(0, _start, _end)(encode)
        encode_crop = reshape()(encode_crop)
        encode_out = mean()(encode_crop)
        batch_out.append(encode_out)
    # Output-Layer
    concat_out= Concatenate(axis=0)(batch_out)
    outputs = output_net(feat_num)(concat_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)    
    return model
###############################################################################

argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-atten", "--atten_type", required=True)
args = vars(argparse.parse_args())

# Parameters
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']
atten_type = args['atten_type']


# Paths Setting
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'

params_train = {'batch_size': batch_size,
                'split_set': 'Train',
                'emo_attr': emo_attr,
                'shuffle': True}

params_valid = {'batch_size': batch_size,
                'split_set': 'Validation', 
                'emo_attr': emo_attr,   
                'shuffle': False}

# Generators
training_generator = DataGenerator_LLD(root_dir, label_dir, **params_train)
validation_generator = DataGenerator_LLD(root_dir, label_dir, **params_valid)

# Optimizer
adam = optimizers.Adam(lr=0.0001)

# Model Saving Settings 
if os.path.exists('./Models'):
    pass
else:    
    os.mkdir('./Models/')
filepath='./Models/LSTM_model[epoch'+str(epochs)+'-batch'+str(batch_size)+']_'+atten_type+'_'+emo_attr+'.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
time_callback = TimeHistory()
callbacks_list = [checkpoint, time_callback]

# Model Architecture
if atten_type == 'GatedVec':
    model = UttrAtten_GatedVec(atten_gated(feat_num=130, C=11))
elif atten_type == 'RnnAttenVec':    
    model = UttrAtten_AttenVec(atten_rnn(feat_num=130, C=11))
elif atten_type == 'SelfAttenVec':    
    model = UttrAtten_AttenVec(atten_selfMH(feat_num=130, C=11))
elif atten_type == 'NonAtten':
    model = UttrAtten_NonAtten()
#print(model.summary())

# Model Compile Settings
model.compile(optimizer=adam, loss=cc_coef)
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=12,
                    epochs=epochs, 
                    verbose=1,
                    callbacks=callbacks_list)        

# Show training & validation loss
v_loss = model.history.history['val_loss']
t_loss = model.history.history['loss']
plt.plot(t_loss,'b')
plt.plot(v_loss,'r')
plt.savefig('./Models/LSTM_model[epoch'+str(epochs)+'-batch'+str(batch_size)+']_'+atten_type+'_'+emo_attr+'.png')
# Record training time cost per epoch
print('Epochs: '+str(epochs)+', ')
print('Batch_size: '+str(batch_size)+', ')
print('Emotion: '+emo_attr+', ')
print('Chunk_type: dynamicOverlap, ')
print('Model_type: LSTM, ')
print('Atten_type: '+atten_type+', ')
print('Avg. Training Time(s/epoch): '+str(np.mean(time_callback.times))+', ')
print('Std. Training Time(s/epoch): '+str(np.std(time_callback.times)))

####### Saving Model Weights/Bias seperately due to different info-flow in the testing stage
model = None # clean gpu-memory
if atten_type=='SelfAttenVec':
    best_model = load_model(filepath, custom_objects={'cc_coef':cc_coef,
                                                      'ScaledDotProductAttention':ScaledDotProductAttention,
                                                      'LayerNormalization':LayerNormalization})
else:
    best_model = load_model(filepath, custom_objects={'cc_coef':cc_coef})

# Saving trained model weights only
best_model.save_weights('./Models/LSTM_model[epoch'+str(epochs)+'-batch'+str(batch_size)+']_'+atten_type+'_'+emo_attr+'_weights.h5')

