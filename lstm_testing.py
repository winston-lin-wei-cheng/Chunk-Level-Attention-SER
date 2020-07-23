#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input, Multiply
import random
import time
from utils import getPaths, DynamicChunkSplitTestingData, evaluation_metrics
from scipy.io import loadmat
from model_utils import mean, reshape
from model_utils import atten_gated, atten_rnn, atten_selfMH, output_net
import argparse
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99


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


# Data/Label/Model Dir
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
#model_path = './Models/LSTM_model[epoch'+str(epochs)+'-batch'+str(batch_size)+']_'+atten_type+'_'+emo_attr+'_weights.h5'
model_path = './trained_model_v1.6/LSTM_model[epoch'+str(epochs)+'-batch'+str(batch_size)+']_'+atten_type+'_'+emo_attr+'_weights.h5'

# Loading Norm-Parameters
Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']
if emo_attr == 'Act':
    Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Dom':
    Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Val':
    Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0] 

# Regression Task
test_file_path, test_file_tar = getPaths(label_dir, split_set='Test', emo_attr=emo_attr)
#test_file_path, test_file_tar = getPaths(label_dir, split_set='Validation', emo_attr=emo_attr)

# Setting Online Prediction Model Graph (predict sentence by sentence rather than a data batch)
time_step = 62        # same as the number of frames within a chunk (i.e., m)
feat_num = 130        # number of LLDs features

if atten_type == 'GatedVec':
    # LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    encode = reshape()(encode)     
    # Attention Layer
    a_weighted = atten_gated(feat_num=130, C=11)(encode)
    attention_vector = Multiply()([encode, a_weighted])
    attention_vector = mean()(attention_vector)  
    # Output Layer
    outputs = output_net(feat_num)(attention_vector) 
    model = Model(inputs=inputs, outputs=outputs)    

elif atten_type == 'RnnAttenVec':
    # LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    encode = reshape()(encode) 
    # Attention Layer
    attention_vector = atten_rnn(feat_num=130, C=11)(encode)
    # Output Layer
    outputs = output_net(feat_num)(attention_vector)      
    model = Model(inputs=inputs, outputs=outputs)   

elif atten_type == 'SelfAttenVec':
    # LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    encode = reshape()(encode) 
    # Attention Layer
    attention_vector = atten_selfMH(feat_num=130, C=11)(encode)
    # Output Layer
    outputs = output_net(feat_num)(attention_vector)   
    model = Model(inputs=inputs, outputs=outputs)  

elif atten_type == 'NonAtten':
    # LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    encode = reshape()(encode) 
    # Output Layer
    avg_vec = mean()(encode)
    outputs = output_net(feat_num)(avg_vec) 
    model = Model(inputs=inputs, outputs=outputs)   

# loading model trained weights
model.load_weights(model_path)
#print(model.summary())

# Split testing set into 15 subsets for statistical T-test
subset_num = 15
indexes = np.arange(len(test_file_path))
np.random.seed(random_seed)
np.random.shuffle(indexes)
test_file_path = test_file_path[indexes]
test_file_tar = test_file_tar[indexes]
subset_idx = np.arange(0, len(test_file_path),int(np.round(len(test_file_path)/subset_num)))
Subset_Idx = []
for i in range(len(subset_idx)):
    try:
        Subset_Idx.append(np.arange(subset_idx[i], subset_idx[i+1]))
    except:
        Subset_Idx.append(np.arange(subset_idx[i], len(test_file_path)))

# Online Testing Process for subsets of the test set
Pred_Rsl = []
Time_Cost = []
for ii in range(len(Subset_Idx)):
    Test_pred = []
    Test_label = []
    Time_cost = []
    for i in Subset_Idx[ii]:  
        data = loadmat(root_dir + test_file_path[i].replace('.wav','.mat'))['Audio_data']
        data = data[:,1:]                           # remove time-info
        data = (data-Feat_mean)/Feat_std            # Feature Normalization
        # Bounded NormFeat Ranging from -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # split sentence into data chunks
        chunk_data = DynamicChunkSplitTestingData([data], m=62, C=11, n=1)
        # Recording prediction time cost
        tic = time.time()    
        pred = model.predict(chunk_data)
        toc = time.time()
        Time_cost.append((toc-tic)*1000) # unit of time = 10^-3
        # Output prediction results    
        pred = np.mean(pred)
        Test_pred.append(pred)
        Test_label.append(test_file_tar[i])
    Test_pred = np.array(Test_pred)
    Test_label = np.array(Test_label)
    # Time Cost Result
    Time_cost = np.array(Time_cost)
    # Regression Task => Prediction & De-Normalize Target
    Test_pred = (Label_std*Test_pred)+Label_mean
    Pred_Rsl.append(evaluation_metrics(Test_label, Test_pred)[0])
    Time_Cost.append(np.mean(Time_cost))
    
# Subset results for Statistic Test
Pred_Rsl = np.array(Pred_Rsl)
Time_Cost = np.array(Time_Cost)
print('Model_type: LSTM')
print('Epochs: '+str(epochs))
print('Batch_size: '+str(batch_size))
print('Emotion: '+emo_attr)
print('Chunk_type: dynamicOverlap')
print('Atten_type: '+atten_type)
print('Avg. CCC testing performance: '+str(np.mean(Pred_Rsl)))
print('Std. CCC testing performance: '+str(np.std(Pred_Rsl)))
print('Avg. Prediction Time(ms/uttr): '+str(np.mean(Time_Cost)))
print('Std. Prediction Time(ms/uttr): '+str(np.std(Time_Cost)))
