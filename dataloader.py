#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
from scipy.io import loadmat
import keras
import random
from utils import getPaths, DynamicChunkSplitTrainingData
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99

class DataGenerator_LLD(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_dir, label_dir, batch_size, split_set, emo_attr, shuffle=True):
        'Initialization'
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.split_set = split_set                        # 'Train' or 'Validation'
        self.emo_attr = emo_attr                          # 'Act', 'Dom' or 'Val'
        self.shuffle = shuffle
        # Loading Norm-Feature Parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        # Loading Norm-Label Parameters
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Dom':    
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]         
        # Loading Data Paths/Labels
        self._paths, self._labels = getPaths(label_dir, split_set, emo_attr)        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(getPaths(self.label_dir, self.split_set, self.emo_attr)[0])/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find Batch list of Loading Paths
        list_paths_temp = [self._paths[k] for k in indexes]
        list_labels_temp = [self._labels[k] for k in indexes]       
        
        # Generate data
        data, label = self.__data_generation(list_paths_temp, list_labels_temp)
        return data, label        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        _paths, _labels = getPaths(self.label_dir, self.split_set, self.emo_attr)
        self.indexes = np.arange(len(_paths))
        if self.shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp, list_labels_temp):
        'Generates data containing batch_size with fixed chunck samples'           
        batch_x = []
        batch_y = []
        for i in range(len(list_paths_temp)):
            # Store Norm-Data
            x = loadmat(self.root_dir + list_paths_temp[i].replace('.wav','.mat'))['Audio_data']
            # we use the Interspeech 2013 computational paralinguistics challenge LLDs feature set
            # which includes totally 130 features (i.e., the "IS13_ComParE" configuration)
            x = x[:,1:]                                     # remove time-info from the extracted OpenSmile LLDs
            x = (x-self.Feat_mean)/self.Feat_std            # LLDs feature normalization (z-norm)
            # Bounded NormFeat Ranging from -3~3 and assign NaN to 0
            x[np.isnan(x)]=0
            x[x>3]=3
            x[x<-3]=-3            
            # Store Norm-Label
            y = (list_labels_temp[i]-self.Label_mean)/self.Label_std
            batch_x.append(x)
            batch_y.append(y)

        # split sentences into fixed length and fixed number of small chunks
        batch_chunck_x, batch_chunck_y = DynamicChunkSplitTrainingData(batch_x, batch_y, m=62, C=11, n=1)
        return batch_chunck_x, batch_chunck_y
