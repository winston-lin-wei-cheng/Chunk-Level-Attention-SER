#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import os
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
from utils import CombineListToMatrix


if __name__=='__main__': 
    
    # checking/creating output directory
    if os.path.exists('./NormTerm'):
        pass
    else:    
        os.mkdir('./NormTerm/')    

    # Get Label-Table & Data-Feature
    label_table = pd.read_csv('/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv')
    data_root = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
    
    # Get Desired Attributes from Label-Table
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_set = (label_table['Split_Set'].values).astype('str')
    emo_act = label_table['EmoAct'].values
    emo_dom = label_table['EmoDom'].values
    emo_val = label_table['EmoVal'].values

    # Acoustic-Feature/Label Normalization Parameters based on Training Set
    Train_Data = []
    Train_Label_act = []
    Train_Label_dom = []
    Train_Label_val = []
    for i in range(len(whole_fnames)):
        if split_set[i]=='Train':
            data = loadmat(data_root + whole_fnames[i].replace('.wav','.mat'))['Audio_data']
            data = data[:,1:]  # remove time-info
            Train_Data.append(data)
            Train_Label_act.append(emo_act[i])
            Train_Label_dom.append(emo_dom[i])
            Train_Label_val.append(emo_val[i])
    Train_Data = CombineListToMatrix(Train_Data)
    Train_Label_act = np.array(Train_Label_act)
    Train_Label_dom = np.array(Train_Label_dom)
    Train_Label_val = np.array(Train_Label_val)
    
    # Feature Normalization Parameters
    Feat_mean = np.mean(Train_Data,axis=0)
    Feat_std = np.std(Train_Data,axis=0)       
    savemat('./NormTerm/feat_norm_means.mat', {'normal_para':Feat_mean})
    savemat('./NormTerm/feat_norm_stds.mat', {'normal_para':Feat_std})
    Label_mean_Act = np.mean(Train_Label_act)
    Label_std_Act = np.std(Train_Label_act)
    savemat('./NormTerm/act_norm_means.mat', {'normal_para':Label_mean_Act})
    savemat('./NormTerm/act_norm_stds.mat', {'normal_para':Label_std_Act})    
    Label_mean_Dom = np.mean(Train_Label_dom)
    Label_std_Dom = np.std(Train_Label_dom)    
    savemat('./NormTerm/dom_norm_means.mat', {'normal_para':Label_mean_Dom})
    savemat('./NormTerm/dom_norm_stds.mat', {'normal_para':Label_std_Dom})    
    Label_mean_Val = np.mean(Train_Label_val)
    Label_std_Val = np.std(Train_Label_val)      
    savemat('./NormTerm/val_norm_means.mat', {'normal_para':Label_mean_Val})
    savemat('./NormTerm/val_norm_stds.mat', {'normal_para':Label_std_Val})    
            