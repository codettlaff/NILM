# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:52:36 2026

@author: codett
"""

import os
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide Warnings

from scipy.io import loadmat

import time
import platform
import psutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'Data')
results_dir = os.path.join(base_dir, 'Results')

ukdale_folderpath = os.path.join(data_dir, 'ukdale')
processed_data_folderpath = os.path.join(data_dir, 'ukdale_processed')
model_save_folderpath = os.path.join(results_dir, 'nilm_cnn_model')

T_limit = 86400 # Two Months
train_test_val_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 1
epochs = 20
batch_size = 32

def load_data(ukdale_filepath, T_limit):
    
    data = loadmat(ukdale_filepath)
    print(data.keys())
    
if __name__ == '__main__':
    
    ukdale_filepath = os.path.join(ukdale_folderpath, 'ukdale1.mat')
    load_data(ukdale_filepath, T_limit)
    print('')
    