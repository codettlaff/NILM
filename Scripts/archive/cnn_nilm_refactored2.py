# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:09:58 2026

@author: codett
"""

import os
import time
import platform
import psutil

from tqdm import tqdm
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from scipy.io import loadmat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directory Dicts

# data_directory_dict
# -- train
# -- -- X_p: filepath
# -- -- X_time: filepath
# -- -- Y: filepath
# -- val
# -- -- X_p: filepath
# -- -- X_time: filepath
# -- -- Y: filepath
# -- test
# -- -- X_p: filepath
# -- -- X_time: filepath
# -- -- Y: filepath
# -- metadata: filepath

# model_directory_dict
# -- model: filepath
# -- metadata: filepath
# -- results_npy: filepath
# -- results_pkl: filepath

# data_set_directory_dict
# -- data_1
# -- -- train: filepath
# -- -- val: filepath
# -- -- test: filepath
# -- data_2
# ...

# model_set_directory_dict
# -- model_1
# -- -- model: filepath
# -- -- metadata: filepath
# -- -- results_npy: filepath
# -- -- results_pkl: filepath
# -- model_2
# ...

# Environment Infor

def environment_info():
    return {
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'os': platform.system(),
        'os_release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'total_RAM_GB': psutil.virtual_memory().total / 1024**3}

# Metadata Dicts

def data_split(num_timesteps, num_samples, size):
    return {
        'num_timesteps': num_timesteps,
        'num_samples': num_samples,
        'size_MB': size / 1024**2}

def normalization_factors(x_min, x_max, y_min, y_max):
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max}

def data_metadata(name, input_labels, output_labels, window_length, stride, normalization_factors, total_timesteps, num_samples, processing_time, processing_peak_ram, processing_env, train_test_val_split, train_split, val_split, test_split):
    timesteps_used = train_split['num_timesteps'] + val_split['num_timesteps'] + test_split['num_timesteps']
    num_samples = train_split['num_samples'] + val_split['num_samples'] + test_split['num_samples']
    size_MB = train_split['size_MB'] + val_split['size_MB'] + test_split['size_MB']
    return {
        'type': 'data',
        'name': name,
        'input_labels': input_labels,
        'output_labels': output_labels,
        'window_length': window_length,
        'stride': stride,
        'normalization_factors': normalization_factors,
        'total_timesteps': total_timesteps,
        'timesteps_used': timesteps_used,
        'timesteps_discarded': total_timesteps - timesteps_used,
        'num_samples': num_samples,
        'processing_time': processing_time,
        'processing_peak_RAM_MB': processing_peak_ram,
        'processing_env': processing_env,
        'size_MB': size_MB,
        'train_test_val_split': train_test_val_split,
        'train_split': train_split,
        'val_split': val_split,
        'test_split': test_split}

def model_metadata(name, model, train_time_seconds, epochs_requested, epochs_completed, batch_size, train_loss, val_loss, train_peak_ram, size):
    return {
        'type': 'model',
        'name': name,
        'num_layers': len(model.layers),
        'num_trainable_layers': sum(not isinstance(layer, tf.keras.layers.InputLayer) for layer in model.layers),
        'layer_sequence': [f'{layer.name}: {layer.__class__.__name__} -> {layer.output.shape}' for layer in model.layers],
        'trainable_parameters': model.count_params(),
        'data_metadata': data_metadata,
        'train_time_seconds': train_time_seconds,
        'epochs_requested': epochs_requested,
        'epochs_completed': epochs_completed,
        'batch_size': batch_size,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_peak_RAM_MB': train_peak_ram,
        'size_MB': size / 1024**2}