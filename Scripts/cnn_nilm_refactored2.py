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

# Environment Information

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

def data_metadata(name, input_labels, output_labels, window_length, stride, normalization_factors, num_chunks, total_timesteps, num_samples, processing_time, processing_peak_ram, processing_env, train_test_val_split, train_split, val_split, test_split):
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
        'num_chunks': num_chunks,
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

def test_results(mse_norm, rmse_norm, mse_denorm, rmse_denorm, mae, eacc, inference_time, peak_ram, env):
    return {
        'mse_norm': mse_norm,
        'rmse_norm': rmse_norm,
        'mse': mse_denorm,
        'rmse': rmse_denorm,
        'mae': mae,
        'eacc': eacc,
        'inference_time': inference_time,
        'inference_peak_RAM_MB': peak_ram / 1024**2,
        'inference_environment': env}

def model_metadata(name, model, train_time_seconds, epochs_requested, epochs_completed, batch_size, train_loss, val_loss, train_peak_ram, size, test_results=None):
    metadata = {
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
    if test_results: metadata['test_results'] = test_results
    return metadata

# Helper Functions

def read_pickle(filepath):
    with open(filepath, 'rb') as f: data = pickle.load(f)
    return data

def write_pickle(data, filepath):
    with open(filepath, 'wb') as f: pickle.dump(data, f)
    
# Pre-process Data

def load_data(ampds_filepath, T_limit):
    
    data = np.load(ampds_filepath)
    X, Y = data['X'], data['Y'] 
    appliance_names = data['out_labels']
    T = X.shape[0]
    T_limit = min(T, T_limit) if T_limit is not None else T
    X, Y = X[:T_limit], Y[:T_limit]
    
    X = X[:, [0,2]] # Keep only P and Q
    Y = Y[:,:,0] # Keep only P
    
    return {
        'X': X,
        'Y': Y,
        'T': T_limit,
        'appliance_names': appliance_names}

def filter_by_appliances(data, target_appliances):
    
    target_data = data.copy()
    appliance_names = data['appliance_names']
    indices = [i for i, name in enumerate(appliance_names) if name in target_appliances]
    appliance_names = appliance_names[indices]
    target_data['appliance_names'] = appliance_names
    target_data['Y'] = target_data['Y'][:,indices]
    return target_data

def precompute_indices(num_timesteps, window_length, stride, train_val_test_split, number_blocks, seed=42):
    
    center_offset = window_length // 2
    guard = window_length - 1
    guard_left = guard // 2 
    guard_right = guard - guard_left
    rng = np.random.default_rng(seed)
    
    # Number of blocks in each split
    n_train_blocks = int(round(train_val_test_split[0] * number_blocks))
    n_val_blocks = int(round(train_val_test_split[1] * number_blocks))
    n_test_blocks = number_blocks - n_train_blocks - n_val_blocks
    
    # Timesteps in each split
    n_train = int(train_val_test_split[0] * num_timesteps)
    n_val = int(train_val_test_split[1] * num_timesteps)
    n_test = num_timesteps - n_train - n_val
    
    # Divide timesteps over given number of blocks
    def make_lengths(total_length, n_blocks):
        lengths = np.full(n_blocks, total_length // n_blocks, dtype=int)
        lengths[:total_length % n_blocks] += 1
        return lengths
    
    block_lengths = np.concatenate([
        make_lengths(n_train, n_train_blocks),
        make_lengths(n_val, n_val_blocks),
        make_lengths(n_test, n_test_blocks)])
    
    block_labels = (
        ['train'] * n_train_blocks + 
        ['val'] * n_val_blocks +
        ['test'] * n_test_blocks)
    
    # Randomly assign blocks
    perm = rng.permutation(number_blocks)
    block_lengths = block_lengths[perm]
    block_labels = [block_labels[i] for i in perm]
    
    # Compute block boundaries
    starts = np.zeros(number_blocks, dtype=int)
    ends = np.zeros(number_blocks, dtype=int)
    start=0
    for i, length in enumerate(block_lengths):
        starts[i] = start
        end = min(start + length, num_timesteps)
        ends[i] = end
        start = end
        
    split = {
        'train': ([],[]),
        'val': ([],[]),
        'test': ([],[])}
    
    for i in range(number_blocks):
        start = starts[i]
        end = ends[i]
        
        # Split guard across both sides of a boundary
        usable_start = start
        usable_end = end
        if i > 0 and block_labels[i] != block_labels[i - 1]: usable_start += guard_right 
        if i < number_blocks - 1 and block_labels[i] != block_labels[i + 1]: usable_end -= guard_left 
        if usable_end >= usable_start + window_length:
            inp = np.arange(usable_start, usable_end - window_length  + 1, stride) 
            out = inp + center_offset 
            split[block_labels[i]][0].append(inp) 
            split[block_labels[i]][1].append(out)
            
    # Recombine windows from multiple blocks and shuffle
    idx_dict = {}
    for label in ['train', 'val', 'test']:
        if split[label][0]:
            inp = np.concatenate(split[label][0])
            out = np.concatenate(split[label][1])
            perm = rng.permutation(len(inp))
            idx_dict[label] = (inp[perm], out[perm])
        else: idx_dict[label] = (np.array([], dtype=int), np.array([], dtype=int))
    return idx_dict

def normalize_data(data, idx_dict):
    
    data_normalized = data.copy()
    
    # Training timesteps
    train_inp, train_out = idx_dict['train']
    train_X = data['X'][train_inp]
    train_Y = data['Y'][train_out]
    
    x_min = train_X.min(axis=0)
    x_max = train_X.max(axis=0)
    y_min = train_Y.min(axis=0)
    y_max = train_Y.max(axis=0)
    
    # Prevent divide-by-zero
    x_range = np.maximum(x_max - x_min, 1e-12)
    y_range = np.maximum(y_max - y_min, 1e-12)
    
    data_normalized['X'] = (data['X'] - x_min) / x_range
    data_normalized['Y'] = (data['Y'] - y_min) / y_range
    
    scaling_factors = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max}
    data_normalized['scaling_factors'] = scaling_factors
    
    return data_normalized

