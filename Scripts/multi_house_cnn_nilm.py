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

# Change Model
# Q data is not available for UK-DALE
# Want to make use of temporal (time-of-use) data.
# Branch 1 (CNN): P-only image (1D image)
# Branch 2 (MPL): Temporal Features
# Concatenate CNN features and temporal embedding beofre final dense layers.
# Multi-Branch Architecture easy to extend later with additional inputs such as weather.

def load_data(ukdale_filepath, T_limit=None):
    
    data = loadmat(ukdale_filepath)
    inputs = data['input']
    outputs = data['output']
    
    time_seconds = inputs[:, 0]
    time_of_day = (time_seconds % 86400) / 3600.0 # Convert to hour of day.
    
    P_agg = inputs[:, 1]
    X = np.column_stack((time_of_day, P_agg)).astype(np.float32)
    
    Y = outputs[:, 2:]
    appliance_names = [str(name[0]) for name in data["labelOut"].squeeze()[2:]]
    if T_limit is not None:
        X = X[:T_limit]
        Y = Y[:T_limit]
    
    return {
        'X': X,
        'Y': Y,
        'appliance_names': appliance_names}

def filter_by_appliance(data, target_appliances):
    
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
    
    # Training Indices
    train_inp, train_out = idx_dict['train']
    train_X = data['X'][train_inp]
    train_Y = data['Y'][train_out]
    
    # Convert hour-of-day to cyclic features
    hour = data['X'][:, 0]
    theta = 2 * np.pi * hour / 24.0
    sin_hour = np.sin(theta).astype(np.float32)
    cos_hour = np.cos(theta).astype(np.float32)
    
    # 00:00 → ( 0, 1)
    # 06:00 → ( 1, 0)
    # 12:00 → ( 0,-1)
    # 18:00 → (-1, 0)
    # 24:00 → ( 0, 1)
    
    # Normalize aggregate power using training data only
    p_min = train_X[:, 1].min()
    p_max = train_X[:, 1].max()
    p_range = max(p_max - p_min, 1e-12)
    p_agg = (data['x'][:, 1] - p_min) / p_range
    
    # Normalize appliance powers
    y_min = train_Y.min(axis=0)
    y_max = train_Y.max(axis=0)
    y_range = np.maximum(y_max - y_min, 1e-12)
    Y = (data['Y'] - y_min) / y_range
    
    data_normalized['X'] = np.column_stack((sin_hour, cos_hour, p_agg)).astype(np.float32)
    data_normalized['Y'] = Y.astype(np.float32)
    
    scaling_factors = {
        'x_min': p_min,
        'x_max': p_max,
        'y_min': y_min,
        'y_max': y_max}
    
    data_normalized['scaling_factors'] = scaling_factors
    return data_normalized

def process_window(x_win):
    
    x_win = np.asarray(x_win, dtype=np.float32)
    p_seq = x_win[:, 2:3]
    center = center = len(x_win) //  2
    time_features = x_win[center, 0:2]
    return p_seq, time_features

def generate_sample(x_data, y_data, i_inp, i_out, window_length):
    
    x_win = x_data[i_inp : i_inp + window_length]
    if x_win.shape[0] != window_length: return None
    
    p_seq, time_features = process_window(x_win)
    center = i_out + window_length // 2
    y_target = y_data[center]
    
    return (p_seq, time_features), y_target

def prepare_data(data, idx_dict, window_length, stride, save_filepath):
    
    data = normalize_data(data, idx_dict)
    x_data = data['X']
    y_data = data['Y']
    scaling = data['scaling_factors']
    
    n_appliances = y_data.shape[1]
    filepaths = {}
    process = psutil.Process(os.getpid())
    
    for split in ['train', 'val', 'test']:
        
        split_start = time.perf_counter()
        inp_idx, out_idx = idx_dict[split]
        n_samples = len(inp_idx)
        
        # Allocate arrays
        X_p = np.empty((n_samples, window_length, 1), dtype=np.float32)
        X_time = np.empty((n_samples, 2), dtype=np.float32)
        Y_p = np.empty((n_samples, n_appliances), dtype=np.float32)
        
        # Generate samples
        for j, (i_inp, i_out) in enumerate(zip(inp_idx, out_idx)):
            (p_seq, time_features), y = generate_sample(x_data, y_data, i_inp, i_out, window_length)
            X_p[j] = p_seq
            X_time[j] = time_features
            Y_p[j] = y
            
        split_time = time.perf_counter() - split_start
        
        # Output directory
        split_dir = f"{save_filepath}_{split}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(split_dir, 'X_p.npy'), X_p)
        np.save(os.path.join(split_dir, 'X_time.npy'), X_time)
        np.save(os.path.join(split_dir, 'Y_p.npy'), Y_p)
        
        metadata = {} # Add this later
        metadata['normalization'] = scaling
        metadata_filepath = os.path.join(split_dir, 'metadata.pkl')
        with open(metadata_filepath, 'wb') as f: pickle.dump(metadata, f)
        
        filepaths[split] = {
            'X_p': os.path.join(split_dir, 'X_p.npy'),
            'X_time': os.path.join(split_dir, 'X_time.npy'),
            'Y_p': os.path.join(split_dir, 'Y_p.npy'),
            'metadata': metadata_filepath}

    return filepaths

def load_processed_data(processed_data_filepaths_dict):
    
    with open(processed_data_filepaths_dict['metadata'], 'rb') as f: metadata = pickle.load(f)
    return {
        'X_p': np.load(processed_data_filepaths_dict['X_p'], mmap_mode='r'),
        'X_time': np.load(processed_data_filepaths_dict['X_time'], mmap_mode='r'),
        'Y_p': np.load(processed_data_filepaths_dict['Y_p'], mmap_mode='r'),
        'normalization': metadata['normalization']}
    
def generate_batch(processed_data, idx_list):
    
    X_p_batch = processed_data['X_p'][idx_list]
    X_time_batch = processed_data['X_time'][idx_list]
    Y_p_batch = processed_data['Y_p'][idx_list]
    return (X_p_batch, X_time_batch), Y_p_batch

def build_model(window_length):
    
    # CNN branch (aggregate power sequence)
    inp_power = layers.Input(shape=(window_length,1), name='power_input')
    
    x1 = layers.Conv1D(32, 5, activation='relu', padding='same')(inp_power)
    x1 = layers.Conv1D(64, 5, activation='relu', padding='same')(x1)
    x1 = layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # MLP branch (time features)
    inp_time = layers.Input(shape=(2,), name='time_input')
    x2 = layers.Dense(16, activation='relu')(inp_time)
    x2 = layers.Dense(64, activation='relu')(x2)
    
    # Concatenation
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    out = layers.Dense(1, name='power_output')(x)
    model = models.Model(inputs=[inp_power, inp_time], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), los='mse')
    
    return model

def train_model(model, processed_training_data, processed_val_data, epochs, batch_size, model_filepath):
    
    training_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    
    best_val_loss = np.inf
    best_epoch = 0
    patience = 5 
    patience_counter = 0 
    
    n_train = len(processed_training_data['Y_p'])
    n_val = len(processed_val_data['Y_p'])
    
    epoch_times = []
    train_losses = []
    val_losses = []
    
    epochs_completed = 0
    
    for epoch in tqdm(range(epochs), esc='Epochs'):
        
        epoch_start = time.perf_counter()
        
        # Training
        train_loss = 0.0 
        num_train_batches = 0 
        perm = np.random_permutation(n_train) 
        
        for i in tqdm(range(0, n_train, batch_size), desc='Training', leave=False):
            
            batch_idx = perm[i:i + batch_size]
            (X_p, X_time), Y_p = generate_batch(processed_training_data, batch_idx)
            loss = model.train_on_batch([X_p, X_time], Y_p)
            train_loss += loss
            num_train_batches += 1
            peak_ram = max(peak_ram, process.memory_info().rss)
            
        train_loss /= num_train_batches
        
        # Validation
        val_loss = 0.0 
        num_val_batches = 0
        
        for i in tqdm(range(0, n_val, batch_size), desc='Validation', leave=False):
            
            batch_idx = np.arange(i, min(i + batch_size, n_val))
            (X_p, X_time), Y_p = generate_batch(processed_val_data, batch_idx)
            loss = model.test_on_batch([X_p, X_time], Y_p)
            val_loss += loss
            num_val_batches += 1
            
        val_loss /= num_val_batches
        epoch_time = time.perf_counter() - epoch_start
        
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_completed += 1 
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0 
            model.save(model_filepath)
            
        else:
            patience_counter += 1
            if patience_counter >= patience: break
        
    total_training_time = (time.perf_counter - training_start)
    model.save(model_filepath)
    
    training_metadata = {} # Add this later
    metadata_filepath = (os.path.splitext(model_filepath)[0] + '_metadata.pkl')
    with open(metadata_filepath, 'wb') as f: pickle.dump(training_metadata, f)
    
    return model_filepath, metadata_filepath
    
if __name__ == '__main__':
    
    ukdale_filepath = os.path.join(ukdale_folderpath, 'ukdale1.mat')
    load_data(ukdale_filepath, T_limit)
    print('')
    