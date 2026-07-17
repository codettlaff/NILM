# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:07:59 2026

@author: codett
"""

import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide Warnings

base_dir = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base_dir, 'Data')
results_dir = os.path.join(base_dir, 'Results')
ampds_filepath = os.path.join(data_dir, 'ampds2.npz')
processed_data = os.path.join(data_dir, 'ampds2_processed')
model_save_filepath = os.path.join(results_dir, 'nilm_cnn_model_2month.keras')

T_limit = 86400 # Two Months
train_test_val_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 1
epochs = 20
batch_size = 32
target_appliances = ['DWE']

def load_data(ampds_filepath, T_limit, target_appliances=None):
    
    data = np.load(ampds_filepath)
    X, Y = data['X'], data['Y'] 
    appliance_names = data['out_labels']
    T = X.shape[0]
    T_limit = min(T, T_limit) if T_limit is not None else T
    X, Y = X[:T_limit], Y[:T_limit]
    
    X = X[:, [0,2]] # Keep only P and Q
    Y = Y[:,:,0] # Keep only P
    
    if not target_appliances: target_appliances = appliance_names
    indices = [i for i, name in enumerate(appliance_names) if name in target_appliances]
    appliance_names = appliance_names[indices]
    Y = Y[:,indices]
    
    return {
        'X': X,
        'Y': Y,
        'T': T,
        'appliance_names': appliance_names}

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
    # data_normalized['X'] = data['X'].copy()
    # data_normalized['Y'] = data['Y'].copy()
    
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
    
    data_normalized['x_min'] = x_min
    data_normalized['x_max'] = x_max
    data_normalized['y_min'] = y_min
    data_normalized['y_max'] = y_max
    
    return data_normalized

def process_window(x_win):
    x_win = np.asarray(x_win, dtype=np.float32)
    p, q = x_win[:, 0], x_win[:, 1]
    
    # Build PQ Signature
    p_col = p[:, np.newaxis] # (W, 1)
    q_row = q[np.newaxis, :] # (1, W)
    S_xy = np.sqrt(p_col**2, q_row**2, dtype=np.float32)
    top = np.concatenate([np.zeros((1,1), dtype=np.float32), q_row], axis=1)
    left = np.concatenate([p_col, S_xy], axis=1)
    S = np.concatenate([top, left], axis=0)
    S = S[..., np.newaxis] # (W+1, W+1, 1)
    
    # Build FFT
    S_fft = np.abs(np.fft.fft2(S[:, :, 0])).astype(np.float32)
    S_fft = S_fft[..., np.newaxis]
    
    return S, S_fft

def process_window_old(x_win):
    x_win = tf.convert_to_tensor(x_win, dtype=tf.float32)
    p, q = x_win[:, 0], x_win[:, 1]

    # Build PQ Signature
    p_col, q_row = tf.reshape(p, (-1,1)), tf.reshape(q, (1, -1))
    S_xy = tf.sqrt(p_col**2 + q_row**2) # Compute pairwise S_xy, producing (W,W) matrix.
    top = tf.concat([tf.zeros((1,1)), q_row], axis=1) # Build first row (q only)
    left = tf.concat([p_col, S_xy], axis=1) # Build remaining rows. First column is p only.
    S = tf.concat([top, left], axis=0) # Combine into PQ image.
    S = tf.expand_dims(S, -1) # Add Channel Dimension.

    # Build FFT
    S_fft = tf.signal.fft2d(tf.cast(S[:,:,0], tf.complex64)) # Compute 2D Fourier transform of PQ image.
    S_fft = tf.abs(S_fft) # Keep magnitude only.
    S_fft = tf.expand_dims(S_fft, -1) # Add Channel Dimension

    return (S, S_fft)

def generate_sample(x_data, y_data, i_inp, i_out, window_length):

    x_window = x_data[i_inp : i_inp + window_length] # (W,2)
    if x_window.shape[0] != window_length: return None
    y_target = y_data[i_out]
    S, S_fft = process_window(x_window)
    return (S, S_fft), y_target

def prepare_data(data, idx_dict, window_length, save_filepath):
    
    # Normalize using training statistics
    data = normalize_data(data, idx_dict)
    x_data = data['X']
    y_data = data['Y']
    scaling = {
        'x_min': data['x_min'],
        'x_max': data['x_max'],
        'y_min': data['y_min'],
        'y_max': data['y_max']}
    image_size = window_length + 1
    n_appliances = y_data.shape[1]
    
    for split in ['train', 'val', 'test']:
        inp_idx, out_idx = idx_dict[split]
        n_samples = len(inp_idx)
        
        # Preallocate Arrays
        S = np.empty((n_samples, image_size, image_size), dtype=np.float32)
        FFT = np.empty((n_samples, image_size, image_size, 1),dtype=np.float32)
        Y = np.empty((n_samples, n_appliances), dtype=np.float32)
        
        # Fill arrays in place
        for j, (i_inp, i_out) in enumerate(zip(inp_idx, out_idx)):
            
            (s, s_fft), y = generate_sample(x_data, y_data, i_inp, i_out, window_length)
            S[j] = s.numpy()
            FFT[j] = s_fft.numpy
            Y[j] = y
            
        np.save(f"{save_filepath}_{split}.npz", S=S, FFT=FFT, Y=Y, **scaling)
        
# Reminaing bottleneck is generate_sample()
# Converts every window to a TensorFlow tensors individually
# Best to rewrite process_window using pure Numpy instead of TensorFlow
# Could make preprocessing 2-5x faster and use less memory.
        
def get_sample(processed_data, idx):
    pass

# Data is only loaded for a short time when batch is being fetched.
# Saves memory.
def generate_batch(processed_data_filepath, idx_list):
    pass

def build_model():
    pass

def train_model(model, processed_data_filepath, processed_data_idx_dict, epochs, batch_size, model_filepath):
    pass

def test_model(model_filepath, processed_data_filepath, processed_data_idx_dict, batch_size, scaling_factors, show=False):
    pass

if __name__ == '__main__':
    pass

