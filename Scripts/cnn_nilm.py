# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:52:35 2026

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

import time
import platform
import psutil

base_dir = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base_dir, 'Data')
results_dir = os.path.join(base_dir, 'Results')

ampds_filepath = os.path.join(data_dir, 'ampds2.npz')
processed_data_folderpath = os.path.join(data_dir, 'ampds2_processed')
model_save_folderpath = os.path.join(results_dir, 'nilm_cnn_model')

T_limit = 86400 # Two Months
train_test_val_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 1
epochs = 20
batch_size = 32

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

def process_window(x_win):
    x_win = np.asarray(x_win, dtype=np.float32)
    p, q = x_win[:, 0], x_win[:, 1]
    
    # Build PQ Signature
    p_col = p[:, np.newaxis] # (W, 1)
    q_row = q[np.newaxis, :] # (1, W)
    S_xy = np.sqrt(p_col**2 + q_row**2, dtype=np.float32) # This line causing bug
    top = np.concatenate([np.zeros((1,1), dtype=np.float32), q_row], axis=1)
    left = np.concatenate([p_col, S_xy], axis=1)
    S = np.concatenate([top, left], axis=0)
    S = S[..., np.newaxis] # (W+1, W+1, 1)
    
    # Build FFT
    S_fft = np.abs(np.fft.fft2(S[:, :, 0])).astype(np.float32)
    S_fft = S_fft[..., np.newaxis]
    
    return S, S_fft

def generate_sample(x_data, y_data, i_inp, i_out, window_length):

    x_window = x_data[i_inp : i_inp + window_length]
    if x_window.shape[0] != window_length: return None
    y_target = y_data[i_out] 
    S, S_fft = process_window(x_window)
    return (S, S_fft), y_target

def prepare_data(data, idx_dict, window_length, stride, save_filepath):

    # Normalize using training statistics
    data = normalize_data(data, idx_dict)
    x_data = data['X']
    y_data = data['Y']
    scaling = data['scaling_factors']
    image_size = window_length + 1
    n_appliances = y_data.shape[1]
    filepaths = {}
    
    process = psutil.Process(os.getpid())
    
    for split in ['train', 'val', 'test']:
        split_start = time.perf_counter()
        inp_idx, out_idx = idx_dict[split]
        n_samples = len(inp_idx)
        
        # Allocate Arrays
        S = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)
        FFT = np.empty_like(S)
        Y = np.empty((n_samples, n_appliances), dtype=np.float32)
        
        # Generate Samples
        for j, (i_inp, i_out) in enumerate(zip(inp_idx, out_idx)):
            (s, s_fft), y = generate_sample(x_data, y_data, i_inp, i_out, window_length)
            S[j] = s
            FFT[j] = s_fft
            Y[j] = y
            
        split_time = time.perf_counter() - split_start
        
        # Output Directory
        split_dir = f"{save_filepath}_{split}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Save Arrays
        np.save(os.path.join(split_dir, "S.npy"), S)
        np.save(os.path.join(split_dir, "FFT.npy"), FFT)
        np.save(os.path.join(split_dir, "Y.npy"), Y)
        
        # Dataset Metadata
        metadata = {
            # Dataset
            "split": split,
            "dataset": {
                "num_timesteps": int(data["T"]),
                "num_samples": int(n_samples),
                "target_appliances": list(data["appliance_names"])},
            "preprocessing":{
                "window_length": window_length,
                "stride": stride},
            "dimensions": {
                "input_shape_time": S.shape[1:],
                "input_shape_fft": FFT.shape[1:],
                "output_shape": Y.shape[1:]},
            "normalization": {
                "x_min": scaling["x_min"],
                "x_max": scaling["x_max"],
                "y_min": scaling["y_min"],
                "y_max": scaling["y_max"]},
            "timing": {
                "preprocessing_seconds": split_time,
                "samples_per_second": (n_samples / split_time if split_time > 0 else None)},
            "computation": {
                "S_size_MB":
                    S.nbytes / 1024**2,
                "FFT_size_MB":
                    FFT.nbytes / 1024**2,
                "Y_size_MB":
                    Y.nbytes / 1024**2,
                "total_dataset_size_MB":
                    (S.nbytes + FFT.nbytes + Y.nbytes) / 1024**2,
                "process_RAM_MB":
                    process.memory_info().rss / 1024**2},
            "environment": {
                "python_version": platform.python_version(),
                "tensorflow_version": tf.__version__,
                "os": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "total_RAM_GB":
                    psutil.virtual_memory().total / 1024**3}}
            
        metadata_filepath = os.path.join(split_dir, "metadata.pkl")
        with open(metadata_filepath, "wb") as f: pickle.dump(metadata, f)
        
        filepaths[split] = {
            "S": os.path.join(split_dir, "S.npy"),
            "FFT": os.path.join(split_dir, "FFT.npy"),
            "Y": os.path.join(split_dir, "Y.npy"),
            "metadata": metadata_filepath}
        
        return filepaths

def load_processed_data(processed_data_filepaths_dict):

    with np.load(processed_data_filepaths_dict["scaling"]) as scaling:
        scaling_factors = {
            "x_min": scaling["x_min"],
            "x_max": scaling["x_max"],
            "y_min": scaling["y_min"],
            "y_max": scaling["y_max"]}

    return {
        "S": np.load(processed_data_filepaths_dict["S"], mmap_mode="r"),
        "FFT": np.load(processed_data_filepaths_dict["FFT"], mmap_mode="r"),
        "Y": np.load(processed_data_filepaths_dict["Y"], mmap_mode="r"),
        "scaling_factors": scaling_factors}

