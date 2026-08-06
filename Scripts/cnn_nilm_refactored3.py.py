# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:53:13 2026

@author: codett
"""

import os
import time
import platform
import psutil

from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import load_model
from scipy.io import loadmat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Metadata Helpers

def get_environment_info():
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

def get_dataset_split_info(
    num_timesteps: int,
    num_samples: int,
    size_MB: float):
    return {
        'num_timesteps': num_timesteps,
        'num_samples': num_samples,
        'size_MB': size_MB}

def get_dataset_info(
    name:str,
    input_labels: list(str),
    output_labels: list(str),
    window_length: int,
    stride: int,
    normalization_factors: dict[float],
    num_chunks: int,
    processing_env: dict,
    processing_time_seconds: float,
    processing_peak_RAM_MB: float,
    train_val_test_split: tuple[float, float, float],
    splits: tuple[dict, dict, dict]):
    num_timesteps = sum(s['num_timesteps'] for s in splits)
    num_samples = sum(s['num_samples'] for s in splits)
    size_MB = sum(s['size_MB'] for s in splits)
    return{
        'name': name,
        'input_labels': input_labels,
        'output_labels': output_labels,
        'window_length': window_length,
        'stride': stride,
        'normalization_factors': normalization_factors,
        'num_chunks': num_chunks,
        'num_timesteps': num_timesteps,
        'num_samples': num_samples,
        'size_MB': size_MB,
        'processing_env': processing_env,
        'processing_time_seconds': processing_time_seconds,
        'processing_peak_RAM_MB': processing_peak_RAM_MB,
        'train_val_test_split': train_val_test_split,
        'train_split': splits[0],
        'val_split': splits[1],
        'test_split': splits[2]}

def get_model_info(
        name: str,
        model: Model,
        dataset_info: dict,
        train_env: dict,
        train_time_seconds: float,
        epochs_requested: int,
        epochs_completed: int,
        batch_size: int,
        train_loss_history: list[float],
        val_loss_history: list[float],
        train_peak_RAM_MB: float,
        size_MB: float):
    return {
        'name': name,
        'num_layers': len(model.layers),
        'num_trainable_layers': sum(not isinstance(layer, tf.keras.layers.InputLayer) for layer in model.layers),
        'layer_sequence': [f'{layer.name}: {layer.__class__.__name__} -> {layer.output.shape}' for layer in model.layers],
        'trainable_parameters': model.count_params(),
        'dataset_info': dataset_info,
        'train_env': train_env,
        'train_time_seconds': train_time_seconds,
        'epochs_requested': epochs_requested,
        'epochs_completed': epochs_completed,
        'batch_size': batch_size,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_peak_RAM_MB': train_peak_RAM_MB,
        'size_MB': size_MB}

def get_model_performance_info(
    mse_norm: float,
    rmse_norm: float,
    mse: float,
    rmse: float,
    mae: float,
    eacc: float,
    inference_env: dict,
    inference_time_seconds: float,
    inference_peak_RAM_MB: float):
    return {
        'mse_norm': mse_norm,
        'rmse_norm': rmse_norm,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'eacc': eacc,
        'inference_env': inference_env,
        'inference_time_seconds': inference_time_seconds,
        'inference_peak_RAM_MB': inference_peak_RAM_MB}

# Other Helpers

def read_pickle(filepath: str): 
    with open(filepath, 'rb') as f: return pickle.load(f)

def write_pickle(data: dict, filepath: str): 
    with open(filepath, 'wb') as f: pickle.dump(data,f)
    
def create_directory_dict(folderpath: str):
    
    def build_dict(path):
        directory = {}
        for name in sorted(os.listdir(path)):
            fullpath = os.path.join(path, name)
            if os.path.dir(fullpath): directory[name] = build_dict(fullpath)
            else: directory[os.path.splittext(name)[0]] = fullpath
    directory_dict = build_dict(folderpath)
    return directory_dict
    
# Load and Process Data

def load_ukdale(ukdale_filepath: str):
    data = loadmat(ukdale_filepath)
    X, Y = data['input'], data['output']
    appliance_names = np.array([name.strip() for name in data["labelOut"][2:]])
    T = X.shape[0]
    input_labels = np.array(['P_agg'])
    output_labels = np.array([f'P_{app}' for app in appliance_names])
    X = X[:, 2].reshape(-1,1)
    Y = Y[:, 2:]
    return {
        'X': X,
        'Y': Y,
        'T': T,
        'input_labels': input_labels,
        'output_labels': output_labels}

def filter_by_timesteps(data: dict, idx: list[int]):
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == data['T'] else v)
            for k, v in data.items()}

def filter_by_appliances(data: dict, apps: list[str]):
    idx = np.isin(data['output_labels'], apps)
    return {**data, 'Y': data['Y'][:, idx], 'output_labels': data['output_labels'][idx]}

# Turn Raw Data into Windowed Samples

def precompute_indices(
    num_timesteps: int, 
    window_length: int, 
    stride: int, 
    train_val_test_split: tuple[float, float, float], 
    num_chunks: int, 
    seed=42):
    
    center_offset = window_length // 2
    guard = window_length - 1
    guard_left = guard // 2
    guard_right = guard - guard_left
    rng = np.random.default_rng(seed)
    
    # Calculate numer of chunks in each split.
    n_train_chunks = int(round(train_val_test_split[0] * num_chunks))
    n_val_chunks = int(round(train_val_test_split[1] * num_chunks))
    n_test_chunks = num_chunks - n_train_chunks - n_val_chunks
    
    # Calculate the number of timesteps in each split
    n_train = int(train_val_test_split[0] * num_timesteps)
    n_val = int(train_val_test_split[1] * num_timesteps)
    n_test = num_timesteps - n_train - n_val
    
    # Divide timesteps over given number of chunks
    def make_lengths(total_length, num_chunks):
        lengths = np.full(num_chunks, total_length // num_chunks, dtype=int)
        lengths[:total_length % num_chunks] += 1
        return lengths
    
    chunk_lengths = np.concatenate([
        make_lengths(n_train, n_train_chunks),
        make_lengths(n_val, n_val_chunks),
        make_lengths(n_test, n_test_chunks)])
    
    chunk_labels = (
        ['train'] * n_train_chunks + 
        ['val'] * n_val_chunks +
        ['test'] * n_test_chunks)
    
    # Randomly assign chunks to splits
    perm = rng.permutation(num_chunks)
    chunk_lengths = chunk_lengths[perm]
    chunk_labels = [chunk_labels[i] for i in perm]
    
    # Compute chunk boundaries
    starts = np.zeros(num_chunks, dtype=int)
    ends = np.zeros(num_chunks, dtype=int)
    start=0
    for i, length in enumerate(chunk_lengths):
        starts[i] = start
        end = min(start + length, num_timesteps)
        ends[i] = end
        start = end
        
    split = {
        'train': ([],[]),
        'val': ([],[]),
        'test': ([],[])}
    
    # Assign window indices to splits.
    for i in range(num_chunks):
        start = starts[i]
        end = ends[i]
        
        # Split guard across both sides of a boundary
        usable_start = start
        usable_end = end
        if i > 0 and chunk_labels[i] != chunk_labels[i - 1]: usable_start += guard_right 
        if i < num_chunks - 1 and chunk_labels[i] != chunk_labels[i + 1]: usable_end -= guard_left 
        if usable_end >= usable_start + window_length:
            inp = np.arange(usable_start, usable_end - window_length  + 1, stride) 
            out = inp + center_offset 
            split[chunk_labels[i]][0].append(inp) 
            split[chunk_labels[i]][1].append(out)
            
    # Link windows from different chunks and shuffle
    idx_dict = {}
    idx_dict['num_blocks'] = num_chunks
    for label in ['train', 'val', 'test']:
        if split[label][0]:
            inp = np.concatenate(split[label][0])
            out = np.concatenate(split[label][1])
            perm = rng.permutation(len(inp))
            idx_dict[label] = (inp[perm], out[perm])
        else: idx_dict[label] = (np.array([], dtype=int), np.array([], dtype=int))
        
    idx_dict['train_val_test_split'] = train_val_test_split
    return idx_dict

def count_unique_timesteps(idx_dict: dict, window_length: int):
    
    def count(inp):
        if len(inp) == 0: return 0
        timesteps = np.concatenate([np.arange(i, i + window_length) for i in inp])
        return len(np.unique(timesteps))
    return tuple(count(idx_dict[split][0]) for split in ('train', 'val', 'test'))

def normalize_data(data: dict, idx_dict: dict):

    data_norm = data.copy()

    # Training Indices
    train_inp, train_out = idx_dict['train']
    train_X = data['X'][train_inp]
    train_Y = data['Y'][train_inp]
    
    # Convert hour-of-day to cyclic features
    hour = data['X'][:, 0]
    theta = 2 * np.pi * hour / 24.0
    sin_hour = np.sin(theta).astype(np.float32)
    cos_hour = np.cos(theta).astype(np.float32)
    
    # Normalize aggregate power using training data only
    p_min = train_X.min()
    p_max = train_X.max()
    p_range = max(p_max - p_min, 1e-12)
    p_agg = (data['X'] - p_min) / p_range
    
    # Normalize appliance powers
    y_min = train_Y.min(axis=0)
    y_max = train_Y.max(axis=0)
    y_range = np.maximum(y_max - y_min, 1e-12)
    Y = (data['Y'] - y_min) / y_range
    
    data_norm['X'] = np.column_stack((sin_hour, cos_hour, p_agg)).astype(np.float32)
    data_norm['Y'] = Y.astype(np.float32)
    data_norm['input_labels'] = ['sin_hour', 'cos_hour', 'P_agg']
    
    normalization_factors = {
        'P_agg_min': p_min,
        'P_agg_max': p_max,
        'P_apps_min': y_min,
        'P_apps_max': y_max}
    
    data_norm['normalization_factors'] = normalization_factors
    return data_norm

def process_window(x_win: NDArray[np.float32]):
    x_win = np.asarray(x_win, dtype=np.float32)
    p_seq = x_win[:, 2:3]
    center = center = len(x_win) // 2 
    time_features = x_win[center, 0:2]
    return p_seq, time_features

def generate_sample(
    x_data: NDArray[np.float32], 
    y_data: NDArray[np.float32], 
    i_inp: int, 
    i_out: int, 
    window_length: int):
    
    x_win = x_data[i_inp : i_inp + window_length]
    if x_win.shape[0] != window_length: return None
    p_seq, time_features = process_window(x_win)
    y_target = y_data[i_out]
    return (p_seq, time_features), y_target

def prepare_data(
    data: dict, 
    idx_dict: dict, 
    num_chunks: int, 
    window_length: int, 
    stride: int, 
    save_folderpath: str):
    
    processing_env = get_environment_info()
    
    data = normalize_data(data, idx_dict)
    X, Y = data['X'], data['Y']
    n_appliances = len(data['output_labels'])
    
    n_timesteps_split = count_unique_timesteps(idx_dict, window_length)
    
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    processing_start_time = time.perf_counter()
    
    split_info = []
    for split in ['train', 'val', 'test']:
        
        inp_idx, out_idx = idx_dict[split]
        n_samples = len(inp_idx)
        
        # Allocate Arrays
        X_p = np.empty((n_samples, window_length, 1), dtype=np.float32)
        X_time = np.empty((n_samples, 2), dtype=np.float32)
        Y_p = np.empty((n_samples, n_appliances), dtype=np.float32)
        peak_ram = max(peak_ram, process.memory_info().rss)
    
        # Generate Samples
        for j, (i_inp, i_out) in enumerate(zip(inp_idx, out_idx)):
            (p_seq, time_features), y_target = generate_sample(X, Y, i_inp, i_out, window_length)
            peak_ram = max(peak_ram, process.memory_info().rss)
            X_p[j] = p_seq
            X_time[j] = time_features
            Y_p[j] = y_target
    
        split_size = (X_p.nbytes + X_time.nbytes + Y_p.nbytes)
        
        # Output Directory
        split_dir = os.path.join(save_folderpath, split)
        os.makedirs(split_dir, exist_ok=True)
    
        # Save Arrays
        np.save(os.path.join(split_dir, 'X_p.npy'), X_p)
        np.save(os.path.join(split_dir, 'X_time.npy'), X_time)
        np.save(os.path.join(split_dir, 'Y_p'), Y_p)
        peak_ram = max(peak_ram, process.memory_info().rss)
        
        split_info.append(get_dataset_split_info(n_timesteps_split[0], n_samples, split_size))
    
    processing_time_seconds = time.perf_counter() - processing_start_time()
        
    # Metadata
    name = os.path.basename(save_folderpath)
    normalization_factors = data['normalization_factors']
    metadata = get_dataset_info(
        name,
        data['input_labels'],
        data['output_labels'],
        window_length,
        stride,
        normalization_factors,
        num_chunks,
        processing_env,
        processing_time_seconds,
        peak_ram / 1024**2,
        idx_dict['train_val_test'],
        split_info[0],
        split_info[1],
        split_info[2])
    metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
    write_pickle(metadata, metadata_filepath)
    
# Model Training
    
def load_processed_data(directory_dict: dict, split: dict):
    with open(directory_dict['metadata'], 'rb') as f: metadata = pickle.load(f)
    processed_data_dict = {
        'X_p': np.load(directory_dict[split]['X_p'], mmap_mode='r'),
        'X_time': np.load(directory_dict[split]['X_time'], mmap_mode='r'),
        'Y_p': np.load(directory_dict[split]['Y_p'], mmap_mode='r'),
        'normalization_factors': metadata['normalization_factors']}
    return processed_data_dict

def generate_batch(processed_data_dict: dict, idx_list: list[int]):
    X_p_batch = processed_data_dict['X_p'][idx_list]
    X_time_batch = processed_data_dict['X_time'][idx_list]
    Y_p_batch = processed_data_dict['Y_p'][idx_list]
    return (X_p_batch, X_time_batch), Y_p_batch

def build_model(window_length: int):
    
    # CNN branch
    inp_power = layers.Input(shape=(window_length,1), name='power_input')
    x1 = layers.Conv1D(32, 5, activation='relu', padding='same')(inp_power)
    x1 = layers.Conv1D(64, 5, activation='relu', padding='same')(x1)
    x1 = layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # MLP branch
    inp_time = layers.Input(shape=(2,), name='time_input')
    x2 = layers.Dense(16, activation='relu')(inp_time)
    x2 = layers.Dense(64, activation='relu')(x2)
    
    # Concatenate
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    out = layers.Dense(1, name='power_output')(x)
    model = models.Model(inputs=[inp_power, inp_time], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model



        