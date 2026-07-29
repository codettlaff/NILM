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

def data_metadata(name, input_labels, output_labels, window_length, stride, normalization_factors, num_chunks, total_timesteps, processing_time, processing_peak_ram, processing_env, train_test_val_split, train_split, val_split, test_split):
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

def model_performance(mse_norm, rmse_norm, mse_denorm, rmse_denorm, mae, eacc, inference_time, peak_ram, env):
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

def model_metadata(name, model, data_metadata, train_env, train_time_seconds, epochs_requested, epochs_completed, batch_size, train_loss, val_loss, train_peak_ram, size, performance=None):
    metadata = {
        'type': 'model',
        'name': name,
        'num_layers': len(model.layers),
        'num_trainable_layers': sum(not isinstance(layer, tf.keras.layers.InputLayer) for layer in model.layers),
        'layer_sequence': [f'{layer.name}: {layer.__class__.__name__} -> {layer.output.shape}' for layer in model.layers],
        'trainable_parameters': model.count_params(),
        'data_metadata': data_metadata,
        'train_env': train_env,
        'train_time_seconds': train_time_seconds,
        'epochs_requested': epochs_requested,
        'epochs_completed': epochs_completed,
        'batch_size': batch_size,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_peak_RAM_MB': train_peak_ram,
        'size_MB': size / 1024**2}
    if performance: metadata['performance'] = performance
    return metadata

def print_metadata(metadata_filepath, show=False, txt_filepath=None, overwrite=False):
    
    metadata = read_pickle(metadata_filepath)
    
    def format_value(value, indent=0):
        lines = []
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list, tuple)):
                    lines.append(' ' * indent + f'{k}:')
                    lines.extend(format_value(v, indent + 4))
                else: lines.append(' ' * indent + f'{k:<32}: {v}')
                
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (dict, list, tuple)): lines.extend(format_value(item, indent + 4))
            else: lines.append(' ' * indent + f'- {item}')
            
        else: lines.append(' ' * indent + str(value))
        return lines
    
    title = f"Metadata: {metadata['type']}"
    lines = [title.upper(), '=' * 80]
    lines.extend(format_value({k: v for k, v in metadata.items() if k != "metadata_name"}))
    text = '\n'.join(lines)
    
    if show: print(text)
    if txt_filepath is not None:
        mode = 'w' if overwrite else 'a'
        with open(txt_filepath, mode) as f:
            if not overwrite: f.write('\n\n')
        f.write(text)

# Helper Functions

def read_pickle(filepath):
    with open(filepath, 'rb') as f: data = pickle.load(f)
    return data

def write_pickle(data, filepath):
    with open(filepath, 'wb') as f: pickle.dump(data, f)
    
# Pre-process Data

def load_data(ampds_filepath, T_limit):
    
    data = loadmat(ampds_filepath)
    X, Y = data['input'], data['output'] 
    appliance_names = data['labelOut'][2:]
    T = X.shape[0]
    T_limit = min(T, T_limit) if T_limit is not None else T
    X, Y = X[:T_limit], Y[:T_limit]
    
    X = X[:, 2].reshape(-1,1)      # Keep only P (3rd column)
    Y = Y[:, 2:]    # Keep only appliance columns, discard time and id
    
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
    idx_dict['num_blocks'] = number_blocks
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
    p_seq = x_win[:, 2:3]
    center = center = len(x_win) // 2 
    time_features = x_win[center, 0:2]
    return p_seq, time_features

def generate_sample(x_data, y_data, i_inp, i_out, window_length):
    x_win = x_data[i_inp : i_inp + window_length]
    if x_win.shape[0] != window_length: return None
    p_seq, time_features = process_window(x_win)
    y_target = y_data[i_out]
    return (p_seq, time_features), y_target

def prepare_data(data, idx_dict, num_chunks, window_length, stride, save_folderpath):
    
    processing_env = environment_info()
    
    data = normalize_data(data, idx_dict)
    X, Y = data['X'], data['Y']
    n_timesteps = Y.shape[0]
    n_appliances = Y.shape[1]
    in_labels = ['time_of_day', 'p_aggregate']
    out_labels = [f'p_{appliance}' for appliance in data['appliance_names']]
    
    directory_dict = {}
    split_info = {}
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info()
    timesteps_used = 0
    
    processing_start_time = time.perf_counter()
    for split in ['train', 'val', 'test']:
        inp_idx, out_idx = idx_dict[split]
        n_samples = len(inp_idx)
        
        # Allocate Arrays
        X_p = np.empty((n_samples, window_length, 1), dtype=np.float32)
        X_time = np.empty((n_samples, 2), dtype=np.float32)
        Y_p = np.empty((n_samples, n_appliances), dtype=np.float32)
        
        # Generate Samples
        for j, (i_inp, i_out) in enumerate(zip(inp_idx, out_idx)):
            (p_seq, time_features), y_target = generate_sample(X, Y, i_inp, i_out, window_length)
            X_p[j] = p_seq
            X_time[j] = time_features
            Y_p[j] = y_target
            peak_ram = max(peak_ram, process.memory_info().rss)
            
        dataset_size = (X_p.nbytes + X_time.nbytes + Y_p.nbytes) 
        
        # Output Directory
        split_dir = os.path.join(save_folderpath, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save Arrays
        np.save(os.path.join(split_dir, 'X_p.npy'), X_p)
        np.save(os.path.join(split_dir, 'X_time.npy'), X_time)
        np.save(os.path.join(split_dir, 'Y_p'), Y_p)
        
        # Save Split Info
        split_timesteps_used = n_samples * window_length
        timesteps_used += split_timesteps_used
        split_info[split] = data_split(split_timesteps_used, n_samples, dataset_size)
        
        # Update Directory Dict
        directory_dict[split] = {
            'X_p': os.path.join(split_dir, 'X_p.npy'),
            'X_time': os.path.join(split_dir, 'X_time.npy'),
            'Y_p': os.path.join(split_dir, 'Y_p.npy')}
    processing_time = time.perf_counter() - processing_start_time
    
    # Data Metadata
    name = os.path.basename(save_folderpath)
    metadata = data_metadata(name, in_labels, out_labels, window_length, stride, normalization_factors, num_chunks, n_timesteps, processing_time, peak_ram, processing_env, idx_dict['train_test_val_split'], split_info['train'], split_info['val'], split_info['test'])
    metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
    write_pickle(metadata, metadata_filepath)
    
    # Directory Dict
    directory_dict['metadata'] = metadata_filepath
    directory_dict_filepath = os.path.join(save_folderpath, 'directory_dict.pkl')
    
    return directory_dict_filepath

def load_processed_data(directory_dict, split):
    with open(directory_dict['metadata'], 'rb') as f: metadata = pickle.load(f)
    processed_data_dict = {
        'X_p': np.load(directory_dict[split]['X_p'], mmap_mode='r'),
        'X_time': np.load(directory_dict[split]['X_time'], mmap_mode='r'),
        'Y_p': np.load(directory_dict[split]['Y_p'], mmap_mode='r'),
        'normalization_factors': metadata['normalization_factors']}
    return processed_data_dict

def generate_batch(processed_data_dict, idx_list):
    X_p_batch = processed_data_dict['X_p'][idx_list]
    X_time_batch = processed_data_dict['X_time'][idx_list]
    Y_p_batch = processed_data_dict['Y_p'][idx_list]
    return (X_p_batch, X_time_batch), Y_p_batch

def build_model(window_length):
    
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

def train_model(name, data_directory_dict, epochs, batch_size, save_folderpath):
    
    train_env = environment_info()
    
    dataset_metadata = read_pickle(data_directory_dict['metadata'])
    window_length = dataset_metadata['window_length']
    
    model = build_model(window_length)
    train_data = load_processed_data(data_directory_dict, 'train')
    val_data = load_processed_data(data_directory_dict, 'val')
    
    num_samples_train = len(train_data['Y_p'])
    num_samples_val = len(val_data['Y_p'])

    best_val_loss = np.inf
    epochs_completed = 0
    patience = 5
    patience_counter = 0
    
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    train_start_time = time.perf_counter()
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        
        # Training
        train_loss = 0.0
        perm = np.random.permutation(num_samples_train)
        num_train_batches = 0
        for i in tqdm(range(0, num_samples_train, batch_size), desc='Training', leave=False):
            batch_idx = perm[i: i + batch_size]
            (X_p, X_time), Y_p = generate_batch(train_data, batch_idx)
            loss = model.train_on_batch([X_p, X_time], Y_p)
            peak_ram = max(peak_ram, process.memory_info().rss)
            train_loss += loss
            num_train_batches += 1
        train_loss /= num_train_batches
        
        # Validation
        val_loss = 0.0
        num_val_batches = 0
        for i in tqdm(range(0, num_samples_val, batch_size), desc='Validation', leave=False):
            batch_idx = np.arange(i, min(i + batch_size, num_samples_val))
            (X_p, X_time), Y_p = generate_batch(val_data, batch_idx)
            loss = model.test_on_batch([X_p, X_time], Y_p)
            peak_ram = max(peak_ram, process.memory_info().rss)
            val_loss += loss
            num_val_batches += 1
        val_loss /= num_val_batches
        best_val_loss = min(best_val_loss, val_loss)
        
        epochs_completed += 1 
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else: 
            patience_counter += 1
            if patience_counter >= patience: break
        
    train_time = time.perf_counter() - train_start_time
    model_filepath = os.path.join(save_folderpath, name + '.keras')
    model.save(model_filepath)
    model_size = os.path.getsize(model_filepath)
    
    # Model Metadata
    metadata = model_metadata(name, model, dataset_metadata, train_env, train_time, epochs, epochs_completed, batch_size, train_loss, val_loss, peak_ram, model_size)
    metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
    write_pickle(metadata, metadata_filepath)
    
    # Model Directory Dict
    directory_dict = {
        'model': model_filepath,
        'metadata': metadata_filepath}
    directory_dict_filepath = os.path.join(save_folderpath, 'directory_dict.pkl')
    write_pickle(directory_dict, directory_dict_filepath)
    
    return directory_dict_filepath

def test_model(model_directory_dict_filepath, data_directory_dict_filepath):
    
    model_directory_dict = read_pickle(model_directory_dict_filepath)
    data_directory_dict = read_pickle(data_directory_dict_filepath)
    model_metadata = read_pickle(model_directory_dict['metadata'])
    model = load_model(model_directory_dict['model'])
    data_metadata = read_pickle(data_directory_dict['metadata'])
    test_data = load_processed_data(data_directory_dict, 'test')
    
    y_min = data_metadata['normalization_factors']['y_min']
    y_max = data_metadata['normalization_factors']['y_max']
    num_test_samples = data_metadata['test_split']['num_samples']
    batch_size = model_metadata['batch_size']
    
    y_true_all, y_pred_all = [], []
    inference_env = environment_info()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    inference_start_time = time.perf_counter()
    
    # Inference
    for i in tqdm(range(0, num_test_samples, batch_size), desc='Inference'):
        batch_idx = np.arange(i, min(i + batch_size, num_test_samples))
        (X_p, X_time), Y_true = generate_batch(test_data, batch_idx)
        Y_pred = model.predict_on_batch([X_p, X_time])
        peak_ram = max(peak_ram, process.memory_info().rss)
        y_true_all.append(Y_true)
        y_pred_all.append(Y_pred)
    inference_time = time.perf_counter() - inference_start_time
    
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    
    # Metrics (Normalized)
    mse_norm = np.mean((y_pred - y_true)**2)
    rmse_norm = np.sqrt(mse_norm)
    
    # Convert back to Watts
    y_true_denorm = y_true * (y_max - y_min) + y_min
    y_pred_denorm = y_pred * (y_max - y_min) + y_min
    
    # Metrics
    mse_denorm = np.mean((y_pred_denorm - y_true_denorm)**2)
    rmse_denorm = np.sqrt(mse_denorm)
    abs_error = np.abs(y_pred_denorm - y_true_denorm)
    mae = np.mean(abs_error)
    eacc = 1.0 - (np.sum(abs_error) / (2.0 * np.sum(y_true_denorm)))
    
    # Save Results
    results_filepath = os.path.join(os.bath.base_dir(model_directory_dict['model']), 'test_results.npz')
    np.savez(results_filepath, y_true=y_true_denorm, y_pred=y_pred_denorm)
    model_directory_dict['npy_results'] = results_filepath
    write_pickle(model_directory_dict, model_directory_dict_filepath)
    
    results = model_performance(mse_norm, rmse_norm, mse_denorm, rmse_denorm, mae, eacc, inference_time, peak_ram, inference_env)
    model_metadata['performance'] = results
    write_pickle(model_metadata, model_directory_dict['metadata'])
    
# Automation

def process_house_data(raw_data_filepath, window_length, stride, num_chunks, train_val_test_split, save_folderpath, T_limit=None, target_appliances=None):
    data = load_data(raw_data_filepath, T_limit=T_limit)
    if target_appliances: data = filter_by_appliances(data, target_appliances)
    idx_dict = precompute_indices(
        num_timesteps=len(data['X']),
        window_length=window_length,
        stride=stride,
        train_val_test_split=train_val_test_split,
        number_blocks=num_chunks,
        seed=42)
    
    # One Dataset per Appliance
    directory_dict = {}
    for appliance in data['appliance_names']:
        target_data = filter_by_appliances(data, [appliance])
        target_data_save_folderpath = os.path.join(save_folderpath, appliance.strip())
        os.makedirs(target_data_save_folderpath, exist_ok=True)
        target_directory_dict_filepath = prepare_data(target_data, idx_dict, num_chunks, window_length, stride, target_data_save_folderpath)
        directory_dict[appliance] = read_pickle(target_directory_dict_filepath)
        
    directory_dict_filepath = os.path.join(save_folderpath, 'directory_dict.pkl')
    write_pickle(directory_dict, directory_dict_filepath)
    return directory_dict_filepath
    
def process_multiple_house_data(raw_data_filepath_list, window_length, stride, num_chunks, train_val_test_split, save_folderpath, T_limit=None, target_appliances=None):
    directory_dict = {}
    for filepath in raw_data_filepath_list:
        house_name = os.path.basename(filepath).replace('.mat', '')
        house_save_folderpath = os.path.join(save_folderpath, house_name)
        os.makedirs(house_save_folderpath, exist_ok=True)
        directory_dict_filepath = process_house_data(filepath, window_length, stride, num_chunks, train_val_test_split, house_save_folderpath, T_limit, target_appliances)
        directory_dict[house_name] = read_pickle(directory_dict_filepath)
    directory_dict_filepath = os.path.join(save_folderpath, 'directory_dict.pkl')
    write_pickle(directory_dict, directory_dict_filepath)
    return directory_dict_filepath

# Centralize Data

def centralize_data(inp_directory_dict, save_folderpath):
    
    def concatenate_datasets(npy_filepath_list, output_filepath):
        total_rows = 0
        shape = None
        dtype = None
        for filepath in npy_filepath_list:
            arr = np.load(filepath, mmap_mode='r')
            total_rows += arr.shape[0]
            if shape is None:
                shape = arr.shape[1:]
                dtype = arr.dtype
                
        # Create Memory-Mapped Output Array
        output = np.lib.format.open_memmap(output_filepath, mode='w+', dtype=dtype, shape=(total_rows, *shape))
        
        # Copy one dataset at a time.
        start = 0
        for filepath in npy_filepath_list:
            arr = np.load(filepath, mmap_mode='r')
            end = start + arr.shape[0]
            output[start:end] = arr
            start = end
            
        del output # Flush to disk.
        
    def group_filepaths(directory_dict):
        splits = ('train', 'val', 'test')
        fields = ('input_labels', 'output_labels', 'window_length', 'stride')
        X_p_list, X_time_list, Y_list, metadata_dict = {}, {}, {}, {}
        
        for house, house_dict in directory_dict.items():
            for appliance, appliance_dict in house_dict.items():
                metadata = read_pickle(appliance_dict['metadata'])
                
                # First Occurance of this Appliance
                if appliance not in metadata_dict:
                    metadata_dict[appliance] = metadata
                    X_p_list[appliance] = {split: [] for split in splits}
                    X_time_list[appliance] = {split: [] for split in splits}
                    Y_list[appliance] = {split: [] for split in splits}
                    
                # Verify shared metadata fields match previous homes.
                else:
                    reference = metadata_dict[appliance][0]
                    if any(metadata[field] != reference[field] for field in fields): continue # Skip appliance
                    metadata_dict[appliance].append(metadata)
                    
                # Append Filepaths
                for split in splits:
                    X_p_list[appliance][split].append(appliance_dict[split]['X_p'])
                    X_time_list[appliance][split].append(appliance_dict[split]['X_time'])
                    Y_list[appliance][split].append(appliance_dict[split]['Y'])
                    
        return X_p_list, X_time_list, Y_list, metadata_dict
    
    X_p_lists, X_time_lists, Y_lists, metadata_dict = group_filepaths
    splits = ('train', 'val', 'test')
    
    # Create Output Directory
    directory_dict = {}
    for appliance in metadata_dict:
        directory_dict[appliance] = {'metadata': os.path.join(save_folderpath, appliance, 'metadata.pkl')}
        os.makedirs(os.path.join(save_folderpath, appliance), exist_ok=True)
        for split in splits:
            split_dir = os.path.join(save_folderpath, appliance, split)
            os.makedirs(split_dir, exist_ok=True)
            directory_dict[appliance][split] = {
                'X_p': os.path.join(split_dir, 'X_p.npy'),
                'X_time': os.path.join(split_dir, 'X_time.npy'),
                'Y': os.path.join(split_dir, 'Y.npy')}
            
    # Concatenate and Save Datasets
    for name, file_lists in {
            'X_p': X_p_lists,
            'X_time': X_time_lists,
            'Y': Y_lists}.items():
        for appliance, split_dict in file_lists.items():
            for split, filepath_list in split_dict.items():
                np.save(directory_dict[appliance][split], concatenate_datasets(filepath_list))
                
    # Centralize Metadata
    switch = {
        'train': 'training_dataset_metadata',
        'val': 'validation_dataset_metadata',
        'test': 'testing_dataset_metadata'}
       
    # Accumulate MetaData
    for appliance, metadata_list in metadata_dict.items():
        reference = metadata_list[0]
        centralized_metadata = reference.copy()
        
        global_nfs = centralized_metadata['normalization_factors']
        total_timesteps = 0
        processing_time = 0.0
        peak_RAM_MB = 0.0
        for metadata in metadata_list:
            nf = metadata['normalization_factors']
            global_nfs['x_min'] = min(global_nfs['x_min'], nf['x_min'])
            global_nfs['x_max'] = max(global_nfs['x_max'], nf['x_max'])
            global_nfs['y_min'] = min(global_nfs['y_min'], nf['y_min'])
            global_nfs['y_max'] - max(global_nfs['y_max'], nf['y_max'])
            total_timesteps += metadata['total_timesteps']
            processing_time += metadata['processing_time']
            peak_RAM_MB = max(peak_RAM_MB, metadata['processing_peak_RAM_MB'])
        centralized_metadata['normalization_factors'] = global_nfs
        centralized_metadata['total_timesteps'] = total_timesteps
        centralized_metadata['processing_time'] = processing_time
        centralized_metadata['processing_peak_RAM_MB'] = peak_RAM_MB
        
        # Accumulate Split Info
        timesteps_used, num_samples,size_MB = 0, 0, 0.0
        for split in splits:
            split_num_timesteps, split_num_samples, split_size_MB = 0, 0
            
            for metadata in metadata_list:
                split_info = metadata[switch[split]]
                split_num_timesteps += split_info['num_timesteps']
                split_num_samples += split_info['num_samples']
                split_size_MB += split_info['size_MB']
                
            split_info = data_split(split_num_timesteps, split_num_samples, split_size_MB*1024**2)
            timesteps_used += split_num_timesteps
            num_samples += split_num_samples
            size_MB += split_size_MB
            centralized_metadata[f'{split}_split'] = split_info
            
        centralized_metadata['timesteps_used'] = timesteps_used
        centralized_metadata['num_samples'] = num_samples
        centralized_metadata['size_MB'] = size_MB
        
        centralized_metadata['name'] = os.path.basename(save_folderpath) + f'_{appliance}'
        centralized_metadata['timesteps_discarded'] = total_timesteps = timesteps_used
        train_split = centralized_metadata['train_split']['num_samples'] / centralized_metadata['num_samples']
        val_split = centralized_metadata['val_split']['num_samples'] / centralized_metadata['num_samples']
        test_split = centralized_metadata['test_split']['num_samples'] / centralized_metadata['num_samples']
        train_val_test_split = [train_split, val_split, test_split]
        centralized_metadata['train_val_test_split'] = train_val_test_split
        
        centralized_metadata.pop('processing_env') # No longer makes sense, since every house may have a different processing env.
        centralized_metadata.pop('num_chunks') # No longer useful, every house may have used different number of chunks.
        
        centralized_metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
        write_pickle(centralized_metadata, centralized_metadata_filepath)
        
    # Write Directory Dict
    directory_dict_filepath = os.path.join(save_folderpath, 'directory_dict.pkl')
    write_pickle(directory_dict, directory_dict_filepath)
    return directory_dict_filepath

def train_all_appliance_models(data_directory_dict, save_folderpath, epochs, batch_size):
    model_base_name = os.path.basename(save_folderpath)
    all_models_directory_dict = {}
    for appliance, appliance_dict in data_directory_dict.items():
        model_name = f'{model_base_name}_{appliance}'
        model_save_folderpath = os.path.join(save_folderpath, appliance)
        os.makedirs(model_save_folderpath, exist_ok=True)
        model_directory_dict_filepath = train_model(model_name, appliance_dict, epochs, batch_size, model_save_folderpath)
        all_models_directory_dict[appliance] = read_pickle[model_directory_dict_filepath]
    directory_dict_filepath = os.path.join(save_folderpath, 'directory_dict.pkl')
    write_pickle(all_models_directory_dict, directory_dict_filepath)
    return directory_dict_filepath
    
        
def test_all_appliance_models(data_directory_dict, model_directory_dict, batch_size):
    appliance_names = model_directory_dict.keys()
    for appliance in tqdm(appliance_names, desc='Appliances'):
        dataset_metadata_filepath = data_directory_dict[appliance]['metadata']
        dataset_metadata = read_pickle(dataset_metadata_filepath)
        test_model(model_directory_dict['appliance'], dataset_metadata, batch_size)
        
if __name__ == '__main__':
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Settings
    T_limit = 172800 # Two Days
    train_val_test_split = [0.7, 0.15, 0.15]
    window_length = 300 # 5 Minutes
    stride = 1 
    num_chunks = 100
    epochs = 20 
    batch_size = 32
    
    # Pre-Process Data
    ukdale_folderpath = os.path.join(data_dir, 'ukdale')
    ukdale_processed_data_folderpath = os.path.join(data_dir, 'ukdale_processed')
    os.makedirs(ukdale_processed_data_folderpath, exist_ok=True)
    ukdale_filepath_list = [
        os.path.join(ukdale_folderpath, f)
        for f in os.listdir(ukdale_folderpath)
        if f.endswith('.mat')]
    
    ukdale_processed_directory_dict_filepath = process_multiple_house_data(
        ukdale_filepath_list,
        window_length,
        stride,
        num_chunks,
        train_val_test_split,
        ukdale_processed_data_folderpath,
        T_limit)
    
    # Centralize Data
    centralized_data_folderpath = os.path.join(data_dir, 'ukdale_centralized')
    os.makedirs(centralized_data_folderpath, exist_ok=True)
    centralized_data_directory_dict_filepath = centralize_data(
        ukdale_processed_data_folderpath,
        centralized_data_folderpath)
    
    # Single House, 1 Model Per Appliance
    house1_data_directory_dict = load_data(ukdale_processed_directory_dict_filepath)['house1']
    house1_model_save_folderpath = os.path.join(models_dir, 'house1')
    os.makedirs(house1_model_save_folderpath, exist_ok=True)
    house1_model_directory_dict_filepath = train_all_appliance_models(house1_data_directory_dict, house1_model_save_folderpath, epochs, batch_size)
    
    # Train Centralized Model
    model_name = 'ukdale_centralized'
    centralized_model_folderpath = os.path.join(model_name)
    os.makedirs(models_dir, exist_ok=True)
    train_all_appliance_models(read_pickle(centralized_data_directory_dict_filepath), centralized_model_folderpath, epochs, batch_size)
    