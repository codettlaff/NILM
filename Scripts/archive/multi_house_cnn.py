# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:50:16 2026

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

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'Data')
results_dir = os.path.join(base_dir, 'Results')

ukdale_folderpath = os.path.join(base_dir, 'ukdale')
processed_data_folderpath = os.path.join(data_dir, 'ukdale_processed')
model_save_folderpath = os.path.join(results_dir, 'nilm_cnn_model')

T_limit = 172800 # Two Days
train_val_test_split = [0.7, 0.15, 0.15]
window_length = 300 # 5 Minutes
stride = 100

def get_dataset_split_info(split_name, num_timesteps, num_samples, window_length, stride, preprocessing_time, preprocessing_peak_ram, dataset_size_MB):
    return{
    'split_name': split_name,
    'num_timesteps': num_timesteps,
    'num_samples': num_samples,
    'preprocessing_time': preprocessing_time,
    'preprocessing_peak_RAM_MB': preprocessing_peak_ram / 1024**2,
    'dataset_size_MB': dataset_size_MB}

def get_model_info(model, model_filepath):
    return {
        'name': os.path.basename(model_filepath),
        'num_layers': len(model.layers),
        'num_trainable_layers': sum(not isinstance(layer, tf.keras.layers.InputLayer) for layer in model.layers),
        'layer_sequence': [f'{layer.name}: {layer.__class__.__name__} -> {layer.output.shape}' for layer in model.layers],
        'size_MB': os.path.getsize(model_filepath) / 1024**2,
        'trainable_parameters': model.count_params()}

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

def get_training_info(n_train, n_val, batch_size, epochs_requested, epochs_completed, train_losses, validation_losses, epoch_times, peak_ram):
    return {
        'num_training_samples': n_train,
        'num_validation_samples': n_val,
        'batch_size': batch_size,
        'epochs_requested': epochs_requested,
        'epochs_completed': epochs_completed,
        'training_loss_history': train_losses,
        'validation_loss_history': validation_losses,
        'epoch_training_time_seconds': epoch_times,
        'total_training_time_seconds': float(np.sum(epoch_times)),
        'avg_epoch_training_time_seconds': float(np.mean(epoch_times)),
        'fastest_epoch_training_time_seconds': float(np.min(epoch_times)),
        'slowest_epoch_training_time_seconds': float(np.max(epoch_times)),
        'peak_RAM_MB': peak_ram / 1024**2}

def get_dataset_metadata(dataset_name, input_labels, output_labels, window_length, stride, normalization_factors, total_timesteps, timesteps_used, num_blocks, timesteps_discarded, train_val_test_split, training_dataset_metadata, validation_dataset_metadata, testing_dataset_metadata):
    return{
    'metadata_name': f'Dataset: {dataset_name}',
    'dataset_name': dataset_name,
    'input_labels': input_labels,
    'output_labels': output_labels,
    'window_length': window_length,
    'stride': stride,
    'normalization_factors': normalization_factors,
    'total_timesteps': total_timesteps,
    'num_timesteps_used': timesteps_used,
    'num_blocks': num_blocks,
    'num_timesteps_discarded': timesteps_discarded,
    'training_split_fraction': train_val_test_split[0],
    'validation_split_fraction': train_val_test_split[1],
    'testing_split_fraction': train_val_test_split[2],
    'training_dataset_metadata': training_dataset_metadata,
    'validation_dataset_metadata': validation_dataset_metadata,
    'testing_dataset_metadata': testing_dataset_metadata,
    'total_preprocessing_time': training_dataset_metadata['preprocessing_time'] + validation_dataset_metadata['preprocessing_time'] + testing_dataset_metadata['preprocessing_time'],
    'preprocessing_peak_RAM_MB': max(training_dataset_metadata['peak_RAM_MB'], validation_dataset_metadata['peak_RAM_MB'], testing_dataset_metadata['peak_RAM_MB'])}

def get_results_metadata(inference_time, peak_ram, mse_norm, rmse_norm, mse_denorm, rmse_denorm, mae, eacc):
    return {
        'execution_environment': get_environment_info(),
        'inference_time_seconds': inference_time,
        'peak_RAM_MB': peak_ram / 1024**2,
        'mse': mse_denorm,
        'rmse': rmse_denorm,
        'mse_norm': mse_norm,
        'rmse_norm': rmse_norm,
        'mae': mae,
        'eacc': eacc}

def get_model_metadata(model_info, training_info, dataset_metadata=None, results_metadata=None):
    model_name = model_info['name']
    model_metadata = {
        'metadata_name': f'Model: {model_name}',
        'model_information': model_info,
        'training_information': training_info}
    if dataset_metadata: model_metadata['dataset_metadata'] = dataset_metadata
    if results_metadata: model_metadata['testing_results_metadata'] = results_metadata
    return model_metadata

def print_metadata(metadata, show=False, txt_filepath=None, overwrite=False):
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
    
    title = metadata.get('metadata_name', 'Metadata')
    lines = [title.upper(), '=' * 80]
    lines.extend(format_value({k: v for k, v in metadata.items() if k != "metadata_name"}))
    text = '\n'.join(lines)
    
    if show: print(text)
    if txt_filepath is not None:
        mode = 'w' if overwrite else 'a'
        with open(txt_filepath, mode) as f:
            if not overwrite: f.write('\n\n')
        f.write(text)
            
def load_data(ukdale_filepath, T_limit=None):
    data = loadmat(ukdale_filepath)
    inputs = data['input']
    outputs = data['output']
    time_seconds = inputs[:, 0]
    time_of_day = (time_seconds % 86400) / 3600.0 # Convert to Hour of Day
    P_agg = inputs[:, 2]
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
    target_data['appliance_names'] = [appliance_names[i] for i in indices]
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
    
    # Training Indices
    train_inp, train_out = idx_dict['train']
    train_X = data['X'][train_inp]
    train_Y = data['Y'][train_out]
    
    # Convert hour-of-day to cyclic features
    hour = data['X'][:, 0]
    theta = 2 * np.pi * hour / 24.0
    sin_hour = np.sin(theta).astype(np.float32)
    cos_hour = np.cos(theta).astype(np.float32)
    
    # Normalize aggregate power using training data only
    p_min = train_X[:, 1].min()
    p_max = train_X[:, 1].max()
    p_range = max(p_max - p_min, 1e-12)
    p_agg = (data['X'][:, 1] - p_min) / p_range
    
    # Normalize appliance powers
    y_min = train_Y.min(axis=0)
    y_max = train_Y.max(axis=0)
    y_range = np.maximum(y_max - y_min, 1e-12)
    Y = (data['Y'] - y_min) / y_range
    
    data_normalized['X'] = np.column_stack((sin_hour, cos_hour, p_agg)).astype(np.float32)
    data_normalized['Y'] = Y.astype(np.float32)
    
    normalization_factors = {
        'x_min': p_min,
        'x_max': p_max,
        'y_min': y_min,
        'y_max': y_max}
    
    data_normalized['normalization_factors'] = normalization_factors
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

def prepare_data(data, idx_dict, window_length, stride, dataset_folderpath):
    data = normalize_data(data, idx_dict)
    x_data = data['X']
    y_data = data['Y']
    n_timesteps = y_data.shape[0]
    n_appliances = y_data.shape[1]
    in_labels = ['Time_of_Day', 'P_aggregate']
    out_labels = [f'P_{appliance}' for appliance in data['appliance_names']]
    
    directory_dict = {}
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    
    metadata_dict = {}
    timesteps_used = 0
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
            peak_ram = max(peak_ram, process.memory_info().rss)
            
        split_time = time.perf_counter() - split_start
        dataset_size_MB = (X_p.nbytes + X_time.nbytes + Y_p.nbytes) / 1024**2
        
        # Output directory
        split_dir = f"{dataset_folderpath}_{split}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(split_dir, 'X_p.npy'), X_p)
        np.save(os.path.join(split_dir, 'X_time.npy'), X_time)
        np.save(os.path.join(split_dir, 'Y_p.npy'), Y_p)
        
        # Save Split Metadata
        n_timesteps_split = n_samples * window_length
        timesteps_used += n_timesteps_split
        metadata = get_dataset_split_info(split, n_timesteps_split, n_samples, window_length, stride, split_time, peak_ram, dataset_size_MB)
        metadata_dict[split] = metadata
        
        directory_dict[split] = {
            'X_p': os.path.join(split_dir, 'X_p.npy'),
            'X_time': os.path.join(split_dir, 'X_time.npy'),
            'Y_p': os.path.join(split_dir, 'Y_p.npy')}
        
    # Save Dataset Metadata
    dataset_name = os.path.basename(dataset_folderpath)
    timesteps_discarded = n_timesteps - timesteps_used
    metadata = get_dataset_metadata(dataset_name, in_labels, out_labels, window_length, stride, data['normalization_factors'], n_timesteps, timesteps_used, idx_dict['num_blocks'], timesteps_discarded, train_val_test_split, metadata_dict['train'], metadata_dict['val'], metadata_dict['test'])
    metadata_filepath = os.path.join(dataset_folderpath, 'metadata.pkl')
    with open(metadata_filepath, 'wb') as f: pickle.dump(metadata, f)
    
    directory_dict['metadata'] = metadata_filepath
    directory_dict_filepath = os.path.join(dataset_folderpath, 'directory_dict.pkl')
    with open(directory_dict_filepath, 'wb') as f: pickle.dump(directory_dict, f)
    
    return metadata, directory_dict

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

def train_model(model, processed_training_data, processed_val_data, epochs, batch_size, model_filepath, dataset_metadata=None):
    
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
    for epoch in tqdm(range(epochs), desc='Epochs'):
        epoch_start = time.perf_counter()
        
        # Training
        train_loss = 0.0
        num_train_batches = 0
        perm = np.random.permutation(n_train)
        for i in tqdm(range(0, n_train, batch_size), desc='Training', leave=False):
            batch_idx = perm[i:i + batch_size]
            (X_p, X_time), Y_p = generate_batch(processed_training_data, batch_idx)
            peak_ram = max(peak_ram, process.memory_info().rss)
            loss = model.train_on_batch([X_p, X_time], Y_p)
            peak_ram = max(peak_ram, process.memory_info().rss)
            train_loss += loss
            num_train_batches += 1
        train_loss /= num_train_batches
        train_losses.append(train_loss)
        
        # Validation
        val_loss = 0.0
        num_val_batches = 0
        for i in tqdm(range(0, n_val, batch_size), desc='Validation', leave=False):
            batch_idx = np.arange(i, min(i + batch_size, n_val))
            (X_p, X_time), Y_p = generate_batch(processed_val_data, batch_idx)
            peak_ram = max(peak_ram, process.memory_info().rss)
            loss = model.test_on_batch([X_p, X_time], Y_p)
            peak_ram = max(peak_ram, process.memory_info().rss)
            val_loss += loss
            num_val_batches += 1
        val_loss /= num_val_batches
        val_losses.append(val_loss)
        
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        epochs_completed += 1 
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            model.save(model_filepath)  
        else:
            patience_counter += 1 
            if patience_counter >= patience: break 
        
    model.save(model_filepath)
    
    # Metadata
    model_info = get_model_info(model, model_filepath)
    training_info = get_training_info(n_train, n_val, batch_size, epochs, epochs_completed, train_losses, val_losses, epoch_times, peak_ram)
    metadata = get_model_metadata(model_info, training_info, dataset_metadata)
    metadata_filepath = (os.path.splitext(model_filepath)[0] + '_metadata.pkl')
    with open(metadata_filepath, 'wb') as f: pickle.dump(metadata, f)
    return model_filepath, metadata_filepath

def test_model(model, processed_testing_data, batch_size, show=False, save_folderpath=None, model_metadata_filepath=None):
    inference_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    
    y_min = processed_testing_data['normalization']['y_min']
    y_max = processed_testing_data['normalization']['y_max']
    n_samples = len(processed_testing_data['Y_p'])
    
    y_true_all = []
    y_pred_all = []
    
    # Inference
    for i in range(0, n_samples, batch_size):
        batch_idx = np.arange(i, min(i + batch_size, n_samples))
        (X_p, X_time), y_true = generate_batch(processed_testing_data, batch_idx)
        y_pred = model.predict_on_batch([X_p, X_time])
        peak_ram = max(peak_ram, process.memory_info().rss)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        
    inference_time = (time.perf_counter() - inference_start)
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    
    # Metrics (normalized)
    mse_norm = np.mean((y_pred - y_true) ** 2)
    rmse_norm = np.sqrt(mse_norm)
    
    # Convert back to watts
    y_true_denorm = y_true * (y_max - y_min) + y_min
    y_pred_denorm = y_pred * (y_max - y_min) + y_min
    
    mse_denorm = np.mean((y_pred_denorm - y_true_denorm) ** 2)
    rmse_denorm = np.sqrt(mse_denorm)
    abs_error = np.abs(y_pred_denorm - y_true_denorm)
    mae = np.mean(abs_error)
    eacc = (1.0 - np.sum(abs_error) / (2.0 * np.sum(y_true_denorm)))
    
    results_metadata = get_results_metadata(inference_time, peak_ram, mse_norm, rmse_norm, mse_denorm, rmse_denorm, mae, eacc)
    if model_metadata_filepath:
        with open(model_metadata_filepath, 'rb') as f: model_metadata = pickle.load(f)
        model_metadata['testing_results_metadata'] = results_metadata
        with open(model_metadata_filepath, 'wb') as f: pickle.dump(model_metadata)
    
    if save_folderpath:
        os.makedirs(save_folderpath, exist_ok=True)
        results_filepath = os.path.join(save_folderpath, 'results.npz')
        np.savez(results_filepath, y_true=y_true_denorm, y_pred=y_pred_denorm)
        metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
        with open(metadata_filepath, 'wb') as f: pickle.dump(results_metadata, f)
        
    return y_true_denorm, y_pred_denorm, results_metadata
    