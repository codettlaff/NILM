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
    input_labels: list[str],
    output_labels: list[str],
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
        training_dataset_info: dict,
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
        'training_dataset_info': training_dataset_info,
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
            if os.path.isdir(fullpath): directory[name] = build_dict(fullpath)
            else: directory[os.path.splitext(name)[0]] = fullpath
        return directory
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

def filter_by_timesteps(data: dict, idx: tuple[int, int]):
    start, end = idx
    return {k: (v[start:end] if isinstance(v, np.ndarray) and len(v) == data['T'] else v)
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
    idx_dict['num_chunks'] = num_chunks
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
    i = 0
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
        
        split_info.append(get_dataset_split_info(n_timesteps_split[i], n_samples, split_size/1024**2))
        i += 1
    
    processing_time_seconds = time.perf_counter() - processing_start_time
        
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
        idx_dict['train_val_test_split'],
        split_info)
    metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
    write_pickle(metadata, metadata_filepath)
    
# Model Training
    
def load_processed_data(directory_dict: dict, split: str):
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

def train_model(
    dataset_directory_dict: dict,
    epochs: int,
    batch_size: int,
    save_folderpath: str):
    
    name = os.path.basename(save_folderpath)
    train_env = get_environment_info()
    
    dataset_metadata = read_pickle(dataset_directory_dict['metadata'])
    window_length = dataset_metadata['window_length']
    model = build_model(window_length)
    train_data = load_processed_data(dataset_directory_dict, 'train')
    val_data = load_processed_data(dataset_directory_dict, 'val')
    n_samples_train = len(train_data['Y_p'])
    n_samples_val = len(val_data['Y_p'])
    
    best_val_loss = np.inf
    epochs_completed = 0
    patience = 5
    patience_counter = 0
    
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    train_loss_history_epochs, val_loss_history_epochs, train_loss_history_batches, val_loss_history_batches = [],[],[],[]
    train_start_time = time.perf_counter()
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        
        # Training
        train_loss = 0.0
        perm = np.random.permutation(n_samples_train)
        n_train_batches = 0
        for i in tqdm(range(0, n_samples_train, batch_size), desc='Training', leave=False):
            batch_idx = perm[i: i + batch_size]
            (X_p, X_time), Y_p = generate_batch(train_data, batch_idx)
            loss = model.train_on_batch([X_p, X_time], Y_p)
            peak_ram = max(peak_ram, process.memory_info().rss)
            train_loss += loss
            train_loss_history_batches.append(loss)
            n_train_batches += 1
        train_loss /= n_train_batches
        train_loss_history_epochs.append(train_loss)
        
        # Validation
        val_loss = 0.0
        num_val_batches = 0
        for i in tqdm(range(0, n_samples_val, batch_size), desc='Validation', leave=False):
            batch_idx = np.arange(i, min(i + batch_size, n_samples_val))
            (X_p, X_time), Y_p = generate_batch(val_data, batch_idx)
            loss = model.test_on_batch([X_p, X_time], Y_p)
            peak_ram = max(peak_ram, process.memory_info().rss)
            val_loss += loss
            val_loss_history_batches.append(loss)
            num_val_batches += 1
        val_loss /= num_val_batches
        val_loss_history_epochs.append(val_loss)
        best_val_loss = min(best_val_loss, val_loss)
        epochs_completed += 1 
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else: 
            patience_counter += 1
            if patience_counter >= patience: break
        
    train_time_seconds = time.perf_counter() - train_start_time
    model_filepath = os.path.join(save_folderpath, 'model.keras')
    model.save(model_filepath)
    model_size_MB = os.path.getsize(model_filepath) / 1024**2 
    
    # Model Metadata
    model_metadata = get_model_info(
        name,
        model,
        dataset_metadata,
        train_env,
        train_time_seconds,
        epochs,
        epochs_completed,
        batch_size,
        train_loss_history_batches,
        val_loss_history_batches,
        peak_ram / 1024**2,
        model_size_MB)
    metadata_filepath = os.path.join(save_folderpath, 'metadata.pkl')
    write_pickle(model_metadata, metadata_filepath)
    
# Model Testing

def test_model(dataset_directory_dict, model_directory_dict, results_save_filepath=None):
    
    model_metadata = read_pickle(model_directory_dict['metadata'])
    model = load_model(model_directory_dict['model'])
    
    dataset_metadata = read_pickle(dataset_directory_dict['metadata'])
    test_data = load_processed_data(dataset_directory_dict, 'test')
    
    y_min = dataset_metadata['normalization_factors']['P_apps_min']
    y_max = dataset_metadata['normalization_factors']['P_apps_max']
    n_test_samples = dataset_metadata['test_split']['num_samples']
    batch_size = model_metadata['batch_size']
    
    y_true_all, y_pred_all = [], []
    inference_env = get_environment_info()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    inference_start_time = time.perf_counter()
    
    # Inference
    for i in tqdm(range(0, n_test_samples, batch_size), desc='Inference'):
        batch_idx = np.arange(i, min(i + batch_size, n_test_samples))
        (X_p, X_time), Y_true = generate_batch(test_data, batch_idx)
        Y_pred = model.predict_on_batch([X_p, X_time])
        peak_ram = max(peak_ram, process.memory_info().rss)
        y_true_all.append(Y_true)
        y_pred_all.append(Y_pred)
    inference_time_seconds = time.perf_counter() - inference_start_time
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
    if results_save_filepath: np.savez(results_save_filepath, y_true=y_true_denorm, y_pred=y_pred_denorm)
    
    # Update Model Metadata
    model_performance = get_model_performance_info(
        mse_norm,
        rmse_norm,
        mse_denorm,
        rmse_denorm,
        mae,
        eacc,
        inference_env,
        inference_time_seconds,
        peak_ram / 1024**2)
    
    model_metadata['testing_dataset_info'] = dataset_metadata
    model_metadata['performance'] = model_performance
    write_pickle(model_metadata, model_directory_dict['metadata'])
    
# Automation

def create_appliance_datasets(raw_data_filepath, window_length, stride, num_chunks, train_val_test_split, save_folderpath, T_limit=None, target_appliances=None):
    
    data = load_ukdale(raw_data_filepath)
    if T_limit: data = filter_by_timesteps(data, (0, T_limit))
    if target_appliances: data = filter_by_appliances(data, target_appliances)
    idx_dict = precompute_indices(
        num_timesteps=len(data['X']),
        window_length=window_length,
        stride=stride,
        train_val_test_split=train_val_test_split,
        num_chunks=num_chunks,
        seed=42)
    
    # One dataset per appliance
    for out_label in data['output_labels']:
        app = out_label.split('_')[1]
        app_save_folderpath = os.path.join(save_folderpath, app)
        os.makedirs(app_save_folderpath, exist_ok=True)
        prepare_data(
            data,
            idx_dict,
            num_chunks,
            window_length,
            stride,
            app_save_folderpath)
        
def create_house_datasets(raw_data_filepath_list, window_length, stride, num_chunks, train_val_test_split, save_folderpath, T_limit=None, target_appliances=None):
    
    for raw_data_filepath in raw_data_filepath_list: 
        house_name = os.path.basename(raw_data_filepath).split('.')[0]
        house_save_folderpath = os.path.join(save_folderpath, house_name)
        create_appliance_datasets(
            raw_data_filepath,
            window_length,
            stride,
            num_chunks,
            train_val_test_split,
            house_save_folderpath,
            T_limit,
            target_appliances)
    
def centralize_data(folderpath_list, save_folderpath, target_appliances=None):
    
    def concatenate_datasets(npy_filepath_list, output_filepath):
        valid_filepaths = []
        for filepath in npy_filepath_list:
            arr = np.load(filepath, mmap_mode='r')
            if np.isnan(arr).any(): continue
            valid_filepaths.append(filepath)
        
        total_rows, shape, dtype = 0, None, None
        for filepath in valid_filepaths:
            arr = np.load(filepath, mmap_mode='r')
            total_rows += arr.shape[0]
            if shape is None:
                shape = arr.shape[1:]
                dtype = arr.dtype
        
        # Create memory-mapped output array
        output = np.lib.format.open_memmap(output_filepath, mode='w+', dtype=dtype, shape=(total_rows, *shape))
        
        # Copy one dataset at a time
        start = 0
        for filepath in valid_filepaths:
            arr = np.load(filepath, mmap_mode='r')
            end = start + arr.shape[0]
            output[start:end] = arr
            start = end
            
        del output # flush to disk
        
    directory_dict = {}
    for folderpath in folderpath_list:
        house_name = os.path.basename(folderpath)
        directory_dict[house_name] = create_directory_dict(folderpath)
        
    def group_filepaths(directory_dict):
        splits = ('train', 'val', 'test')
        fields = ('input_labels', 'output_labels', 'window_length', 'stride')
        X_p_list, X_time_list, Y_list, metadata_dict = {}, {}, {}, {}
        
        for house, house_dict in directory_dict.items():
            for appliance, appliance_dict in house_dict.items():
                metadata = read_pickle(appliance_dict['metadata'])
                
                # First occurance of this appliance
                if appliance not in metadata_dict.keys():
                    metadata_dict[appliance] = [metadata]
                    X_p_list[appliance] = {split: [] for split in splits}
                    X_time_list[appliance] = {split: [] for split in splits}
                    Y_list[appliance] = {split: [] for split in splits}
                    
                # Verify shared metadata fields match previous homes
                else:
                    reference = metadata_dict[appliance][0]
                    if any(not np.array_equal(metadata[field], reference[field]) for field in fields): continue # skip appliance
                    metadata_dict[appliance].append(metadata)
                    
                # Add filepaths
                for split in splits:
                    X_p_list[appliance][split].append(appliance_dict[split]['X_p'])
                    X_time_list[appliance][split].append(appliance_dict[split]['X_time'])
                    Y_list[appliance][split].append(appliance_dict[split]['Y_p'])
                    
        return X_p_list, X_time_list, Y_list, metadata_dict
    
    X_p_lists, X_time_lists, Y_lists, metadata_dict = group_filepaths(directory_dict)
    splits = ('train', 'val', 'test')
    
    # concatenate and save datasets
    for name, file_lists in {
            'X_p': X_p_lists,
            'X_time': X_time_lists,
            'Y_p': Y_lists}.items():
        for appliance, split_dict in file_lists.items():
            for split, filepath_list in split_dict.items():
                concatenated_dataset_folderpath = os.path.join(save_folderpath, appliance, split)
                os.makedirs(concatenated_dataset_folderpath, exist_ok=True)
                concatenated_dataset_filepath = os.path.join(concatenated_dataset_folderpath, name + '.npy')
                concatenate_datasets(filepath_list, concatenated_dataset_filepath)
    
    # accumulate metadata
    for appliance, metadata_list in metadata_dict.items():
        reference = metadata_list[0]
        centralized_metadata = reference.copy()
        
        global_nfs = centralized_metadata['normalization_factors']
        total_timesteps = 0
        processing_time = 0.0
        peak_RAM_MB = 0.0
        for metadata in metadata_list:
            nf = metadata['normalization_factors']
            global_nfs['P_agg_min'] = min(global_nfs['P_agg_min'], nf['P_agg_min'])
            global_nfs['P_agg_max'] = max(global_nfs['P_agg_max'], nf['P_agg_max'])
            global_nfs['P_apps_min'] = np.minimum(global_nfs['P_apps_min'], nf['P_apps_min'])
            global_nfs['P_apps_max'] = np.maximum(global_nfs['P_apps_max'], nf['P_apps_max'])
            total_timesteps += metadata['num_timesteps']
            processing_time += metadata['processing_time_seconds']
            peak_RAM_MB = max(peak_RAM_MB, metadata['processing_peak_RAM_MB'])
        centralized_metadata['normalization_factors'] = global_nfs
        centralized_metadata['num_timesteps'] = total_timesteps
        centralized_metadata['processing_time_seconds'] = processing_time
        centralized_metadata['processing_peak_RAM_MB'] = peak_RAM_MB
        
        # accumulate split info
        timesteps_used, num_samples, num_timesteps, size_MB = 0, 0, 0, 0.0
        for split in splits:
            split_num_samples, split_num_timesteps, split_size_MB = 0, 0, 0.0
            
            for metadata in metadata_list:
                split_info = metadata[split + '_split']
                split_num_timesteps += split_info['num_timesteps']
                split_num_samples += split_info['num_samples']
                split_size_MB += split_info['size_MB']
                
            split_info = get_dataset_split_info(split_num_timesteps, split_num_samples, split_size_MB)
            num_samples += split_num_samples
            num_timesteps += split_num_timesteps
            size_MB += split_size_MB
            centralized_metadata[f'{split}_split'] = split_info
            
        centralized_metadata['num_samples'] = num_samples
        centralized_metadata['num_timesteps'] = num_timesteps
        centralized_metadata['size_MB'] = size_MB
        
        centralized_metadata['name'] = os.path.basename(save_folderpath) + f'_{appliance}'
        train_split = round(centralized_metadata['train_split']['num_samples'] / centralized_metadata['num_samples'], 2)
        val_split = round(centralized_metadata['val_split']['num_samples'] / centralized_metadata['num_samples'], 2)
        test_split = round(centralized_metadata['test_split']['num_samples'] / centralized_metadata['num_samples'], 2)
        train_val_test_split = [train_split, val_split, test_split]
        centralized_metadata['train_val_test_split'] = train_val_test_split
        
        # centralized_metadata.pop('processing_env') # No longer makes sense, since every house may have a different processing env.
        # centralized_metadata.pop('num_chunks') # No longer useful, every house may have used different number of chunks.
        
        centralized_metadata_filepath = os.path.join(save_folderpath, appliance, 'metadata.pkl')
        write_pickle(centralized_metadata, centralized_metadata_filepath)
        
def train_all_appliance_models(data_folderpath, save_folderpath, epochs, batch_size):
    data_directory_dict = create_directory_dict(data_folderpath)
    model_base_name = os.path.basename(save_folderpath)
    for appliance, appliance_dict in data_directory_dict.items():
        model_name = f'{model_base_name}_{appliance}'
        model_save_folderpath = os.path.join(save_folderpath, model_name)
        os.makedirs(model_save_folderpath, exist_ok=True)
        train_model(appliance_dict, epochs, batch_size, model_save_folderpath)

def test_all_appliance_models(data_folderpath, models_folderpath, batch_size, results_folderpath=None):
    data_directory_dict = create_directory_dict(data_folderpath)
    models_directory_dict = create_directory_dict(models_folderpath)
    model_names = models_directory_dict.keys()
    for model_name in tqdm(model_names, desc='Appliance Models'):
        model_results_folderpath = os.path.join(results_folderpath, model_name)
        os.makedirs(model_results_folderpath, exist_ok=True)
        appliance_model_directory_dict = models_directory_dict[model_name]
        appliance_name = model_name.split('_')[-1]
        appliance_data_directory_dict = data_directory_dict[appliance_name]
        test_model(appliance_data_directory_dict, appliance_model_directory_dict, model_results_folderpath)
        
def resave_model(model_keras_filepath, architecture_json_filepath, weights_filepath, make_json=True):
    if make_json:
        model = load_model(model_keras_filepath)
        with open(architecture_json_filepath, 'w') as f: f.write(model.to_json())
        weights = np.array(model.get_weights(), dtype=object)
        np.save(weights_filepath, weights, allow_pickle=True)
    else:
        with open(architecture_json_filepath, 'r') as f: model = tf.keras.models.model_from_json(f.read())
        weights = np.load(weights_filepath, allow_pickle=True)
        model.set_weights(weights.tolist())
        model.save(model_keras_filepath)
        
def display_model_information(model_metadata_filepath):
    metadata = read_pickle(model_metadata_filepath)

    print(f"\n{'=' * 60}")
    print(f"MODEL: {metadata['name']}")
    print(f"{'=' * 60}")

    print("\nArchitecture")
    print(f"  Layers:               {metadata['num_layers']}")
    print(f"  Trainable layers:     {metadata['num_trainable_layers']}")
    print(f"  Trainable parameters: {metadata['trainable_parameters']:,}")
    print("  Layer sequence:")
    for layer in metadata['layer_sequence']:
        print(f"    {layer}")

    for dataset_type in ('training_dataset_info', 'testing_dataset_info'):
        dataset = metadata.get(dataset_type)
        if dataset is None:
            continue

        print(f"\n{dataset_type.replace('_', ' ').title()}")
        print(f"  Name:                 {dataset['name']}")
        print(f"  Input labels:         {dataset['input_labels']}")
        print(f"  Output labels:        {dataset['output_labels']}")
        print(f"  Window length:        {dataset['window_length']}")
        print(f"  Stride:               {dataset['stride']}")
        print(f"  Num timesteps:        {dataset['num_timesteps']:,}")
        print(f"  Num samples:          {dataset['num_samples']:,}")
        print(f"  Size:                 {dataset['size_MB']:.2f} MB")
        print(f"  Num chunks:           {dataset['num_chunks']}")
        print(f"  Train/Val/Test split: {dataset['train_val_test_split']}")

        print("  Normalization factors:")
        for key, value in dataset['normalization_factors'].items():
            print(f"    {key}: {value}")

        print("  Splits:")
        for split in ('train_split', 'val_split', 'test_split'):
            info = dataset[split]
            print(
                f"    {split.replace('_split', '').title()}: "
                f"{info['num_samples']:,} samples, "
                f"{info['num_timesteps']:,} timesteps, "
                f"{info['size_MB']:.2f} MB"
            )

    print("\nTraining")
    print(f"  Epochs:               {metadata['epochs_completed']} / {metadata['epochs_requested']}")
    print(f"  Batch size:           {metadata['batch_size']}")
    print(f"  Training time:        {metadata['train_time_seconds']:.2f} s")
    print(f"  Peak RAM:             {metadata['train_peak_RAM_MB']:.2f} MB")
    print(f"  Model size:           {metadata['size_MB']:.2f} MB")

    print("\nLoss")
    print(f"  Final train loss:     {metadata['train_loss_history'][-1]:.6f}")
    print(f"  Final validation loss:{metadata['val_loss_history'][-1]:.6f}")

    print("\nPerformance")
    performance = metadata.get('performance', {})
    for key in (
        'mse_norm', 'rmse_norm', 'mse', 'rmse', 'mae', 'eacc',
        'inference_time_seconds', 'inference_peak_RAM_MB'
    ):
        if key in performance:
            print(f"  {key}: {performance[key]}")

    print(f"\n{'=' * 60}\n")
        
if __name__ == '__main__':
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    
    # Settings
    T_limit = 172800 # Two Days
    train_val_test_split = [0.7, 0.15, 0.15]
    window_length = 300 # 5 Minutes
    stride = 1 
    num_chunks = 100
    epochs = 20 #20 
    batch_size = 32
    
    # Script Tasks
    generate_house_data = False # Done
    generate_centralized_data = False # Done
    train_single_house_model = False # Done
    train_centralized_model = False # Done
    test_single_house_model = False # Done
    test_centralized_model = False # Done
    display_results = False # Done
    resave_models = False # Create Json and Npy Files
    resave_models1 = False # Reconstruct keras file from JSON and Npy Files

    # Pre-Process Data
    ukdale_folderpath = os.path.join(data_dir, 'ukdale')
    ukdale_processed_folderpath = os.path.join(data_dir, 'ukdale_processed')
    os.makedirs(ukdale_processed_folderpath, exist_ok=True)
    ukdale_filepath_list = [
        os.path.join(ukdale_folderpath, f)
        for f in os.listdir(ukdale_folderpath)
        if f.endswith('.mat')]
    
    if generate_house_data:
        create_house_datasets(ukdale_filepath_list, window_length, stride, num_chunks, train_val_test_split, ukdale_processed_folderpath, T_limit)
        
    centralized_data_folderpath = os.path.join(data_dir, 'ukdale_centralized')
    os.makedirs(centralized_data_folderpath, exist_ok=True)
    house_name_list = [f for f in os.listdir(ukdale_processed_folderpath) if os.path.isdir(os.path.join(ukdale_processed_folderpath, f))]
    house_name_list.remove('ukdale3') # Contains NaNs
    house_folderpath_list = [os.path.join(ukdale_processed_folderpath, f) for f in house_name_list]
    if generate_centralized_data:
        centralize_data(house_folderpath_list, centralized_data_folderpath)
        
    # Train single house, one model per appliance
    house1_model_folderpath = os.path.join(models_dir, 'ukdale1')
    os.makedirs(house1_model_folderpath, exist_ok=True)
    house1_data_folderpath = house_folderpath_list[0]
    if train_single_house_model:
        train_all_appliance_models(house1_data_folderpath, house1_model_folderpath, epochs, batch_size)
        
    # Train centralized model, one model per appliance
    centralized_model_folderpath = os.path.join(models_dir, 'ukdale_centralized')
    os.makedirs(centralized_model_folderpath, exist_ok=True)
    if train_centralized_model:
        train_all_appliance_models(centralized_data_folderpath, centralized_model_folderpath, epochs, batch_size)
        
    # Test single house, one model per appliance
    if test_single_house_model:
        test_all_appliance_models(house1_data_folderpath, house1_model_folderpath, batch_size, results_dir)
        
    # Test centralized model
    if test_centralized_model:
        test_all_appliance_models(centralized_data_folderpath, centralized_model_folderpath, batch_size, results_dir)
        
    # Compare Results
    single_house_models = create_directory_dict(house1_model_folderpath)
    centralized_models = create_directory_dict(centralized_model_folderpath)
    
    if display_results:
        
        for model_name, directory_dict in single_house_models.items():
            metadata_filepath = directory_dict['metadata']
            display_model_information(metadata_filepath)
        
        for model_name, directory_dict in centralized_models.items():
            metadata_filepath = directory_dict['metadata']
            display_model_information(metadata_filepath)
            
    if resave_models:
        
        for model_name, directory_dict in single_house_models.items():
            model_filepath = directory_dict['model']
            json_filepath = os.path.join(os.path.dirname(model_filepath), 'architecture.json')
            weights_filepath = os.path.join(os.path.dirname(model_filepath), 'weights.npy')
            resave_model(model_filepath, json_filepath, weights_filepath)
        
        for model_name, directory_dict in centralized_models.items():
            model_filepath = directory_dict['model']
            json_filepath = os.path.join(os.path.dirname(model_filepath), 'architecture.json')
            weights_filepath = os.path.join(os.path.dirname(model_filepath), 'weights.npy')
            resave_model(model_filepath, json_filepath, weights_filepath)
            
    if resave_models1:
        
        for model_name, directory_dict in single_house_models.items():
            model_filepath = directory_dict['model']
            json_filepath = os.path.join(os.path.dirname(model_filepath), 'architecture.json')
            weights_filepath = os.path.join(os.path.dirname(model_filepath), 'weights.npy')
            resave_model(model_filepath, json_filepath, weights_filepath, make_json=False)
        
        for model_name, directory_dict in centralized_models.items():
            model_filepath = directory_dict['model']
            json_filepath = os.path.join(os.path.dirname(model_filepath), 'architecture.json')
            weights_filepath = os.path.join(os.path.dirname(model_filepath), 'weights.npy')
            resave_model(model_filepath, json_filepath, weights_filepath, make_json=False)