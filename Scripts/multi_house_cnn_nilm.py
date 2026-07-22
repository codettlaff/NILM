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

def get_model_info(model, model_filepath, model_name):
    return {
        "name": model_name,
        "filepath": model_filepath,
        "size_MB": os.path.getsize(model_filepath) / 1024**2,
        "trainable_parameters": model.count_params(),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape}

def get_environment_info():
    return {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_RAM_GB": psutil.virtual_memory().total / 1024**3}

def summarize_times(times, total):
    return {
        "epoch_times": times,
        "total_seconds": total,
        "average_seconds": float(np.mean(times)),
        "fastest_seconds": float(np.min(times)),
        "slowest_seconds": float(np.max(times))}

def load_data(ukdale_filepath, T_limit=None):
    
    data = loadmat(ukdale_filepath)
    inputs = data['input']
    outputs = data['output']
    
    time_seconds = inputs[:, 0]
    time_of_day = (time_seconds % 86400) / 3600.0 # Convert to hour of day.
    
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
    p_agg = (data['X'][:, 1] - p_min) / p_range
    
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
    y_target = y_data[i_out]
    
    return (p_seq, time_features), y_target

def prepare_data(data, idx_dict, window_length, stride, save_filepath):
    
    data = normalize_data(data, idx_dict)
    x_data = data['X']
    y_data = data['Y']
    
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
        
        # Metadata
        dataset = {
            'num_timesteps': int(len(x_data)),
            'num_samples': int(n_samples),
            'target_appliances': list(data['appliance_names'])}
        
        preprocessing = {
            'window_length': window_length,
            'stride': stride}
        
        dimensions = {
            'power_input_shape': X_p.shape[1:],
            'time_input_shape': X_time.shape[1:],
            'output_shape': Y_p.shape[1:]}
        
        timing = {
            'preprocessing_seconds': split_time,
            'samples_per_second': n_samples / split_time if split_time > 0 else None}
        
        computation = {
            "X_p_size_MB": X_p.nbytes / 1024**2,
            "X_time_size_MB": X_time.nbytes / 1024**2,
            "Y_p_size_MB": Y_p.nbytes / 1024**2,
            "total_dataset_size_MB": (X_p.nbytes + X_time.nbytes + Y_p.nbytes) / 1024**2,
            "process_RAM_MB": process.memory_info().rss / 1024**2}
        
        metadata = {
            'split': split,
            'dataset': dataset,
            'preprocessing': preprocessing,
            'dimensions': dimensions,
            'normalization': data['scaling_factors'],
            'timing': timing,
            'computation': computation,
            'environment': get_environment_info()}
     
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    
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
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        
        epoch_start = time.perf_counter()
        
        # Training
        train_loss = 0.0 
        num_train_batches = 0 
        perm = np.random.permutation(n_train) 
        
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
        
    total_training_time = time.perf_counter() - training_start
    model.save(model_filepath)
    
    # Training Metadata
    training = {
        'epochs_requested': epochs,
        'epochs_completed': epochs_completed,
        'batch_size': batch_size,
        'best_epoch': best_epoch,
        'best_validation_loss': float(best_val_loss),
        'training_loss_history': train_losses,
        'validation_loss_history': val_losses}
    
    computation = {
        'training_samples': n_train,
        'validation_samples': n_val,
        'training_samples_per_second': n_train * epochs_completed / total_training_time,
        'peak_RAM_MB': peak_ram / 1024**2,
        'current_RAM_MB': process.memory_info().rss / 1024**2}
    
    training_metadata = {
        'model': get_model_info(model, model_filepath, model_name=f'CNN NILM'),
        'training': training,
        'timing': summarize_times(epoch_times, total_training_time),
        'computation': computation,
        'environment': get_environment_info()}
    
    metadata_filepath = (os.path.splitext(model_filepath)[0] + '_metadata.pkl')
    with open(metadata_filepath, 'wb') as f: pickle.dump(training_metadata, f)
    
    return model_filepath, metadata_filepath

def test_model(model_filepath, processed_testing_data, batch_size, show=False, save_filepath=None):
    
    inference_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    
    model = load_model(model_filepath)
    y_min = processed_testing_data['normalization']['y_min']
    y_max = processed_testing_data['normalization']['y_max']
    n_samples = len(processed_testing_data['Y_p'])
    
    y_true_all = []
    y_pred_all = []
    batch_times = []
    
    # Inference
    for i in range(0, n_samples, batch_size):
        
        batch_start = time.perf_counter()
        batch_idx = np.arange(i, min(i + batch_size, n_samples))
        (X_p, X_time), y_true = generate_batch(processed_testing_data, batch_idx)
        
        y_pred = model.predict_on_batch([X_p, X_time])
        batch_times.append(time.perf_counter() - batch_start)
        peak_ram = max( peak_ram, process.memory_info().rss)
        
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        
    total_inference_time = (time.perf_counter() - inference_start)
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
    avg_true = np.mean(y_true_denorm)
    eacc = (1.0 - np.sum(abs_error) / (2.0 * np.sum(y_true_denorm)))
    
    # Display Results
    if show:
        print("\nResults")
        print("=" * 80)
        print(f"{'Metric':<20}{'Value':>15}")
        print("-" * 80)
        print(f"{'Normalized MSE':<20}{mse_norm:>15.6f}")
        print(f"{'Normalized RMSE':<20}{rmse_norm:>15.6f}")
        print(f"{'MSE (Watts)':<20}{mse_denorm:>15.6f}")
        print(f"{'RMSE (Watts)':<20}{rmse_denorm:>15.6f}")
        print(f"{'MAE (Watts)':<20}{mae:>15.6f}")
        print(f"{'Average Load':<20}{avg_true:>15.6f}")
        print(f"{'EACC':<20}{eacc:>15.6f}")
        print("=" * 80)
        
    error_results = {
        "mse_norm": mse_norm,
        "rmse_norm": rmse_norm,
        "mse_denorm": mse_denorm,
        "rmse_denorm": rmse_denorm,
        "mae": mae,
        "eacc": eacc}
    
    timing_results = {
        "total_inference_seconds": total_inference_time,
        "average_batch_seconds": float(np.mean(batch_times)),
        "fastest_batch_seconds": float(np.min(batch_times)),
        "slowest_batch_seconds": float(np.max(batch_times)),
        "samples_per_seconds": n_samples / total_inference_time,
        "milliseconds_per_sample": 1000 * total_inference_time / n_samples}
    
    computation_results = {
        "peak_RAM_MB": peak_ram / 1024**2,
        "current_RAM_MB": process.memory_info().rss / 1024**2}
    
    results = {
        "model": get_model_info(model, model_filepath, model_name="1D CNN NILM"),
        "execution_environment": get_environment_info(),
        "timing_results": timing_results,
        "computation_results": computation_results,
        "true_output": y_true_denorm,
        "predicted_output": y_pred_denorm,
        "error_results": error_results}
    
    if save_filepath:
        with open(save_filepath, 'wb') as f: pickle.dump(results, f)
    return results
    
if __name__ == '__main__':
    
    mat_filepaths = [
        os.path.join(ukdale_folderpath, f)
        for f in os.listdir(ukdale_folderpath)
        if f.endswith('.mat')]
    
    ukdale_filepath = os.path.join(ukdale_folderpath, 'ukdale1.mat')
    processed_data_filepaths_dict_save_filepath = os.path.join(processed_data_folderpath, "filepaths.pkl")
    
    def preprocess_data(ukdale_filepath, processed_data_filepath, T_limit, processed_data_filepaths_dict_save_filepath):
    
            data = load_data(ukdale_filepath, T_limit=T_limit)
            idx_dict = precompute_indices(
                num_timesteps=len(data['X']),
                window_length=window_length,
                stride=stride,
                train_val_test_split=train_test_val_split,
                number_blocks=42)
            
            os.makedirs(processed_data_filepath, exist_ok=True)
            processed_data_filepaths = {}
            
            for target_appliance in data['appliance_names']:
                
                target_data = filter_by_appliance(data, [target_appliance])
                target_data_folderpath = os.path.join(processed_data_filepath, target_appliance)
                os.makedirs(target_data_folderpath, exist_ok=True)
                target_data_filepath = os.path.join(target_data_folderpath, os.path.basename(target_data_folderpath))
                filepaths_dict = prepare_data(target_data, idx_dict, window_length, stride, target_data_filepath)
                processed_data_filepaths[target_appliance] = filepaths_dict
                
            with open(processed_data_filepaths_dict_save_filepath, 'wb') as f:
                pickle.dump(processed_data_filepaths, f)
                
            return processed_data_filepaths
    
    def preprocess_all_house_data(ukdale_folderpath, processed_data_folderpath, T_limit, processed_data_filepaths_dict_save_filepath):
        all_house_data_filepaths = {}
        filepaths_dict_save_filepath = os.path.join(processed_data_folderpath, 'all_filepaths.pkl')
        for mat_filepath in mat_filepaths:
            house_name = os.path.basename(mat_filepath).split('.')[0]
            house_processed_data_folderpath = os.path.join(processed_data_folderpath, house_name)
            os.makedirs(house_processed_data_folderpath, exist_ok=True)
            processed_data_filepaths = preprocess_data(mat_filepath, house_processed_data_folderpath, T_limit, processed_data_filepaths_dict_save_filepath)
            all_house_data_filepaths[house_name] = processed_data_filepaths
        with open(filepaths_dict_save_filepath, 'wb') as file: pickle.dump(file, all_house_data_filepaths)
        return all_house_data_filepaths
        
    all_house_data_filepaths = preprocess_all_house_data(ukdale_folderpath, processed_data_folderpath, T_limit, processed_data_filepaths_dict_save_filepath)
    
    def centralize_data(all_house_data_filepaths): 
        # combine all five houses data into one database
        # write to folder with filepaths pickle the way we did for original houses
        # try to avoid openening all files at once and overloading ram
        # do this for each appliance - combine all houses which share this appliance
        
        os.makedirs(output_folder, exist_ok=True)
        centralized_filepaths = {}
        
        # Find all appliances present across every house
        appliances = sorted({
        appliance
        for house in all_house_data_filepaths.values()
        for appliance in house.keys()})
        
        
    
    # Load pre-processed dataset filepaths
    with open(processed_data_filepaths_dict_save_filepath, 'rb') as f:
        processed_data_filepaths = pickle.load(f)
        
    appliance_names = list(processed_data_filepaths.keys())
    results = {}
    
    trained_model_folderpath = os.path.join(results_dir, 'ukdale_1d_cnn_nilm')
    os.makedirs(trained_model_folderpath, exist_ok=True)
    
    for target_appliance in tqdm(appliance_names, desc='Appliances'):
        
        model_filepath = os.path.join(trained_model_folderpath, f'nilm_cnn_{target_appliance}.keras')
        train_data_filepath = processed_data_filepaths[target_appliance]["train"]
        val_data_filepath = processed_data_filepaths[target_appliance]["val"]
        test_data_filepath = processed_data_filepaths[target_appliance]["test"]
        processed_training_data = load_processed_data(train_data_filepath)
        processed_val_data = load_processed_data(val_data_filepath)
        processed_test_data = load_processed_data(test_data_filepath)
        
        # Build model
        model = build_model(window_length)
        
        # Train
        print(f"\n{'=' * 80}")
        print(f"Training model for: {target_appliance}")
        print(f"{'=' * 80}")
        
        train_model(
            model=model,
            processed_training_data=processed_training_data,
            processed_val_data=processed_val_data,
            epochs=epochs,
            batch_size=batch_size,
            model_filepath=model_filepath)
        
        # Test
        print("\nStarting testing...")
        results[target_appliance] = test_model(
            model_filepath=model_filepath,
            processed_testing_data=processed_test_data,
            batch_size=batch_size,
            show=True)
    
    