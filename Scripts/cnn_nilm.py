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

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f: pickle.dump(obj, f)
    
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

def get_ram_usage(process): return process.memory_info().rss / 1024**2

def summarize_times(times, total):
    return {
        "epoch_times": times,
        "total_seconds": total,
        "average_seconds": float(np.mean(times)),
        "fastest_seconds": float(np.min(times)),
        "slowest_seconds": float(np.max(times))}

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

def generate_batch(processed_data, idx_list):
    S_batch = processed_data['S'][idx_list]
    FFT_batch = processed_data['FFT'][idx_list]
    Y_batch = processed_data['Y'][idx_list]
    return (S_batch, FFT_batch), Y_batch

def build_model():
    
    def build_branch(input_layer):
        x = layers.Conv2D(30, (10, 10), activation='relu')(input_layer)
        x = layers.Conv2D(30, (8, 8), activation='relu')(x)
        x = layers.Conv2D(40, (6, 6), activation='relu')(x)
        x = layers.Conv2D(50, (5, 5), activation='relu')(x)
        x = layers.Conv2D(50, (5, 5), activation='relu')(x)
        x = layers.Flatten()(x)
        return x
    
    # Inputs
    inp_time = layers.Input(shape=(31, 31, 1))
    inp_freq = layers.Input(shape=(31, 31, 1))

    # Branches
    branch_time = build_branch(inp_time)
    branch_freq = build_branch(inp_freq)
    x = layers.Concatenate()([branch_time, branch_freq])

    # Dense + Output
    x = layers.Dense(1024, activation='relu')(x)
    out = layers.Dense(1)(x)

    # Model
    model = models.Model(inputs=[inp_time, inp_freq], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse') 
    return model

def train_model(model, processed_training_data, processed_val_data, epochs, batch_size, model_filepath):
    
    training_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    
    best_val_loss = np.inf
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    n_train = len(processed_training_data['Y'])
    n_val = len(processed_val_data['Y'])
    
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
        
        for i in tqdm(range(0, n_train, batch_size), desc="Training", leave=False):
            
            batch_idx = perm[i:i + batch_size]
            (S, FFT), Y = generate_batch(processed_training_data, batch_idx)
            loss = model.train_on_batch([S, FFT], Y)
            train_loss += loss
            num_train_batches += 1
            peak_ram = max(peak_ram, process.memory_info().rss)
            
        train_loss /= num_train_batches
        
        # Validation
        val_loss = 0.0
        num_val_batches = 0
        
        for i in tqdm(range(0, n_val, batch_size), desc='Validation', leave=False):
            batch_idx = np.arange(i, min(i + batch_size, n_val))
            (S, FFT), Y = generate_batch(processed_val_data, batch_idx)
            loss = model.test_on_batch([S, FFT], Y)
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
    training_metadata = {
        "model": get_model_info(model, model_filepath, model_name='PQ CNN NILM'),
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": epochs_completed,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_validation_loss": float(best_val_loss),
            "training_loss_history": train_losses,
            "validation_loss_history": val_losses},
        "timing": summarize_times(epoch_times, total_training_time),
        "computation": {
            "training_samples": n_train,
            "validation_samples": n_val,
            "training_samples_per_second": n_train * epochs_completed / total_training_time,
            "peak_RAM_MB": peak_ram / 1024**2,
            "current_RAM_MB": process.memory_info().rss / 1024**2},
        "environment": get_environment_info()}
    
    metadata_filepath = os.path.splitext(model_filepath)[0] + "_metadata.pkl"
    with open(metadata_filepath, "wb") as f: pickle.dump(training_metadata, f)
    
def test_model(model_filepath, processed_testing_data, batch_size, show=False, save_filepath=None):
    
    inference_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss
    
    model = load_model(model_filepath)
    y_min = processed_testing_data['scaling_factors']['y_min']
    y_max = processed_testing_data['scaling_factors']['y_max']
    
    n_samples = len(processed_testing_data['Y'])
    
    y_true_all = []
    y_pred_all = []
    batch_times = []
    
    # Inference
    for i in range(0, n_samples, batch_size):
        
        batch_start = time.perf_counter()
        batch_idx = np.arange(i, min(i + batch_size, n_samples))
        (S, FFT), y_true = generate_batch(processed_testing_data, batch_idx)
        y_pred = model.predict_on_batch([S, FFT])
        batch_times.append(time.perf_counter() - batch_start)
        peak_ram = max(peak_ram, process.memory_info().rss)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        
    total_inference_time = (time.perf_counter() - inference_start)
    
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    
    # Metrics
    mse_norm = np.mean((y_pred - y_true) ** 2)
    rmse_norm = np.sqrt(mse_norm)
    
    # Convert to Watts
    y_true_denorm = y_true * (y_max - y_min) + y_min
    y_pred_denorm = y_pred * (y_max - y_min) + y_min
    mse_denorm = np.mean((y_pred_denorm - y_true_denorm) ** 2)
    rmse_denorm = np.sqrt(mse_denorm)
    abs_error = np.abs(y_pred_denorm - y_true_denorm)
    mae = np.mean(abs_error)
    avg_true = np.mean(y_true_denorm)
    eacc = 1.0 - (np.sum(abs_error) / (2.0 * np.sum(y_true_denorm)))
    
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
        
        
    # Results
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
        "model": get_model_info(),
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
    
    processed_data_filepaths_dict_save_filepath = os.path.join(processed_data_folderpath, "filepaths.pkl")
    def preprocess_data(ampds_filepath, processed_data_filepath, T_limit, processed_data_filepaths_dict_save_filepath):
    
        data = load_data(ampds_filepath, T_limit=T_limit)
        idx_dict = precompute_indices(
            num_timesteps=data['T'],
            window_length=window_length,
            stride=stride,
            train_val_test_split=train_test_val_split,
            number_blocks=42)
        
        os.makedirs(processed_data_filepath, exist_ok=True)
        processed_data_filepaths = {}
        for target_appliance in data["appliance_names"]:
            target_data = filter_by_appliances(data, [target_appliance])
            target_data_folderpath = os.path.join(processed_data_filepath, target_appliance)
            os.makedirs(target_data_folderpath, exist_ok=True)
            target_data_filepath = os.path.join(target_data_folderpath, os.path.basename(target_data_folderpath))
            filepaths_dict = prepare_data(target_data, idx_dict, window_length, stride, target_data_filepath)
            processed_data_filepaths[target_appliance] = filepaths_dict
        with open(processed_data_filepaths_dict_save_filepath, "wb") as f: pickle.dump(processed_data_filepaths, f)
    preprocess_data(ampds_filepath, processed_data_folderpath, T_limit, processed_data_filepaths_dict_save_filepath)
    
    # Load Preproccessed Data Filepaths Dict
    with open(processed_data_filepaths_dict_save_filepath, "rb") as f: processed_data_filepaths = pickle.load(f)
    appliance_names = list(processed_data_filepaths.keys())
    
    # Train One Model per Appliance
    results = {}
    for target_appliance in tqdm(appliance_names, desc="Appliances"):
        model_filepath = os.path.join(results_dir, f"nilm_cnn_{target_appliance}.keras")
        train_data_filepath = processed_data_filepaths[target_appliance]['train']
        val_data_filepath = processed_data_filepaths[target_appliance]['val']
        test_data_filepath = processed_data_filepaths[target_appliance]['test']
    
        processed_training_data = load_processed_data(train_data_filepath)
        processed_val_data = load_processed_data(val_data_filepath)
        processed_test_data = load_processed_data(test_data_filepath)
        
        # Create Model Architecture
        model = build_model()
        
        # Train
        # print(f"\n{'=' * 80}")
        # print(f"Training model for: {target_appliance}")
        # print(f"{'=' * 80}")
        # train_model(
        #     model=model,
        #     processed_training_data=processed_training_data,
        #     processed_val_data=processed_val_data,
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     model_filepath=model_filepath)
        
        # Test
        print("\nStarting Testing...")
        results[target_appliance] = test_model(
            model_filepath=model_filepath,
            processed_testing_data=processed_test_data,
            batch_size=batch_size,
            show=True)
        