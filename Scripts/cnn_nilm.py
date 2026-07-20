# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:07:59 2026

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

base_dir = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base_dir, 'Data')
results_dir = os.path.join(base_dir, 'Results')
ampds_filepath = os.path.join(data_dir, 'ampds2.npz')
processed_data_filepath = os.path.join(data_dir, 'ampds2_processed')
model_save_filepath = os.path.join(results_dir, 'nilm_cnn_model_2month.keras')

T_limit = 86400 # Two Months
train_test_val_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 1
epochs = 20
batch_size = 32
target_appliances = ['DWE']

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
    
    appliance_names = data['appliance_names']
    indices = [i for i, name in enumerate(appliance_names) if name in target_appliances]
    appliance_names = appliance_names[indices]
    data['appliance_names'] = appliance_names
    data['Y'] = data['Y'][:,indices]
    return data

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
    scaling = data['scaling_factors']
    image_size = window_length + 1
    n_appliances = y_data.shape[1]
    
    filepaths = {}
    for split in ['train', 'val', 'test']:
        inp_idx, out_idx = idx_dict[split]
        n_samples = len(inp_idx)
        
        # Preallocate arrays
        S = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)
        FFT = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)
        Y = np.empty((n_samples, n_appliances), dtype=np.float32)
        
        # Fill arrays
        for j, (i_inp, i_out) in enumerate(zip(inp_idx, out_idx)):
            (s, s_fft), y = generate_sample(x_data, y_data, i_inp, i_out, window_length)
            S[j] = s
            FFT[j] = s_fft
            Y[j] = y
        
        split_dir = f"{save_filepath}_{split}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Save arrays separately (memory-mappable)
        filepaths_dict = {
            'S': os.path.join(split_dir, "S.npy"),
            'FFT': os.path.join(split_dir, "FFT.npy"),
            'Y': os.path.join(split_dir, "Y.npy"),
            'scaling': os.path.join(split_dir, "scaling.npz")}
        np.save(filepaths_dict['S'], S)
        np.save(filepaths_dict['FFT'], FFT)
        np.save(filepaths_dict['Y'], Y)
        np.savez(filepaths_dict['scaling'], **scaling)
        filepaths[split] = filepaths_dict
        
    return filepaths

def load_processed_data(processed_data_filepaths_dict):
    
    scaling_factors = np.load(processed_data_filepaths_dict['scaling_factors'])
    S = np.load(processed_data_filepaths_dict['S'], mmap_mode='r')
    FFT = np.load(processed_data_filepaths_dict['FFT'], mmap_mode='r')
    Y = np.load(processed_data_filepaths_dict['Y'], mmap_mode='r')
    return {
        'S': S,
        'FFT': FFT,
        'Y': Y,
        'scaling_factors': scaling_factors}

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
    
    best_val_loss = np.inf
    patience = 5
    patience_counter = 0
    
    n_train = len(processed_training_data['Y'])
    n_val = len(processed_val_data['Y'])
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        
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
            
        train_loss /= num_train_batches
        
        # Validation
        val_loss = 0.0
        num_val_batches = 0
        
        for i in tqdm(range(0, n_val, batch_size), desc='Validation', leave=False):
            batch_idx = np.arrange(i, min(i + batch_size, n_val))
            (S, FFT), Y = generate_batch(processed_val_data, batch_idx)
            loss = model.test_on_batch([S, FFT], Y)
            val_loss += loss
            num_val_batches += 1
        val_loss /= num_val_batches
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(model_filepath)
        else: 
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping")
                break
            
        model.save(model_filepath)
    
def test_model(model_filepath, processed_data_filepath, batch_size, show=False):
    
    model = load_model(model_filepath)
    processed_data = np.load(processed_data_filepath)
    
    y_min = processed_data['y_min']
    y_max = processed_data['y_max']
    n_samples = len(processed_data['Y'])
    
    y_true_all = []
    y_pred_all = []
    
    # Inference Loop
    for i in range(0, n_samples, batch_size):
        batch_idx = np.arange(i, min(i + batch_size, n_samples))
        (S, FFT), y_true = generate_batch(processed_data, batch_idx)
        y_pred = model.predict_on_batch([S, FFT])
        y_true_all.append(y_true)
        y_pred_all.append(y_pred) 
        
    # Concatenate Batches
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    
    # Metrics in Normalized Units
    mse_norm = np.mean((y_pred - y_true) ** 2)
    rmse_norm = np.sqrt(mse_norm)
    
    # Convert back to Watts
    y_true_denorm = y_true * (y_max - y_min) + y_min
    y_pred_denorm = y_pred * (y_max - y_min) + y_min

    mse_denorm = np.mean((y_pred_denorm - y_true_denorm) ** 2)
    rmse_denorm = np.sqrt(mse_denorm)

    abs_error = np.abs(y_pred_denorm - y_true_denorm)
    mae = np.mean(abs_error)

    avg_true = np.mean(y_true_denorm)

    eacc = 1.0 - (
        np.sum(abs_error) /
        (2.0 * np.sum(y_true_denorm)))
    
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
        
    return {
        "mse_norm": mse_norm,
        "rmse_norm": rmse_norm,
        "mse_denorm": mse_denorm,
        "rmse_denorm": rmse_denorm,
        "mae": mae,
        "eacc": eacc,
        "y_true": y_true_denorm,
        "y_pred": y_pred_denorm}

if __name__ == '__main__':
    
    processed_data_filepaths_dict_save_filepath = os.path.join(processed_data_filepath, "filepaths.pkl")
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
            filepaths_dict = prepare_data(target_data, idx_dict, window_length, target_data_filepath)
            processed_data_filepaths[target_appliance] = filepaths_dict
        with open(processed_data_filepaths_dict_save_filepath, "wb") as f: pickle.dump(processed_data_filepaths, f)
    preprocess_data(ampds_filepath, processed_data_filepath, T_limit, processed_data_filepaths_dict_save_filepath)
    
    # Train One Model per Appliance
    results = {}
    for target_appliance in tqdm(data["appliance_names"], desc="Appliances"):
        model_filepath = os.path.join(results_dir, f"nilm_cnn_{target_appliance}.keras")
        train_data_filepath = processed_data_filepaths['target_appliance']['train']
        val_data_filepath = processed_data_filepaths['target_appliance']['val']
        test_data_filepath = processed_data_filepaths['target_appliance']['test']
    
        processed_training_data = load_processed_data(train_data_filepath)
        processed_val_data = load_processed_data(val_data_filepath)
        processed_test_data = load_processed_data(test_data_filepath)
        
        # Train
        print(f"\n{'=' * 80}")
        print(f"Training model for: {target_appliance}")
        print(f"{'=' * 80}")
        train_model(
            data=data,
            idx_dict=idx_dict,
            window_length=window_length,
            epochs=epochs,
            batch_size=batch_size,
            model_filepath=model_filepath)
        
        # Test
        print("\nStarting Testing...")
        results[target_appliance] = test_model(
            model_filepath=model_filepath,
            data=data,
            test_idx=idx_dict['test'],
            window_length=window_length,
            batch_size=batch_size,
            scaling_factors=data['scaling_factors'],
            show=True)