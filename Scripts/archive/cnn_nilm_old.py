# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:29:09 2026

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
model_save_filepath = os.path.join(results_dir, 'nilm_cnn_model_2month.keras')

# T_limit = 10000
T_limit = 86400 # Two Months
train_test_val_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 1
epochs = 20
# epochs = 1
batch_size = 32
target_appliances = ['DWE']

# Remaining Changes to make:
# Generate_batch() repeatedly computes PQ and FFT for same windows every communication round.
# want to generate and save S, S_fft, targets ahead of time.

# want to split training and testing data first, then save scaling factors

def load_data(ampds_filepath, T_limit, target_appliances=None):

    data = np.load(ampds_filepath)
    X, Y = data['X'], data['Y'] # Shape (1048573, 6), (1048573, 22, 4)
    appliance_names = data['out_labels']
    T = X.shape[0] # Total number of timesteps.
    T_limit = min(T, T_limit) if T_limit is not None else T
    X, Y = X[:T_limit], Y[:T_limit]

    # For X data, keep only columns corresponding to P and Q
    # For Y data, keep only column corresponding to P
    X, Y = X[:, [0,2]], Y[:,:,0]

    # For Y data, keep only columns corresponding to appliances in target appliances list.
    if not target_appliances: target_appliances = appliance_names
    indices = [i for i, name in enumerate(appliance_names) if name in target_appliances]
    appliance_names = appliance_names[indices] # Trim Appliance Names
    Y = Y[:, indices]

    # Normalize Inputs
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    X_norm = (X - x_min) / (x_max - x_min + 1e-8)

    # Normalize Outputs
    y_min = Y.min(axis=0, keepdims=True)
    y_max = Y.max(axis=0, keepdims=True)
    Y_norm = (Y - y_min) / (y_max - y_min + 1e-8)

    scaling_factors = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max}

    return {
        'x': X,
        'x_norm': X_norm,
        'y': Y,
        'y_norm': Y_norm,
        'scaling_factors': scaling_factors,
        'appliance_names': appliance_names,
        'num_timesteps': T_limit}

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
    
    # Timesteps in Each Split
    n_train = int(train_val_test_split[0] * num_timesteps)
    n_val = int(train_val_test_split[1] * num_timesteps)
    n_test = num_timesteps - n_train - n_val
    
    # Divide total number of timesteps as evenly as possible over given number of blocks.
    def make_lengths(total_length, n_blocks):
        lengths = np.full(n_blocks, total_length // n_blocks, dtype=int)
        lengths[:total_length % n_blocks] += 1 # Add leftover timesteps to first blocks.
        return lengths # Array of size equal to number of blocks. Contains n_timesteps for each block.
    
    # Create block lengths (number of timesteps per block).
    block_lengths = np.concatenate([
        make_lengths(n_train, n_train_blocks),
        make_lengths(n_val, n_val_blocks),
        make_lengths(n_test, n_test_blocks)])
    
    # Indicate which dataset each block belongs to.
    # Not yet shuffled.
    block_labels = (
        ['train'] * n_train_blocks + 
        ['val'] * n_val_blocks +
        ['test'] * n_test_blocks)
    
    # Randomly assign blocks among the timeline
    # Solves seasonal drift problem.
    perm = rng.permutation(number_blocks)
    block_lengths = block_lengths[perm]
    block_labels = [block_labels[i] for i in perm]

    # Compute block boundaries
    starts = np.zeros(number_blocks, dtype=int) # Beginning timestep of each block
    ends = np.zeros(number_blocks, dtype=int) # End timestep of each block
    
    start=0
    for i, length in enumerate(block_lengths):
        starts[i] = start
        end = min(start + length, num_timesteps)
        ends[i] = end
        start = end # Move to next block
        
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
        if i > 0 and block_labels[i] != block_labels[i - 1]: usable_start += guard_right # Previous block belongs to different split.
        if i < number_blocks - 1 and block_labels[i] != block_labels[i + 1]: usable_end -= guard_left # Next block belongs to different split.
        if usable_end >= usable_start + window_length: # Check remaining region is large enough
            inp = np.arange(usable_start, usable_end - window_length  + 1, stride) # Input indices are start of each window
            out = inp + center_offset # Target indices are center of each window
            split[block_labels[i]][0].append(inp) # Add to train, val, or test
            split[block_labels[i]][1].append(out)
        
    # Recombine windows from multiple blocks belonging to the same split.
    # Randomly shuffle windows within each split.
    idx_dict = {}
    for label in ['train', 'val', 'test']:
        if split[label][0]:
            inp = np.concatenate(split[label][0])
            out = np.concatenate(split[label][1])
            perm = rng.permutation(len(inp))
            idx_dict[label] = (inp[perm], out[perm])
        else: idx_dict['label'] = (np.array([], dtype=int), np.array([], dtype=int))
        
    return idx_dict

def process_window(x_win):
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

def generate_batch(x_data, y_data, inp_idx, out_idx, window_length):

    S_list = []
    S_fft_list = []
    y_list = []

    for i_inp, i_out in zip(inp_idx, out_idx):
        result = generate_sample(x_data, y_data, i_inp, i_out, window_length)
        if result is None: continue
        (S, S_fft), y = result
        S_list.append(S.numpy()) # (31, 31, 1)
        S_fft_list.append(S_fft[..., np.newaxis])
        y_list.append(y)

    return (np.array(S_list, dtype=np.float32), np.array(S_fft_list, dtype=np.float32)), np.array(y_list, dtype=np.float32)

def train_model(data, idx_dict, window_length, epochs, batch_size, model_filepath):
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse')

    # Unpack Data
    x_data = data['x_norm']
    y_data = data['y_norm']

    train_inp, train_out = idx_dict['train']
    val_inp, val_out = idx_dict['val']

    best_val_loss = np.inf
    patience = 5
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):

        # Training
        train_loss = 0.0
        num_train_batches = 0

        # Shuffle each Epoch
        perm = np.random.permutation(len(train_inp))
        train_inp = train_inp[perm]
        train_out = train_out[perm]

        for i in tqdm(range(0, len(train_inp), batch_size), desc="Training", leave=False):
            batch_inp = train_inp[i:i + batch_size]
            batch_out = train_out[i:i + batch_size]
            (S, S_fft), y = generate_batch(x_data, y_data, batch_inp, batch_out, window_length)

            loss = model.train_on_batch([S, S_fft], y)
            train_loss += loss
            num_train_batches += 1

        train_loss /= num_train_batches

        # Validation
        val_loss = 0.0
        num_val_batches = 0

        for i in tqdm(range(0, len(val_inp), batch_size), desc="Validation", leave=False):
            batch_inp = val_inp[i:i + batch_size]
            batch_out = val_out[i:i + batch_size]
            (S, S_fft), y = generate_batch(x_data, y_data, batch_inp, batch_out, window_length)

            loss = model.test_on_batch([S, S_fft], y)
            val_loss += loss
            num_val_batches += 1

        val_loss /= num_val_batches
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(model_save_filepath)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    model.save(model_filepath)

def test_model(model_filepath, data, test_idx, window_length, batch_size, scaling_factors, show=False):

    model = load_model(model_filepath)

    appliance_name = data['appliance_names'][0]
    x_data = data['x_norm']
    y_data = data['y_norm']
    y_min = scaling_factors['y_min']
    y_max = scaling_factors['y_max']

    test_inp, test_out = test_idx
    y_true_all = []
    y_pred_all = []

    # Inference Loop
    for i in range(0, len(test_inp), batch_size):
        batch_inp = test_inp[i:i+batch_size]
        batch_out = test_out[i:i+batch_size]
        (S, S_fft), y_true = generate_batch(x_data, y_data, batch_inp, batch_out, window_length)
        y_pred = model.predict_on_batch([S, S_fft])
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    # Concatenate Batches
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    # Normalized Metrics
    mse_norm = np.mean((y_pred - y_true) ** 2)
    rmse_norm = np.sqrt(mse_norm)

    # Denormalize
    y_true_denorm = y_true * (y_max - y_min) + y_min
    y_pred_denorm = y_pred * (y_max - y_min) + y_min
    mse_denorm = np.mean((y_pred_denorm - y_true_denorm) ** 2)
    rmse_denorm = np.sqrt(mse_denorm)
    abs_error = np.abs(y_pred_denorm - y_true_denorm)
    mae = np.mean(abs_error)
    avg_true = np.mean(y_true_denorm)
    eacc = 1 - (np.sum(abs_error) / (2 * np.sum(y_true_denorm)))

    # Print Results
    if show:
        print(f"\nResults for {appliance_name}")
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
        "y_pred": y_pred_denorm,
    }

# Characteristics that correlate with disaggregation performance:
# Duty Cycle, Mean Power when On, State Separability, Power Variability, Sparsity, Switching Frequency, Energy Contribution
def nilm_difficulty_score(app_power):
    app_power = np.asarray(app_power).squeeze()

    # Handle All-Zero Profiles
    nonzero = app_power[app_power > 0]
    if nonzero.size == 0: return 0.0, None

    # Determine ON Threshold
    on_threshold = np.percentile(nonzero, 10)
    on = app_power > on_threshold
    duty_cycle = np.mean(on)

    if np.any(on):
        on_power = app_power[on]
        mean_on_power = np.mean(on_power)
        cv_on = np.std(on_power) / (mean_on_power + 1e-6)
    else:
        mean_on_power = 0
        cv_on = 1

    # Number of ON/OFF transitions
    transitions = np.sum(np.abs(np.diff(on.astype(int))))
    switch_rate = transitions / len(app_power)

    # Feature Scores
    power_score = np.clip(mean_on_power / 1500, 0, 1)

    # Best around 20% duty cycle
    duty_score = np.clip(1 - abs(duty_cycle - 0.2) / 0.2, 0, 1)

    # Lower variance while ON is better
    stability_score = np.exp(-cv_on)

    # Reward a moderate amount of switching.
    # Zero switching and extremely frequent switching are both undesirable.
    optimal_switch_rate = 0.005
    switch_score = np.exp(-((switch_rate - optimal_switch_rate) / optimal_switch_rate) ** 2)

    score = (
            0.30 * power_score +
            0.25 * stability_score +
            0.20 * duty_score +
            0.25 * switch_score
    )

    features = {
        "on_threshold": on_threshold,
        "mean_on_power": mean_on_power,
        "duty_cycle": duty_cycle,
        "cv_on": cv_on,
        "transitions": transitions,
        "switch_rate": switch_rate,
        "score": score
    }

    return score, features

def select_target_appliances(n):
    scores = []
    for appliance in appliance_names:
        data = load_data(
            ampds_filepath,
            T_limit=T_limit,
            target_appliances=[appliance])
        score, features = nilm_difficulty_score(data['y'])
        scores.append((score, appliance))
    scores.sort(reverse=True)
    target_appliances = [appliance for score, appliance in scores[:n]]
    return target_appliances

if __name__ == '__main__':

    data = load_data(ampds_filepath, T_limit=T_limit)
    appliance_names = data["appliance_names"]

    #target_appliances = select_target_appliances(n=3)
    # target_appliances = ['EBE', 'RSE', 'FRE']
    # EBE: 0.39
    # RSE: 0.99
    # FRE: 0.079058
    # Scoring function doesn't do a good job predicting the best EACC
    target_appliances = appliance_names

    # Train One Model per Appliance
    results = {}
    for appliance in target_appliances:

        # Load Data for This Appliance Only
        data = load_data(ampds_filepath, T_limit=T_limit, target_appliances=[appliance])

        # Precompute Train/Validation/Test Splits
        idx_dict = precompute_indices(
            num_timesteps=data['num_timesteps'],
            window_length=window_length,
            stride=stride,
            train_val_test_split=train_test_val_split,
            number_blocks=42)

        # Save Each Appliance Model Separately
        model_filepath = os.path.join(results_dir, f"nilm_cnn_{appliance}.keras")

        # Train
        print(f"\n{'=' * 80}")
        print(f"Training model for: {appliance}")
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
        results[appliance] = test_model(
            model_filepath=model_filepath,
            data=data,
            test_idx=idx_dict['test'],
            window_length=window_length,
            batch_size=batch_size,
            scaling_factors=data['scaling_factors'],
            show=True)