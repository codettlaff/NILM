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

# Helper Functions

def read_pickle(filepath): 
    with open(filepath, 'rb') as f: return pickle.load(f)

def write_pickle(data, filepath): 
    with open(filepath, 'wb') as f: pickle.dump(data,f)
    
# Data Processing

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
