# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:07:59 2026

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
processed_data = os.path.join(data_dir, 'ampds2_processed.npz')
model_save_filepath = os.path.join(results_dir, 'nilm_cnn_model_2month.keras')

T_limit = 86400 # Two Months
train_test_val_split = [0.7, 0.15, 0.15]
window_length, stride = 30, 1
epochs = 20
batch_size = 32
target_appliances = ['DWE']

def load_data(ampds_filepath, T_limit, target_appliances=None):
    pass

def precompute_indices(num_timesteps, window_length, stride, train_val_test_split, number_blocks, seed=42):
    pass

# Returns scaling factors
def normalize_data(data, idx_dict):
    pass

def process_window(x_win):
    pass

# Saves and returns dict of {x: (S,FFT), y: target}
# Return new idx_dict for processed data
def prepare_data(data, idx_dict, save_filepath):
    pass

def generate_sample(processed_data, idx):
    pass

# Data is only loaded for a short time when batch is being fetched.
# Saves memory.
def generate_batch(processed_data_filepath, idx_list):
    pass

def build_model():
    pass

def train_model(model, processed_data_filepath, processed_data_idx_dict, epochs, batch_size, model_filepath):
    pass

def test_model(model_filepath, processed_data_filepath, processed_data_idx_dict, batch_size, scaling_factors, show=False):
    pass

if __name__ == '__main__':
    pass

