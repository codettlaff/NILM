# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:13:38 2026

@author: codett
"""
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

def read_pickle(filepath: str): 
    with open(filepath, 'rb') as f: return pickle.load(f)

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
    
def plot_loss_history(metadata_filepath):
    metadata = read_pickle(metadata_filepath)
    plt.plot(metadata['train_loss_history'], label='Train')
    plt.plot(metadata['val_loss_history'], label='Validation')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    ukdale1_DWE_model = os.path.join(models_dir, 'ukdale1', 'ukdale1_DWE', 'model.keras')
    ukdale1_DWE_metadata = os.path.join(models_dir, 'ukdale1', 'ukdale1_DWE', 'metadata.pkl')
    
    # Debug Issue loading models that were trained on kamiak
    model = tf.keras.models.load_model(ukdale1_DWE_model, compile=False)
    
    plot_loss_history(ukdale1_DWE_metadata)
    
    print('')