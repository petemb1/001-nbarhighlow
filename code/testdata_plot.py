#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
# Import model definitions
from lib.model import PriceGraph, output_layer
# Import necessary functions from dataset.py
from dataset import load_dataframe, sample_by_dates
import yaml
import json
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For formatting dates on plot

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- Helper functions copied/adapted from trainer.py ---

def load_embeddings_and_ci(data, dataset_type, train_df, validation_df, test_df): # Pass DFs
    """Loads pre-computed embeddings and CI values, handling potential errors."""
    # Dictionaries to cache loaded data: (stock, feature) -> data
    loaded_embeddings = {}
    loaded_ci = {}

    # Pre-allocate lists with None. Correct length.
    for feature in config['data']['features']:
        data[f'{feature}_ems'] = [None] * len(data['stock'])
        data[f'{feature}_cis'] = [None] * len(data['stock'])

    for i in range(len(data['stock'])):
        stock = data['stock'][i]
        day_index = data['day'][i]

        # Get the correct DataFrame based on dataset_type
        if dataset_type == 'train':
            original_df = train_df
        elif dataset_type == 'validation':
            original_df = validation_df
        elif dataset_type == 'test':
            original_df = test_df
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        # Get the date string directly from the DataFrame's index
        if day_index < len(original_df): # Check index bounds
            date = str(original_df.index[day_index])
        else:
            print(f"WARNING: day_index {day_index} out of bounds for DataFrame of length {len(original_df)}. Skipping sample.")
            continue # Skip this sample if index is bad

        for feature in config['data']['features']:
            embedding_file = os.path.join(config['paths']['struc2vec_dir'], dataset_type, feature, f"{stock}.json")
            ci_file = os.path.join(config['paths']['ci_dir'], dataset_type, feature, f"{stock}.json")

            # --- Load Embedding (if not already loaded) ---
            if (stock, feature) not in loaded_embeddings:
                if os.path.exists(embedding_file):
                    try:
                        with open(embedding_file, 'r') as f:
                            loaded_embeddings[(stock, feature)] = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"WARNING: Error loading {embedding_file}: {e}. Setting to empty.")
                        loaded_embeddings[(stock, feature)] = {}
                else:
                    print(f"WARNING: Embedding file not found: {embedding_file}")
                    loaded_embeddings[(stock, feature)] = {}

            # --- Embedding Lookup ---
            if date in loaded_embeddings.get((stock, feature), {}):
                embeddings_for_date = loaded_embeddings[(stock, feature)][date]
                try:
                    embedding_list = [embeddings_for_date[str(j)] for j in range(config['data']['time_step'])]
                    data[f'{feature}_ems'][i] = np.array(embedding_list, dtype=np.float32)
                except KeyError as e:
                     print(f"WARNING: Node key {e} not found in embeddings for {stock}, {feature}, {date}. Setting sample ems to None.")
                     data[f'{feature}_ems'][i] = None # Handle missing node keys
                except Exception as e:
                     print(f"ERROR processing embeddings for {stock}, {feature}, {date}: {e}")
                     data[f'{feature}_ems'][i] = None
            #else: # Already None

            # --- Load CI (if not already loaded) ---
            if (stock, feature) not in loaded_ci:
                if os.path.exists(ci_file):
                    try:
                        with open(ci_file, 'r') as f:
                            loaded_ci[(stock, feature)] = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"WARNING: Error loading {ci_file}: {e}. Setting to empty.")
                        loaded_ci[(stock, feature)] = {}
                else:
                    print(f"WARNING: CI file not found: {ci_file}")
                    loaded_ci[(stock, feature)] = {}

            # --- CI Lookup ---
            if date in loaded_ci.get((stock, feature), {}):
                ci_data_for_date = loaded_ci[(stock, feature)][date]
                try:
                     # Convert keys to integers for sorting, get values as floats
                    ci_values = [float(ci_data_for_date[str(k)]) for k in sorted(ci_data_for_date.keys(), key=int) if str(k) in ci_data_for_date] # Make sure keys are str
                    if len(ci_values) == config['data']['time_step']: # Ensure correct length
                        data[f'{feature}_cis'][i] = np.array(ci_values, dtype=np.float32)
                    else:
                        print(f"WARNING: Incorrect number of CI values for {stock}, {feature}, {date}. Expected {config['data']['time_step']}, got {len(ci_values)}. Setting sample cis to None.")
                        data[f'{feature}_cis'][i] = None
                except (KeyError, ValueError) as e:
                    print(f"WARNING: Error processing CI keys/values for {stock}, {feature}, {date}: {e}. Setting sample cis to None.")
                    data[f'{feature}_cis'][i] = None
            #else: # Already None

    return data

def get_batch(data, start_index, batch_size):
    end_index = min(start_index + batch_size, len(data['stock']))
    batch = {}
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            batch[key] = data[key][start_index:end_index]
        elif isinstance(data[key], list): # Handle lists (embeddings/CI)
            batch[key] = data[key][start_index:end_index]
        else:
            batch[key] = data[key]
    return batch

def to_variable(data, num_features, device): # Pass num_features and device
    var = []
    batch_size = len(data['stock']) # Get batch size from data

    for i in range(num_features):
        feature_name = config['data']['features'][i]
        var_dict = {
            'ems': None,
            'ys': None, # Initialize ys as None
            'cis': None
        }

        # Process ys
        if data[f'{feature_name}_ys'] is not None:
            ys_batch = data[f'{feature_name}_ys']
            # Ensure ys_batch is a list of numpy arrays before stacking
            if isinstance(ys_batch, list) and all(isinstance(item, np.ndarray) for item in ys_batch):
                try:
                    ys_stacked = np.stack(ys_batch) # Stack list of arrays
                    var_dict['ys'] = torch.tensor(ys_stacked, dtype=torch.float32).unsqueeze(-1).to(device)
                except ValueError as e:
                    print(f"Error stacking ys for {feature_name}. Check array shapes. Error: {e}")
                    # Create a tensor of zeros if stacking fails, matching expected shape
                    var_dict['ys'] = torch.zeros((batch_size, config['data']['time_step'], 1), dtype=torch.float32).to(device)
            elif isinstance(ys_batch, np.ndarray) and ys_batch.ndim == 2: # If already a 2D numpy array
                 var_dict['ys'] = torch.tensor(ys_batch, dtype=torch.float32).unsqueeze(-1).to(device)
            else:
                 print(f"Warning: Invalid format for ys data for {feature_name}. Expected list of NumPy arrays or 2D NumPy array.")
                 var_dict['ys'] = torch.zeros((batch_size, config['data']['time_step'], 1), dtype=torch.float32).to(device)


        # Process ems (Handle None values and stack)
        if data[f'{feature_name}_ems'] is not None:
            ems_list = [item if item is not None else np.zeros((config['data']['time_step'], config['model']['embedding_dim']), dtype=np.float32)
                        for item in data[f'{feature_name}_ems']]
            if ems_list: # Check if list is not empty after processing Nones
                try:
                    ems_stacked = np.stack(ems_list)
                    var_dict['ems'] = torch.tensor(ems_stacked, dtype=torch.float32).to(device)
                except ValueError as e:
                     print(f"Error stacking ems for {feature_name}. Check array shapes. Error: {e}")
                     # Create a tensor of zeros on error, matching expected shape
                     var_dict['ems'] = torch.zeros((batch_size, config['data']['time_step'], config['model']['embedding_dim']), dtype=torch.float32).to(device)

        # Process cis (Handle None values and stack)
        if data[f'{feature_name}_cis'] is not None:
            cis_list = [item if item is not None else np.zeros(config['data']['time_step'], dtype=np.float32)
                        for item in data[f'{feature_name}_cis']]
            if cis_list:
                try:
                   cis_stacked = np.stack(cis_list)
                   var_dict['cis'] = torch.tensor(cis_stacked, dtype=torch.float32).to(device)
                except ValueError as e:
                     print(f"Error stacking cis for {feature_name}. Check array shapes. Error: {e}")
                     # Create tensor of zeros on error
                     var_dict['cis'] = torch.zeros((batch_size, config['data']['time_step']), dtype=torch.float32).to(device)

        var.append(var_dict)
    return var

    # --- Main Processing Logic ---

def main(config_path):
    """Main function to load data, model, make predictions, and plot."""
    # Load configuration again within main scope if needed, or ensure global config is used
    # Using the globally loaded config for simplicity here
    global config

    # --- Setup ---
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    output_dir = config["model"]["model_name"] + "_analytics"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # --- Load Test DataFrame ---
    print("Loading test dataframe...")
    test_df = load_dataframe('test_df')
    if test_df is None:
        print("ERROR: test_df.pkl not found or failed to load. Exiting.")
        sys.exit(1)
    print("Test dataframe loaded.")

    # --- Sample Test Data ---
    print("Sampling test data...")
    test_data = sample_by_dates(test_df, config['data']['time_step'])
    if test_data['stock'].size == 0:
        print("ERROR: No valid samples generated from test data. Exiting.")
        sys.exit(1)
    print("Test data sampled.")

    # --- Load Embeddings & CI for Test Data ---
    print("Loading embeddings and CI for test data...")
    # Need to pass the original test_df for date index lookup
    test_data = load_embeddings_and_ci(test_data, 'test', test_df, test_df, test_df)
    print("Embeddings and CI loaded for test data.")

    # --- Load Trained Model ---
    model_path = config['testing']['model_path']
    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: Trained model path not found or not specified in config: {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    # Reload config from checkpoint to ensure consistency (optional but good practice)
    loaded_config = checkpoint.get('config', config) # Fallback to current config if not saved

    # Determine feature size from loaded data
    feature_size = len(test_data['close_ys'][0]) if test_data['close_ys'].size > 0 else 0
    num_features = len(loaded_config['data']['features'])
    print(f"Model configured with: Num Features={num_features}, Feature Size={feature_size}")


    emtree = PriceGraph(feature_size, loaded_config['model']['hidden_size'], loaded_config['data']['time_step'], loaded_config['model']['dropout_ratio'], num_features).to(device)
    output = output_layer(last_hidden_size=loaded_config['model']['hidden_size'], output_size=4).to(device) # Assuming 3 classes (0,1,2)

    try:
        emtree.load_state_dict(checkpoint['emtree_state_dict'])
        output.load_state_dict(checkpoint['output_state_dict'])
    except KeyError as e:
        print(f"ERROR: Model state dictionary keys not found in checkpoint: {e}")
        sys.exit(1)
    except RuntimeError as e:
         print(f"ERROR: Could not load model state dictionaries. Mismatch in layers/parameters? Error: {e}")
         sys.exit(1)


    emtree.eval()
    output.eval()
    print("Model loaded successfully.")

    # --- Make Predictions ---
    print("Making predictions on test data...")
    all_predictions = []
    all_true_targets = []
    batch_size = loaded_config['model']['batch_size'] # Use batch size from loaded config

    with torch.no_grad():
        for batch_idx in range(0, len(test_data['stock']), batch_size):
            batch_data = get_batch(test_data, batch_idx, batch_size)
            var = to_variable(batch_data, num_features, device) # Pass num_features and device

            emtree_out = emtree(var)
            logits = output(emtree_out)
            # --- Prediction Logic (Assuming target is 0, 1, 2) ---
            batch_predictions = torch.argmax(logits, dim=1) # Prediction is class index 0, 1, or 2
            all_predictions.extend(batch_predictions.cpu().tolist())
            all_true_targets.extend(batch_data['target'].tolist()) # Collect true targets

    print("Predictions generated.")

    # --- Prepare Data for Plotting ---
    print("Preparing data for plotting...")
    # Get the dates corresponding to the samples using the integer index
    plot_dates = test_df.index[test_data['day']]
    # Get the 'close' prices corresponding to these dates
    # Use iloc with the integer indices from test_data['day']
    plot_close = test_df['close'].iloc[test_data['day']]

    if len(plot_dates) != len(all_predictions):
         print(f"WARNING: Length mismatch between dates ({len(plot_dates)}) and predictions ({len(all_predictions)}). Plot might be inaccurate.")
         # Adjust lengths if possible, e.g., take the minimum length
         min_len = min(len(plot_dates), len(all_predictions), len(plot_close))
         plot_dates = plot_dates[:min_len]
         all_predictions = all_predictions[:min_len]
         plot_close = plot_close[:min_len]
         all_true_targets = all_true_targets[:min_len]


    plot_df = pd.DataFrame({
        'Close': plot_close.values,
        'Prediction': all_predictions,
        'TrueTarget': all_true_targets
        }, index=plot_dates)
    print("Plotting data prepared.")

    # --- Plot Results ---
    plot_results(plot_df, test_data['stock'][0], output_dir) # Assume single ticker for title

    # --- Plotting Function ---

def plot_results(plot_df, ticker, output_dir):
    """Plots Close Price vs. Predictions and saves the figure."""
    print(f"Generating plot for {ticker}...")

    fig, ax1 = plt.subplots(figsize=(15, 7)) # Create figure and primary axis

    # Plot Close Price on primary axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color=color)
    ax1.plot(plot_df.index, plot_df['Close'], color=color, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True) # Add grid for primary axis

    # Create secondary axis for Predictions
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Prediction (0=Short, 1=Neutral, 2=Long)', color=color)
    # Use step plot for categorical predictions
    ax2.step(plot_df.index, plot_df['Prediction'], color=color, where='post', label='Prediction', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    # Set y-axis limits and ticks for predictions (0, 1, 2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_ylim([-0.5, 2.5]) # Add some padding

    # Optional: Plot True Target for comparison
    # color_true = 'tab:green'
    # ax2.scatter(plot_df.index, plot_df['TrueTarget'], color=color_true, label='True Target', marker='o', s=10) # Use scatter for clarity

    # Title and Legend
    plt.title(f'{ticker} Close Price and Model Predictions')
    fig.tight_layout() # Adjust layout to prevent overlap
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')


    # Format Date Axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate() # Auto-rotate dates

    # Save the plot
    plot_filename = os.path.join(output_dir, f'{ticker}_close_vs_prediction.png')
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig) # Close the figure

# --- Main Execution Block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot test data predictions")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)