#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import pickle
import pandas as pd
from multiprocessing import Pool
import yaml
#Import here to prevent issues if not installed.
import pyunicorn
from pyunicorn import timeseries
import contextlib
# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def make_visibility_graph(input_):
    df, feature, time_step, dataset_type, ticker = input_
    vg_dir = os.path.join(config['paths']['vg_dir'], dataset_type, feature)
    os.makedirs(vg_dir, exist_ok=True)
    vg_file = os.path.join(vg_dir, f'{ticker}.pickle')

    vgs = {}
    printed_ts = False  # Flag to track if we've printed the time series
    num_vgs_visualized = 0 #Counter for visualised data
    for date in df.index:
        date_str = str(date)
        iloc = df.index.get_loc(date) + 1
        if iloc < time_step:
            continue

        # Get the raw OHLCV data for the window
        df_window = df.iloc[iloc-time_step:iloc].copy()

        # Feature engineering *within* make_visibility_graph, *before* VG creation
        if feature in ['bar_shape', 'bar_range', 'bar_overlap']:
            # These are the engineered features.  Calculate them.
            if feature == 'bar_shape':
                df_window.loc[:, 'bar_shape'] = (df_window['close'] - df_window['open']) / (df_window['high'] - df_window['low']).replace(0, 0.0001)
            elif feature == 'bar_range':
                df_window.loc[:, 'bar_range'] = (df_window['high'] - df_window['low']) / df_window['close']
            elif feature == 'bar_overlap':
                df_window.loc[:, 'prev_high'] = df_window['high'].shift(1)
                df_window.loc[:, 'prev_low'] = df_window['low'].shift(1)
                df_window.loc[:, 'bar_overlap'] = (df_window[['high', 'prev_high']].min(axis=1) - df_window[['low', 'prev_low']].max(axis=1)) / (df_window['high'] - df_window['low']).replace(0, 0.0001)
                df_window.drop(['prev_high', 'prev_low'], axis=1, inplace=True)  # Drop temporary columns
                df_window.fillna(0, inplace=True) #Drop introduced NaN
                if df_window.empty:
                  continue #Skip if the window is now empty
            time_series = df_window[feature]  # Now access the feature *after* calculation

        else: #For the original features like open, high, low, close.
            # Access directly
            time_series = df.iloc[iloc-time_step:iloc][feature]


        if len(set(time_series.values)) == 1:
            #print(f"Skipping date {date_str} due to constant values in time series.")  # Optional
            continue

        if not printed_ts:
            print(f"  Sample time series values (first 5) for {dataset_type}, {ticker}, {feature}: {time_series.values[:5]}")
            printed_ts = True

        # Suppress pyunicorn output using contextlib
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            net = timeseries.visibility_graph.VisibilityGraph(time_series.values)
        vgs[date_str] = net.adjacency

                # --- Visualization (Limited) ---
        if num_vgs_visualized < 3:  # Only visualize the first 3 VGs per ticker/feature
            print(f"    Visibility Graph for {dataset_type}, {ticker}, {feature}, date: {date_str}:")
            print(f"      Number of nodes: {net.adjacency.shape[0]}")  # Number of nodes
            print(f"      Adjacency matrix (snippet):\n{net.adjacency[:5, :5]}")  # Snippet of the matrix
            print(f"      Degree sequence (first 10): {net.degree()[:10]}") # First 10 degrees
            num_vgs_visualized += 1

    with open(vg_file, 'wb') as fp:
        pickle.dump(vgs, fp)
    print(f"    Saved VG data to {vg_file}") #Added indent

def process_dataframes():
    # Load dataframes using load_dataframe
    from dataset import load_dataframe
    train_df = load_dataframe('train_df')
    validation_df = load_dataframe('validation_df')
    test_df = load_dataframe('test_df')

    if train_df is None or validation_df is None or test_df is None:
        print("ERROR: Could not load one or more dataframes.  Exiting.")
        return

    dataframes = {
        'train': train_df,
        'validation': validation_df,
        'test': test_df
    }

    for dataset_type in config['vg_processing']['datasets']:
        print(f"Calculating visibility relations for dataset type: {dataset_type}")
        if dataset_type not in dataframes:
            print(f"WARNING: Dataset type '{dataset_type}' not found. Skipping.")
            continue

        df = dataframes[dataset_type]
        tickers = df['file'].unique().tolist()

        for ticker in tickers:
            print(f"Processing {dataset_type} data for ticker: {ticker}")
            df_ticker = df[df['file'] == ticker]
            args_list = []
            for feature in config['data']['features']:
                if feature in df_ticker.columns:
                    args_list.append((df_ticker, feature, config['data']['time_step'], dataset_type, ticker))
                else:
                    print(f"Feature {feature} not found.")
            # Use multiprocessing
            with Pool() as pool: # Use a context manager for the Pool
                pool.map(make_visibility_graph, args_list)


if __name__ == '__main__':
    process_dataframes()  # No command-line arguments