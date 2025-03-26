#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
import yaml
import yfinance as yf
from functools import partial
import pickle
from pathlib import Path
import talib  # Add this import
from sklearn.linear_model import LinearRegression

# Get the absolute path of the directory containing dataset.py
PWD = os.path.dirname(os.path.abspath(__file__))

# Load configuration
with open(os.path.join(PWD, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Make paths absolute, relative to PWD
config['paths']['data_dir'] = os.path.join(PWD, config['paths']['data_dir'])
config['paths']['vg_dir'] = os.path.join(PWD, config['paths']['vg_dir'])
config['paths']['ci_dir'] = os.path.join(PWD, config['paths']['ci_dir'])
config['paths']['struc2vec_dir'] = os.path.join(PWD, config['paths']['struc2vec_dir'])
config['paths']['dataset_dir'] = os.path.join(PWD, config['paths']['dataset_dir'])
config['paths']['models_dir'] = os.path.join(PWD, config['paths']['models_dir'])
config['paths']['temp_dir'] = os.path.join(PWD, config['paths']['temp_dir'])

def z_score(series):
    """Calculates z-score for a Pandas Series."""
    return (series - series.mean()) / series.std() if series.std() != 0 else pd.Series(np.zeros(len(series)), index=series.index)

def download_data(tickers, data_dir):
    print(f"Downloading data to: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading data for ticker: {ticker}")
        try:
            data = yf.download(ticker, period="max")
            if data.empty:
                print(f"ERROR: yfinance returned empty DataFrame for {ticker}")
                continue

            # Flatten multi-index and set 'Date' as index
            data.columns = [' '.join(col).strip().lower().replace(' ', '_') for col in data.columns.values]
            data.index.name = 'Date'
            # Don't drop NaNs here.

            file_path = os.path.join(data_dir, f"{ticker}.csv")
            print(f"Saving data to: {file_path}")
            data.to_csv(file_path)
            print(f"Downloaded and saved data for {ticker}")
        except Exception as e:
            print(f"ERROR: Failed to download or save data for {ticker}. Error: {e}")
            continue

def stock_sample(df, d, T):
    if d not in df.index:
        return None
    iloc = df.index.get_loc(d) + 1
    if iloc < T:
        return None
    if iloc - 1 < 0:
        return None

    df_window = df.iloc[iloc - T:iloc].copy()  # Get window. Include current day

    if len(df_window) < T:
        return None

    # --- Feature Engineering (within the window) ---
    # Use .loc for all assignments to avoid SettingWithCopyWarning
    #df_window.loc[:, 'rsi'] = talib.RSI(df_window['close'], timeperiod=14)  # RSI with period 14
    #df_window.loc[:, 'slowk'], _ = talib.STOCH(
    #    df_window['high'], df_window['low'], df_window['close'],
    #    fastk_period=14, slowk_period=1, slowd_period=3
    #)  # STOCH with correct periods
    df_window.loc[:, 'bar_range'] = (df_window['high'] - df_window['low']) / df_window['close']
    df_window.loc[:, 'bar_shape'] = (df_window['close'] - df_window['open']) / (df_window['high'] - df_window['low']).replace(0, 0.0001) #or close - low
    df_window.loc[:, 'bar_close_mid'] = (df_window['close'] - (df_window['high'] + df_window['low']) / 2) / (df_window['high'] - df_window['low']).replace(0, 0.0001)
    df_window.loc[:, 'prev_high'] = df_window['high'].shift(1)
    df_window.loc[:, 'prev_low'] = df_window['low'].shift(1)
    df_window.loc[:, 'bar_overlap'] = (df_window[['high', 'prev_high']].min(axis=1) - df_window[['low', 'prev_low']].max(axis=1)) / (df_window['high'] - df_window['low']).replace(0, 0.0001)
    #df_window.loc[:, 'bar_close_ema9'] = df_window['close'] - talib.EMA(df_window['close'], timeperiod=9)
    df_window.drop(['prev_high', 'prev_low'], axis=1, inplace=True)  # Drop temp columns

    df_window.fillna(0, inplace=True)  # Fill NaNs introduced by feature engineering with 0.

    if df_window.empty:
        return None
    # --- CRITICAL: Check window length *AFTER* feature engineering ---
    if len(df_window) < T:
        return None

    xss = {}
    for xi in config['data']['features']:
        if xi in df_window.columns:
            yz = np.array(z_score(df_window[xi]))  # Calculate z-score
            if np.isnan(yz).any():  # Check for NaN values after z-score
                return None
            xss[f'{xi}_ys'] = yz  # Store NumPy array

        else:
            print(f"Column {xi} not found")
            return None

    file_ = df['file'].iloc[0]
    target_value = df['target'].iloc[iloc - 1]  # Target from original df
    return file_, iloc - 1, target_value, xss #return the index position

def load_and_split_data(config):
    train_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    test_df = pd.DataFrame()

    data_dir = config['paths']['data_dir']
    print(f"Loading data from directory: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return train_df, validation_df, test_df  # Return empty DataFrames

    for file_ in os.listdir(data_dir):
        if not file_.endswith(".csv"):
            print(f"Skipping non-CSV file: {file_}")
            continue

        file_path = os.path.join(data_dir, file_)
        print(f"Attempting to load: {file_path}")
        try:
            # Load, parse dates, set index:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
            print(f"Loaded raw data for {file_}:\n{df.head()}\n{df.tail()}\n{df.dtypes}")

            if df.empty:
                print(f"ERROR: DataFrame is empty after loading {file_path}")
                continue

            # --- Simplification: Remove Ticker Suffix from Initial Load ---
            # Instead of 'close_spy', we just have 'close', etc.
            df.columns = [col.replace(f"_{file_[:-4].lower()}", "") for col in df.columns]

            # Add filename as a column
            df['file'] = file_[:-4]  # Store filename without extension
            ticker = file_[:-4].lower()

            # --- Feature Engineering: Add back in here. ---
            #df.loc[:, 'rsi'] = talib.RSI(df['close'], timeperiod=14)
            #df.loc[:, 'slowk'], _ = talib.STOCH(
            #    df['high'], df['low'], df['close'],
            #    fastk_period=14, slowk_period=1, slowd_period=3
            #)
            df.loc[:, 'bar_range'] = (df['high'] - df['low']) / df['close']
            df.loc[:, 'bar_shape'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.0001)
            df.loc[:, 'bar_close_mid'] = (df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low']).replace(0, 0.0001)
            df.loc[:, 'prev_high'] = df['high'].shift(1)
            df.loc[:, 'prev_low'] = df['low'].shift(1)
            df.loc[:, 'bar_overlap'] = (df[['high', 'prev_high']].min(axis=1) - df[['low', 'prev_low']].max(axis=1)) / (df['high'] - df['low']).replace(0, 0.0001)
            #df.loc[:, 'bar_close_ema9'] = df['close'] - talib.EMA(df['close'], timeperiod=9)
            df.drop(['prev_high', 'prev_low'], axis=1, inplace=True)

            # --- Fill NaN values with 0 instead of dropping ---
            print(f"DataFrame size BEFORE fillna (feature engineering): {df.shape}")  # Debug print
            df.fillna(0, inplace=True)  # Fill NaN with 0 *before* splitting
            print(f"DataFrame size AFTER fillna (feature engineering): {df.shape}")  # Debug print
            print(f"DataFrame after feature engineering and filling NaNs:\n{df.head()}\n{df.tail()}")

            # --- Split data based on date ranges ---
            train_data = df[(df.index >= config['data']['train_start_date']) & (df.index <= config['data']['train_end_date'])].copy()
            validation_data = df[(df.index >= config['data']['validation_start_date']) & (df.index <= config['data']['validation_end_date'])].copy()
            test_data = df[(df.index >= config['data']['test_start_date']) & (df.index <= config['data']['test_end_date'])].copy()

            # --- Target variable calculation ---
            print(f"Calculating target for train_data (size: {train_data.shape})")
            train_data['target'] = calculate_target(train_data, config['data']['prediction_window'], 'close')  # Use simplified column name
            print(f"Calculating target for validation_data (size: {validation_data.shape})")
            validation_data['target'] = calculate_target(validation_data, config['data']['prediction_window'], 'close')
            print(f"Calculating target for test_data (size: {test_data.shape})")
            test_data['target'] = calculate_target(test_data, config['data']['prediction_window'], 'close')


            print(f"Train data (with engineered features & target):\n{train_data.head()}\n{train_data.tail()}")
            print(f"Validation data (with engineered features & target):\n{validation_data.head()}\n{validation_data.tail()}")
            print(f"Test data (with engineered features & target):\n{test_data.head()}\n{test_data.tail()}")


            train_df = pd.concat([train_df, train_data])
            validation_df = pd.concat([validation_df, validation_data])
            test_df = pd.concat([test_df, test_data])

        except Exception as e:
            print(f"ERROR: Failed to load or process {file_path}. Error: {e}")
            continue

    return train_df, validation_df, test_df

def calculate_target(df, prediction_window, close_col): # close_col is not used, but kept for signature consistency
    """
    Calculates the target based on the slope of a forward/backward-looking
    log-linear regression on the 'close' price.

    Target values:
        0: Slope < short_threshold (Short signal)
        1: short_threshold <= Slope <= long_threshold (Neutral signal)
        2: Slope > long_threshold (Long signal)
    """
    n_before = config['data']['target_n_before']
    n_after = config['data']['target_n_after']
    long_threshold = config['data']['target_long_slope_threshold']
    short_threshold = config['data']['target_short_slope_threshold']

    # Ensure window sizes are non-negative
    if n_before < 0 or n_after < 0:
        raise ValueError("n_before and n_after must be non-negative")
    if n_before == 0 and n_after == 0:
        raise ValueError("n_before and n_after cannot both be zero")


    target = pd.Series(1, index=df.index, dtype='int8')  # Initialize all to neutral (1)

    # Create a numpy array for faster access to close prices
    close_prices = df['close'].values

    for i in range(n_before, len(df) - n_after): # Iterate through valid indices
        # Define the window indices
        start_idx = i - n_before
        end_idx = i + n_after + 1 # +1 to include the end point

        # Extract the 'close' prices for the window
        window_close = close_prices[start_idx:end_idx]

        # Check for non-positive values before log transformation
        if np.any(window_close <= 0):
            # print(f"Skipping index {i} due to non-positive close price in window.") # Optional debug
            continue # Skip this iteration if non-positive values exist

        # Log transform the close prices
        log_close = np.log(window_close)

        # Prepare data for linear regression
        # X is the time index within the window [0, 1, ..., window_length-1]
        X = np.arange(len(log_close)).reshape(-1, 1)
        y = log_close  # y is the log-transformed price

        # Check if we have enough points for regression (at least 2)
        if len(X) < 2:
            continue

        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)

            # Get the slope (rate of change in log-price ~ rate of return)
            slope = model.coef_[0]

            # Assign target based on slope and thresholds
            if slope > long_threshold:
                target.iloc[i] = 2  # Long
            elif slope < short_threshold:
                target.iloc[i] = 0  # Short
            # else: target remains 1 (Neutral)

        except Exception as e:
            print(f"Error during regression at index {i}: {e}") # Handle potential errors
            continue


    # Points at the very beginning/end where a full window isn't possible
    # remain as the initialized neutral value (1).

    return target

def sample_by_dates(df, T):
    dates = df.index.tolist()
    fds = [(df, d, T) for d in dates]
    pool = Pool()
    samples = pool.starmap(stock_sample, fds, chunksize=1)  # Use starmap
    pool.close()
    pool.join()

    samples = [s for s in samples if s is not None]  # Filter out None values

    if not samples:  # If *all* samples are None, return empty dicts
        return {
            'stock': np.array([]),
            'day': np.array([]),
            'target': np.array([], dtype='int8'),
            **{f'{xi}_ys': np.array([]) for xi in config['data']['features']},
            **{f'{xi}_ems': np.array([]) for xi in config['data']['features']},  # Placeholder
            **{f'{xi}_cis': np.array([]) for xi in config['data']['features']}  # Placeholder
        }

    stocks, days, targets, xss_list = zip(*samples)
    #print(f"Days before conversion: {days[:5]}")  # Debug print
    data_dict = {
        'stock': np.array(stocks),
        'day': np.array(days), # return the index
        'target': np.array(targets)
    }

    # --- Corrected Feature Aggregation ---
    for xi in config['data']['features']:
        # Collect the time series data for each feature
        feature_data = []
        for xss in xss_list:
            if f'{xi}_ys' in xss:
                # Pad to ensure all arrays have length T
                padded_array = np.pad(xss[f'{xi}_ys'], (0, T - len(xss[f'{xi}_ys'])), 'constant', constant_values=(np.nan))
                feature_data.append(padded_array)

        if feature_data:
          data_dict[f'{xi}_ys'] = np.array(feature_data)  # Now a 2D NumPy array
        else:
          data_dict[f'{xi}_ys'] = np.array([])


        # Placeholders for embeddings and CI (filled in later)
        data_dict[f'{xi}_ems'] = None  # Placeholder
        data_dict[f'{xi}_cis'] = None  # Placeholder

    return data_dict

def save_dataframe(df, filename):
    """Saves a DataFrame to a pickle file."""
    file_path = os.path.join(config['paths']['dataset_dir'], f"{filename}.pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved DataFrame to {file_path}")

def load_dataframe(filename):
    """Loads a DataFrame from a pickle file."""
    file_path = os.path.join(config['paths']['dataset_dir'], f"{filename}.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"ERROR: DataFrame file not found: {file_path}")
        return None

if __name__ == '__main__':
    # Download data only if download_data is True
    if config['data']['download_data'] == 'yes':  # Corrected comparison
        data_dir = config['paths']['data_dir']
        print("download_data is True. Downloading data.")
        download_data(config['data']['tickers'], data_dir)
    else:
        print("Skipping data download as per config file (download_data is False).")

    train_df, validation_df, test_df = load_and_split_data(config)

    # --- Save the *DataFrames* (with raw features and targets) ---
    save_dataframe(train_df, 'train_df')
    save_dataframe(validation_df, 'validation_df')
    save_dataframe(test_df, 'test_df')
    print("Raw data DataFrames (with engineered features) saved.")

    # --- The sampling/z-scoring happens *within* the trainer, NOT here ---
    # We do *not* create the final datasets here anymore.
    # train_data = sample_by_dates(train_df, config['data']['time_step'])
    # validation_data = sample_by_dates(validation_df, config['data']['time_step'])
    # test_data = sample_by_dates(test_df, config['data']['time_step'])

    # --- We do NOT inspect or save the *sampled* data here. ---
    # def inspect_dataset(data, name):
    #     ...
    # inspect_dataset(...)
    # ...