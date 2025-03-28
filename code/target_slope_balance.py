#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
from sklearn.linear_model import LinearRegression
import itertools
import matplotlib.pyplot as plt # Added for plotting the best distribution
import seaborn as sns          # Added for plotting

# --- Load dataset.py functions and config ---
# Assuming dataset.py is in the same directory or accessible via PYTHONPATH
try:
    from dataset import load_and_split_data, config
except ImportError:
    print("ERROR: Could not import from dataset.py. Make sure it's in the same directory or your PYTHONPATH is set correctly.")
    sys.exit(1)
# -------------------------------------------

# --- Modified calculate_target function to accept parameters ---
def calculate_target_with_params(df, n_before, n_after, long_threshold, short_threshold):
    """
    Calculates the target based on the slope of a forward/backward-looking
    log-linear regression on the 'close' price. Accepts parameters directly.

    Target values:
        0: Slope < short_threshold (Short signal)
        1: short_threshold <= Slope <= long_threshold (Neutral signal)
        2: Slope > long_threshold (Long signal)
    """
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
            continue # Skip this iteration if non-positive values exist

        # Log transform the close prices
        log_close = np.log(window_close)

        # Prepare data for linear regression
        X = np.arange(len(log_close)).reshape(-1, 1)
        y = log_close

        if len(X) < 2: # Need at least 2 points for regression
            continue

        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
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

    return target
# -----------------------------------------------------------

def find_balanced_target_params(config):
    """
    Iterates through parameter ranges to find the combination that yields
    the most balanced target distribution.
    """
    print("Loading and combining all data...")
    train_df, validation_df, test_df = load_and_split_data(config)
    if train_df is None or validation_df is None or test_df is None:
        print("ERROR: Failed to load dataframes from dataset.py")
        return

    # Combine all data for analysis (optional, could focus on train_df)
    # Using only train_df might be better to avoid looking at val/test distributions
    df_full = train_df # Let's focus on the training set distribution
    # df_full = pd.concat([train_df, validation_df, test_df]) # Option to use all data
    print(f"Using data with shape: {df_full.shape}")

    # --- Define NEW Parameter Ranges for Fine-Tuning ---
    # Based on previous best: (n_before=0, n_after=5, long=0.002, short=-0.0005)

    n_before_range = range(0, 3)      # Explore 0, 1, 2 (around 0)
    n_after_range = range(3, 8)       # Explore 3, 4, 5, 6, 7 (around 5)

    # Finer threshold steps around the best found values
    threshold_step = 0.0001           # Use a smaller step (0.01%)
    positive_thresholds = np.arange(0.0010, 0.0031, threshold_step) # Explore 0.10% to 0.30%
    negative_thresholds = np.arange(-0.0015, 0, threshold_step)     # Explore -0.15% to -0.01%

    min_total_window = 3  # Keep the minimum window size constraint
    # -------------------------------------------------
    
    best_params = None
    min_std_dev = float('inf')
    best_distribution = None

    # Create all combinations of parameters
    param_combinations = list(itertools.product(
        n_before_range,
        n_after_range,
        positive_thresholds,
        negative_thresholds
    ))

    print(f"Total parameter combinations to test (before min size filter): {len(param_combinations)}")

    count = 0
    tested_count = 0 # Counter for combinations actually tested
    for params in param_combinations:
        n_before, n_after, long_thresh, short_thresh = params

        # --- USE THE VARIABLE IN THE CHECK ---
        # Ensures n_before + 1 + n_after >= min_total_window
        if n_before + n_after < (min_total_window - 1):
            continue # Skip combinations where the total window size is less than min_total_window
        # -------------------------------------

        count += 1
        tested_count += 1 # Increment the tested counter
        if tested_count % 100 == 0: # Print progress based on tested count
            print(f"Testing combination {tested_count}: {params}")

        # Calculate target with current parameters
        target_series = calculate_target_with_params(df_full, n_before, n_after, long_thresh, short_thresh)

        # Calculate distribution and standard deviation
        counts = target_series.value_counts().reindex([0, 1, 2], fill_value=0) # Ensure all 3 classes are present
        std_dev = counts.std()

        # Check if this combination is better
        if std_dev < min_std_dev:
            min_std_dev = std_dev
            best_params = params
            best_distribution = counts
            print(f"--- New Best Found (Std Dev: {min_std_dev:.4f}) ---")
            print(f"  Params (n_before, n_after, long_thresh, short_thresh): {best_params}")
            print(f"  Distribution:\n{best_distribution}")

    print("\n--- Best Parameter Combination Found ---")

    if best_params:
        print(f"  Parameters (n_before, n_after, long_thresh, short_thresh): {best_params}")
        print(f"  Minimum Standard Deviation of Counts: {min_std_dev:.4f}")
        print(f"  Resulting Target Distribution:\n{best_distribution}")

        # Plot the best distribution
        plt.figure(figsize=(8, 6))
        sns.barplot(x=best_distribution.index, y=best_distribution.values)
        plt.title(f'Best Target Distribution (Std Dev: {min_std_dev:.4f})\nParams: {best_params}')
        plt.xlabel('Target Class (0=Short, 1=Neutral, 2=Long)')
        plt.ylabel('Count')
        plt.savefig('best_target_distribution.png')
        print("Saved plot of best distribution to 'best_target_distribution.png'")
        plt.close()

    else:
        print("Could not find suitable parameters within the specified ranges.")


if __name__ == '__main__':
    find_balanced_target_params(config)