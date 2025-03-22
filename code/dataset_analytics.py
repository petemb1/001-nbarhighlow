# dataset_analytics.py
# To Run:
# python dataset_analytics.py --config config.yaml
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # stops plots from displaying
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataset import load_and_split_data, config  # Import necessary functions and config

def analyze_data(df, ticker, output_dir):
    """Analyzes and visualizes data for a single ticker.

    Args:
        df (pd.DataFrame): DataFrame for a single ticker.
        ticker (str): Ticker symbol.
        output_dir (str): Directory to save plots.
    """

    print(f"\n--- Analyzing Data for Ticker: {ticker} ---")

    # --- 1. Data Snippets (Head and Tail) ---
    print("\n--- Data Snippet (Head) ---")
    print(df.head())
    print("\n--- Data Snippet (Tail) ---")
    print(df.tail())

    # --- 2. Descriptive Statistics ---
    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    # --- 3. Target Value Distribution ---
    print("\n--- Target Value Distribution ---")
    target_counts = df['target'].value_counts()
    print(target_counts)

    # Plot target distribution and save
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title(f'Target Value Distribution - {ticker}')
    plt.savefig(os.path.join(output_dir, f'{ticker}_target_distribution.png'))
    plt.close()  # Close the figure to free memory


    # --- 4. Correlation Analysis ---
    print("\n--- Correlation Analysis ---")
    # Calculate correlation, handling potential errors
    try:
        # --- FIX: Select only numeric columns before calculating correlation ---
        numeric_df = df.select_dtypes(include=np.number)
        correlation_matrix = numeric_df.corr()
        # --------------------------------------------------------------------
        print(correlation_matrix)

        # Plot correlation heatmap and save
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix - {ticker}')
        plt.savefig(os.path.join(output_dir, f'{ticker}_correlation_matrix.png'))
        plt.close() # Close the figure

    except Exception as e:
        print(f"Error calculating or plotting correlation: {e}")

    # --- 5. Time Series Plots (All Features, including 'target') ---
    for column in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[column])  # Use the index (Date) as the time variable.
        plt.title(f'{column} Time Series - {ticker}')
        plt.xlabel('Time (Date)')
        plt.ylabel(column)
        plt.savefig(os.path.join(output_dir, f'{ticker}_{column}_timeseries.png'))
        plt.close() # Close the figure


    # --- 6. Distribution Plots (Histograms - All Features) ---
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)  # Use kde=True for a Kernel Density Estimate
        plt.title(f'Distribution of {column} - {ticker}')
        plt.savefig(os.path.join(output_dir, f'{ticker}_{column}_distribution.png'))
        plt.close() # Close the figure

    # --- 7. Check for missing values ---
    print("\n--- Missing Values ---")
    print(df.isnull().sum())


    # --- 8. Stationarity Test (Augmented Dickey-Fuller Test)  ---
    from statsmodels.tsa.stattools import adfuller

    print("\n--- Stationarity Test (Augmented Dickey-Fuller) ---")
    for column in df.columns:
      if column != "file" and df[column].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8]:
        try:
            result = adfuller(df[column])
            print(f'ADF Statistic for {column}: {result[0]}')
            print(f'p-value for {column}: {result[1]}')
            print('Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value}')
        except ValueError as e:
             print(f"Error performing ADF test on {column}: {e}")
      else:
         print(f"Skipping ADF test for non-numeric or string column: {column}")


def main(config_path):
    """Main function to load data and perform analysis."""

    # Load configuration -- No need to reload, use imported config.

    # --- FIX: Access model name correctly ---
    output_dir = config["model"]["model_name"] + "_analytics"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data DataFrames
    train_df, validation_df, test_df = load_and_split_data(config)

    # Combine all dataframes for a complete analysis
    all_data_df = pd.concat([train_df, validation_df, test_df])

    # Group by ticker and analyze
    for ticker, ticker_data in all_data_df.groupby('file'):
        analyze_data(ticker_data, ticker, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Analytics Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)