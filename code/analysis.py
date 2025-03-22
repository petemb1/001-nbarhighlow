import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def download_spy_data(start_date="1900-01-01", end_date=None):
    """
    Downloads SPY OHLCV data from yfinance.

    Args:
        start_date (str, optional): Start date for data retrieval. Defaults to "1900-01-01".
        end_date (str, optional): End date for data retrieval. Defaults to None (today).

    Returns:
        pandas.DataFrame: DataFrame containing the SPY OHLCV data, or None if download fails.
    """
    try:
        spy_data = yf.download('SPY', start=start_date, end=end_date)
        if spy_data.empty:
            print("No data downloaded. Check the date range or ticker symbol.")
            return None
        return spy_data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def calculate_features(df):
    """
    Calculates the bar shape, bar range, and bar overlap features.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLCV data with 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pandas.DataFrame: DataFrame with the added features. Returns None if input DataFrame is invalid.
    """
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        print("Error: DataFrame must contain 'Open', 'High', 'Low', and 'Close' columns.")
        return None

    df_with_features = df.copy() # Create a copy to avoid modifying the original DataFrame in place.

     # Rename columns to lowercase
    df_with_features.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
    df = df_with_features.copy() # Create another copy to avoid modifying the original DataFrame in place.

    # Avoid division by zero by replacing 0 range with a very small number
    df.loc[:, 'bar_shape'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.0001)
    df.loc[:, 'bar_range'] = (df['high'] - df['low']) / df['close']
    df.loc[:, 'prev_high'] = df['high'].shift(1)
    df.loc[:, 'prev_low'] = df['low'].shift(1)

    # Calculate bar overlap
    df.loc[:, 'highs'] = df[['high', 'prev_high']].min(axis=1)
    df.loc[:, 'lows'] = df[['low', 'prev_low']].max(axis=1)
    df.loc[:, 'bar_overlap'] = (df['highs'] - df['lows']) / (df['high'] - df['low']).replace(0, 0.0001)

    df.drop(['prev_high', 'prev_low'], axis=1, inplace=True)  # Drop temporary columns
    df.fillna(0, inplace=True)
    return df

def plot_features(df, start_date_plot, end_date_plot, title_prefix=""):
    """
    Plots the bar shape, bar range, and bar overlap features over a specified date range.

    Args:
        df (pandas.DataFrame): DataFrame containing the features 'bar_shape', 'bar_range', and 'bar_overlap'.
        start_date_plot (str): Start date for the plot.
        end_date_plot (str): End date for the plot.
        title_prefix (str, optional): Prefix for the plot title. Defaults to "".
    """

    if not all(col in df.columns for col in ['bar_shape', 'bar_range', 'bar_overlap']):
        print("Error: DataFrame must contain 'bar_shape', 'bar_range', and 'bar_overlap' columns.")
        return

    df_plot = df.loc[start_date_plot:end_date_plot].copy() # Create a copy to avoid modifying the original DataFrame.

    if df_plot.empty:
        print(f"No data to plot for the specified date range: {start_date_plot} to {end_date_plot}")
        return

    plt.figure(figsize=(15, 9))
    plt.suptitle(f"{title_prefix}Features: Bar Shape, Bar Range, Bar Overlap", fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(df_plot.index, df_plot['bar_shape'], label='Bar Shape')
    plt.title('Bar Shape')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(df_plot.index, df_plot['bar_range'], label='Bar Range')
    plt.title('Bar Range')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(df_plot.index, df_plot['bar_overlap'], label='Bar Overlap')
    plt.title('Bar Overlap')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def analyze_data(df):
    """
    Performs analytical tools and generates plots to assess data suitability for machine learning.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLCV data and calculated features.
    """
    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nCorrelation Matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

    # Plot distributions of numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    df_numerical = df[numerical_cols] # Create a new dataframe
    n_cols = 3
    n_rows = len(numerical_cols) // n_cols + (len(numerical_cols) % n_cols > 0)  #Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle("Distributions of Numerical Features", fontsize=16)
    axes = axes.flatten() # Flatten the axes array for easier indexing

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        sns.histplot(df_numerical[col], kde=True, ax=ax)
        ax.set_title(col)
        # Check for normality (optional, and can give false signals with financial data)
        mu, std = norm.fit(df_numerical[col])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        plt.xlim(xmin, xmax)

    # Remove any unused subplots if the number of columns is not a multiple of n_cols
    if len(numerical_cols) < n_rows * n_cols:
        for i in range(len(numerical_cols), n_rows * n_cols):
            fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()


if __name__ == "__main__":
    # Download SPY data
    start_date = "2010-01-01"  # You can change this
    end_date = "2024-01-01"    # You can change this
    spy_df = download_spy_data(start_date, end_date)
    if spy_df is None:
        exit()  # Exit if data download failed

    # Calculate features
    spy_df_with_features = calculate_features(spy_df)
    if spy_df_with_features is None:
        exit() # Exit if feature calculation failed.

    # Plot features
    start_date_plot = "2023-01-01"  # You can change this
    end_date_plot = "2023-12-31"    # You can change this
    plot_features(spy_df_with_features, start_date_plot, end_date_plot, title_prefix="SPY ")

    # Analyze data
    analyze_data(spy_df_with_features)
