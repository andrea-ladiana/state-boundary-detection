import pandas as pd
import numpy as np
from . import config

def smooth_series(data_series, window=config.SMOOTHING_WINDOW_SIZE):
    """
    Applies a sliding window mean smoothing to the data series.
    """
    return data_series.rolling(window=window, min_periods=1, center=True).mean()

def check_consistency(df, threshold_ratio=0.5):
    """
    Checks for large jumps in Area that might indicate detection errors.
    
    Args:
        df: DataFrame containing 'Year', 'State_ID', 'Area_px'.
        threshold_ratio: Max allowed change ratio between consecutive measurements.
        
    Returns:
        df_cleaned: DataFrame with outliers removed (or marked).
    """
    # Sort by State and Year
    df = df.sort_values(['State_ID', 'Year'])
    
    # Calculate percent change
    # We need to group by State
    df['Area_Change'] = df.groupby('State_ID')['Area_px'].pct_change().abs()
    
    # Filter out rows with excessive change
    # For now, we'll just flag them.
    df['Is_Outlier'] = df['Area_Change'] > threshold_ratio
    
    return df
