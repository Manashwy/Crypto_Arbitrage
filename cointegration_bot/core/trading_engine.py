import numpy as np
import pandas as pd

def calculate_zscore(spread, window=60):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std

def generate_signals(z, entry_threshold=1, exit_threshold=0):
    signal = np.where(z > entry_threshold, -1,
             np.where(z < -entry_threshold, 1,
             np.where(abs(z) < exit_threshold, 0, np.nan)))
    return pd.Series(signal).ffill().fillna(0)
