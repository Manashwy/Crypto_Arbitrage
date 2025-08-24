import pandas as pd

def load_crypto_data(symbol, path='data/raw/', fmt='csv'):
    file = f"{path}{symbol}.{fmt}"
    if fmt == 'csv':
        return pd.read_csv(file, index_col='timestamp', parse_dates=True)
    elif fmt == 'parquet':
        return pd.read_parquet(file)
