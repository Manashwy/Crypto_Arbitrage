import os
import re
import pandas as pd
import numpy as np

class FileReader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.list_of_files = self.get_list_of_files()
        self.number_of_files = len(self.list_of_files)
        
    def get_list_of_files(self):
        return [(self.split_filename(f), f) for f in os.listdir(self.data_dir) if f.endswith('.csv')]


    def split_filename(self, name):
        base = name.replace('.csv', '')
        if '_' in base:
            return base.split('_')
        elif '-' in base:
            return base.split('-')
        else:
            match = re.match(r'([a-z0-9]+?)(usd|usdt|btc|btcf0|gbp|eth|ust|ustf0|eutf0|testusdt|testusdtf0|jpy|eur|xaut|eut|cnht|mxnt|try|mim|xch)$', base)
            return [match.group(1), match.group(2)] if match else [base, None]


    def filter_by_currency(self, currency=None):
        if currency is None:
            return self.list_of_files
        if not isinstance(currency, str):
            raise ValueError("Currency must be a string")
        return [f for f in self.list_of_files if f[0][1] == currency.lower()]

    def load_latest_data(self, selected_files, number_of_files=None, nrows=None, min_rows=105040):
        if number_of_files is not None:
            selected_files = selected_files[:number_of_files]
        if not selected_files:
            raise ValueError("No files selected for loading.")

        data = {}
        for file in selected_files:
            path = os.path.join(self.data_dir, file[1])
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            if len(df) < min_rows:
                print(f"Skipping {file[1]}: not enough data.")
                continue
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            if nrows is not None:
                df = df.tail(nrows)
            df.sort_index(inplace=True)
            data[file[0][0] + "/" + file[0][1]] = df

        return data
    
    
    