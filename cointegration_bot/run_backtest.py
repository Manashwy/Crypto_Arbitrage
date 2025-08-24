# run_backtest.py

import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
from core import cointegration, trading_engine, portfolio
from config import *

# --- Fetch hourly historical price data from CoinGecko ---
def fetch_hourly_data(coin_id, vs_currency='usd', start_date='2024-10-01', end_date='2025-02-01'):
    cg = CoinGeckoAPI()
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=90)

    all_data = []
    while start_dt < end_dt:
        segment_end = min(start_dt + delta, end_dt)
        from_timestamp = int(start_dt.timestamp())
        to_timestamp = int(segment_end.timestamp())
        
        data = cg.get_coin_market_chart_range_by_id(
            id=coin_id,
            vs_currency=vs_currency,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        prices = pd.DataFrame(data['prices'], columns=['timestamp', coin_id])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        all_data.append(prices)
        start_dt = segment_end

    full_df = pd.concat(all_data).drop_duplicates(subset='timestamp').set_index('timestamp').sort_index()
    return full_df

# --- Load and align BTC and ETH data ---
btc_df = fetch_hourly_data('bitcoin')
eth_df = fetch_hourly_data('ethereum')
df = btc_df.join(eth_df, how='inner')
df.columns = ['BTC', 'ETH']
df = df[START_DATE:END_DATE].copy()

# --- Cointegration test on historical window before simulation start ---
lookback_window = 90 * 24  # 90 days of hourly data
train_start = datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=90)
train_df = df[train_start.strftime('%Y-%m-%d'):START_DATE]

pval, beta, spread_train = cointegration.test_cointegration(train_df['ETH'], train_df['BTC'])
print(f"Cointegration p-value: {pval:.4f}, Hedge Ratio: {beta:.4f}")

# --- Compute spread and z-score for simulation window ---
df['spread'] = df['ETH'] - beta * df['BTC']
df['zscore'] = trading_engine.calculate_zscore(df['spread'], window=Z_WINDOW)

# --- Generate signals and simulate trade ---
signals = trading_engine.generate_signals(df['zscore'], entry_threshold=Z_ENTRY, exit_threshold=Z_EXIT)
df['spread_ret'] = df['spread'].diff()
results = portfolio.simulate_trade(df, signals, df['spread_ret'], fee=FEE_RATE)

# --- Output final performance ---
print(f"Final PnL (Jan 2024): {results['cumulative_pnl'].iloc[-1]:.2f}")
results[['spread', 'zscore', 'position', 'pnl', 'cumulative_pnl']].to_csv('results/jan2024_backtest.csv')
