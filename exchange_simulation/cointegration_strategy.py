# enhanced_cointegration_pairs_strategy.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import ccxt

strategy_name = "Enhanced Cointegration Pairs Trading"

# Strategy parameters
SYMBOL_A = "BTC/USDT"
SYMBOL_B = "ETH/USDT"
LOOKBACK_WINDOW = 30
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
STOP_LOSS_THRESHOLD = 3.0
MIN_HALF_LIFE = 1  # Minimum half-life in days for mean reversion
MAX_HALF_LIFE = 30  # Maximum half-life in days

# Initialize exchange for fetching pair data
exchange = ccxt.binance({'enableRateLimit': True})

# Global state
strategy_state = {
    'hedge_ratio': None,
    'mean_spread': None,
    'std_spread': None,
    'half_life': None,
    'last_update': None,
    'position_entry_price': None,
    'position_entry_zscore': None
}

def fetch_pair_data(symbol_A, symbol_B, limit=100):
    """Fetch historical data for both symbols"""
    try:
        ohlcv_A = exchange.fetch_ohlcv(symbol_A, '1d', limit=limit)
        ohlcv_B = exchange.fetch_ohlcv(symbol_B, '1d', limit=limit)
        
        df_A = pd.DataFrame(ohlcv_A, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_B = pd.DataFrame(ohlcv_B, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df_A['timestamp'] = pd.to_datetime(df_A['timestamp'], unit='ms')
        df_B['timestamp'] = pd.to_datetime(df_B['timestamp'], unit='ms')
        
        return df_A.set_index('timestamp')['close'], df_B.set_index('timestamp')['close']
    except:
        return None, None

def calculate_half_life(spread_series):
    """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process"""
    spread_lag = spread_series.shift(1)
    spread_diff = spread_series.diff()
    
    # Remove NaN values
    mask = ~(np.isnan(spread_lag) | np.isnan(spread_diff))
    spread_lag = spread_lag[mask]
    spread_diff = spread_diff[mask]
    
    if len(spread_lag) < 10:
        return np.inf
    
    # OLS regression: spread_diff = lambda * spread_lag + error
    model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
    lambda_param = model.params[1]
    
    if lambda_param >= 0:
        return np.inf
        
    half_life = -np.log(2) / lambda_param
    return half_life

def update_cointegration_parameters():
    """Update hedge ratio and spread statistics"""
    prices_A, prices_B = fetch_pair_data(SYMBOL_A, SYMBOL_B, LOOKBACK_WINDOW)
    
    if prices_A is None or prices_B is None or len(prices_A) < LOOKBACK_WINDOW:
        return False
    
    # Align the data
    common_dates = prices_A.index.intersection(prices_B.index)
    prices_A = prices_A[common_dates]
    prices_B = prices_B[common_dates]
    
    if len(prices_A) < 20:  # Need minimum data points
        return False
    
    # Test for cointegration
    log_A = np.log(prices_A)
    log_B = np.log(prices_B)
    
    coint_result = coint(log_A, log_B)
    if coint_result[1] > 0.05:  # Not cointegrated
        return False
    
    # Calculate hedge ratio
    X = sm.add_constant(log_B)
    model = sm.OLS(log_A, X).fit()
    hedge_ratio = model.params[1]
    
    # Calculate spread
    spread = log_A - hedge_ratio * log_B
    
    # Calculate half-life
    half_life = calculate_half_life(spread)
    
    # Check if half-life is reasonable for mean reversion
    if half_life < MIN_HALF_LIFE or half_life > MAX_HALF_LIFE:
        return False
    
    # Update global state
    strategy_state.update({
        'hedge_ratio': hedge_ratio,
        'mean_spread': spread.mean(),
        'std_spread': spread.std(),
        'half_life': half_life,
        'last_update': pd.Timestamp.now()
    })
    
    return True

def generate_signal(current_data, positions):
    """Enhanced signal generation with proper pairs trading logic"""
    
    # Update parameters periodically
    if (strategy_state['last_update'] is None or 
        (pd.Timestamp.now() - strategy_state['last_update']).total_seconds() > 600):
        
        success = update_cointegration_parameters()
        if not success and strategy_state['hedge_ratio'] is None:
            return 'hold'  # Wait for valid cointegration
    
    # Get current prices for both assets
    try:
        ticker_A = exchange.fetch_ticker(SYMBOL_A)
        ticker_B = exchange.fetch_ticker(SYMBOL_B)
        price_A = ticker_A['last']
        price_B = ticker_B['last']
    except:
        return 'hold'
    
    # Calculate current z-score
    if any(v is None for v in [strategy_state['hedge_ratio'], 
                              strategy_state['mean_spread'], 
                              strategy_state['std_spread']]):
        return 'hold'
    
    current_spread = (np.log(price_A) - 
                     strategy_state['hedge_ratio'] * np.log(price_B))
    z_score = ((current_spread - strategy_state['mean_spread']) / 
               strategy_state['std_spread'])
    
    # Trading logic
    if abs(positions) < 0.001:  # No position
        if z_score > ENTRY_THRESHOLD:
            # Short the spread: expect mean reversion
            strategy_state['position_entry_zscore'] = z_score
            return 'sell'  # In single-asset context, sell the primary asset
        elif z_score < -ENTRY_THRESHOLD:
            # Long the spread: expect mean reversion
            strategy_state['position_entry_zscore'] = z_score
            return 'buy'   # In single-asset context, buy the primary asset
    else:
        # Have position - check for exit
        if abs(z_score) < EXIT_THRESHOLD:
            # Mean reversion occurred
            strategy_state['position_entry_zscore'] = None
            return 'sell' if positions > 0 else 'buy'
        
        elif abs(z_score) > STOP_LOSS_THRESHOLD:
            # Stop loss - correlation may have broken down
            strategy_state['position_entry_zscore'] = None
            return 'sell' if positions > 0 else 'buy'
    
    return 'hold'
