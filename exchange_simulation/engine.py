import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import importlib.util
import os
import time
from datetime import datetime
from collections import deque

# Streamlit app configuration
st.set_page_config(page_title="Trading Strategy Simulator", layout="wide")
st.title("Generalized Trading Strategy Simulator")

# Sidebar for user inputs
st.sidebar.header("Simulation Parameters")
strategy_file = st.sidebar.file_uploader("Upload Strategy Python Script", type="py")
symbol = st.sidebar.text_input("Trading Symbol (e.g., BTC/USDT)", "BTC/USDT")
initial_balance = st.sidebar.number_input("Initial Fake Balance (USD)", value=10000.0)
position_size = st.sidebar.number_input("Position Size per Trade (USD)", value=1000.0)
lookback_period = st.sidebar.number_input("Lookback Period (days for historical data)", value=30, min_value=10)
update_interval = st.sidebar.number_input("Update Interval (seconds)", value=60, min_value=10)
simulation_duration = st.sidebar.number_input("Simulation Duration (minutes)", value=60, min_value=1)
run_simulation = st.sidebar.checkbox("Run Simulation")

# Instructions for strategy script
st.sidebar.markdown("""
**Strategy Script Requirements:**
- Define a function `generate_signal(current_data, positions)` that returns 'buy', 'sell', 'hold'.
- `current_data` is a dict with 'price', 'historical_df' (pandas Series of closes).
- `positions` is current position quantity.
- Optional: Define `strategy_name` variable.
""")

# Initialize CCXT exchange
exchange = ccxt.binance({'enableRateLimit': True})

# Function to fetch historical OHLCV data
@st.cache_data(ttl=300)
def fetch_historical_data(symbol, timeframe='1d', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')['close']
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.Series()

# Function to load strategy from uploaded file
def load_strategy(file):
    if file is None:
        return None
    try:
        # Create temporary file
        temp_filename = f"temp_strategy_{int(time.time())}.py"
        with open(temp_filename, "wb") as f:
            f.write(file.getvalue())
        
        # Load module
        spec = importlib.util.spec_from_file_location("strategy_module", temp_filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Clean up temp file
        os.remove(temp_filename)
        return module
    except Exception as e:
        st.error(f"Error loading strategy: {str(e)}")
        return None

# Function to calculate performance metrics
def calculate_metrics(trades, balance_history):
    if not trades or len(balance_history) < 2:
        return {}
    
    df_trades = pd.DataFrame(trades)
    df_balance = pd.DataFrame(balance_history, columns=['time', 'balance']).set_index('time')
    
    # Total P&L
    total_pnl = df_trades['pnl'].sum() if 'pnl' in df_trades.columns else 0
    
    # Win rate
    winning_trades = df_trades[df_trades['pnl'] > 0] if 'pnl' in df_trades.columns else pd.DataFrame()
    win_rate = len(winning_trades) / len(df_trades) if len(df_trades) > 0 else 0
    
    # Returns calculation
    returns = df_balance['balance'].pct_change().dropna()
    
    # Sharpe ratio (annualized, assuming risk-free rate=0)
    sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24 * 60 / update_interval) if returns.std() != 0 else 0
    
    # Max drawdown
    peak = df_balance['balance'].cummax()
    drawdown = (df_balance['balance'] - peak) / peak
    max_drawdown = drawdown.min()
    
    # Average trade return
    avg_trade_return = df_trades['pnl'].mean() if len(df_trades) > 0 and 'pnl' in df_trades.columns else 0
    
    return {
        'Total P&L': f"${total_pnl:.2f}",
        'Win Rate': f"{win_rate:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Number of Trades': len(trades),
        'Avg Trade Return': f"${avg_trade_return:.2f}"
    }

# Initialize session state
if 'balance' not in st.session_state:
    st.session_state.balance = initial_balance
    st.session_state.position = 0.0
    st.session_state.trades = []
    st.session_state.balance_history = []
    st.session_state.price_history = deque(maxlen=100)
    st.session_state.signal_history = deque(maxlen=100)
    st.session_state.time_history = deque(maxlen=100)
    st.session_state.simulation_running = False

# Reset button
if st.sidebar.button("Reset Simulation"):
    st.session_state.balance = initial_balance
    st.session_state.position = 0.0
    st.session_state.trades = []
    st.session_state.balance_history = []
    st.session_state.price_history.clear()
    st.session_state.signal_history.clear()
    st.session_state.time_history.clear()
    st.session_state.simulation_running = False
    st.success("Simulation reset successfully!")

# Load strategy
strategy = load_strategy(strategy_file)

# Display strategy information
if strategy:
    strategy_name = getattr(strategy, 'strategy_name', 'Unknown Strategy')
    st.info(f"**Loaded Strategy:** {strategy_name}")

# Main simulation
if run_simulation and strategy and not st.session_state.simulation_running:
    st.session_state.simulation_running = True
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Real-time Price Chart")
        price_chart_placeholder = st.empty()
        signal_status_placeholder = st.empty()
    
    with col2:
        st.header("📊 Performance Dashboard")
        balance_placeholder = st.empty()
        position_placeholder = st.empty()
        metrics_placeholder = st.empty()
    
    # Trading log section
    st.header("📋 Trading Log")
    trades_placeholder = st.empty()
    
    # Balance history chart
    st.header("💰 Balance History")
    balance_chart_placeholder = st.empty()
    
    # Control button
    stop_button = st.button("Stop Simulation", key="stop_sim")
    
    start_time = time.time()
    iteration_count = 0
    
    while (time.time() - start_time < simulation_duration * 60 and 
           st.session_state.simulation_running and not stop_button):
        
        try:
            iteration_count += 1
            
            # Fetch historical and current data
            historical_df = fetch_historical_data(symbol, limit=lookback_period)
            if len(historical_df) < lookback_period:
                st.error("Insufficient historical data.")
                break
            
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            current_data = {'price': price, 'historical_df': historical_df}
            
            # Update price history for visualization
            current_time = datetime.now()
            st.session_state.price_history.append(price)
            st.session_state.time_history.append(current_time)
            
            # Get signal from strategy
            signal = strategy.generate_signal(current_data, st.session_state.position)
            st.session_state.signal_history.append(signal)
            
            # Create price chart with signals
            if len(st.session_state.price_history) > 1:
                # Prepare chart data
                chart_data = pd.DataFrame({
                    'Price': list(st.session_state.price_history),
                    'Time': list(st.session_state.time_history)
                }).set_index('Time')
                
                # Display price chart
                price_chart_placeholder.line_chart(chart_data, height=400)
                
                # Show signal status with colored indicators
                signal_cols = st.columns(4)
                with signal_cols[0]:
                    if signal == 'buy':
                        st.success(f"🟢 **BUY SIGNAL**")
                    elif signal == 'sell':
                        st.error(f"🔴 **SELL SIGNAL**")
                    else:
                        st.info(f"⚪ **HOLD**")
                
                with signal_cols[1]:
                    st.metric("Current Price", f"${price:.2f}")
                
                with signal_cols[2]:
                    price_change = (st.session_state.price_history[-1] - 
                                  st.session_state.price_history[-2]) if len(st.session_state.price_history) > 1 else 0
                    st.metric("Price Change", f"${price_change:.2f}", 
                             delta=f"{((price_change/st.session_state.price_history[-2])*100):.2f}%" if len(st.session_state.price_history) > 1 else None)
                
                with signal_cols[3]:
                    st.metric("Iteration", iteration_count)
            
            # Execute trades based on signal
            trade_executed = False
            
            if signal == 'buy' and st.session_state.position <= 0:
                if st.session_state.balance >= position_size:
                    qty = position_size / price
                    st.session_state.position += qty
                    st.session_state.balance -= position_size
                    trade_executed = True
                    
                    trade_record = {
                        'time': current_time,
                        'action': 'BUY',
                        'qty': qty,
                        'price': price,
                        'value': position_size,
                        'pnl': 0,
                        'balance_after': st.session_state.balance
                    }
                    st.session_state.trades.append(trade_record)
            
            elif signal == 'sell' and st.session_state.position > 0:
                sell_value = st.session_state.position * price
                # Calculate P&L (simplified - assumes last buy price)
                last_buy_trades = [t for t in st.session_state.trades if t['action'] == 'BUY']
                if last_buy_trades:
                    avg_buy_price = sum(t['price'] for t in last_buy_trades) / len(last_buy_trades)
                    pnl = st.session_state.position * (price - avg_buy_price)
                else:
                    pnl = 0
                
                st.session_state.balance += sell_value
                trade_executed = True
                
                trade_record = {
                    'time': current_time,
                    'action': 'SELL',
                    'qty': st.session_state.position,
                    'price': price,
                    'value': sell_value,
                    'pnl': pnl,
                    'balance_after': st.session_state.balance
                }
                st.session_state.trades.append(trade_record)
                st.session_state.position = 0.0
            
            # Update balance history
            total_portfolio_value = st.session_state.balance + (st.session_state.position * price)
            st.session_state.balance_history.append((current_time, total_portfolio_value))
            
            # Update dashboard
            with col2:
                balance_placeholder.metric(
                    "Portfolio Value", 
                    f"${total_portfolio_value:.2f}",
                    delta=f"${total_portfolio_value - initial_balance:.2f}"
                )
                
                position_placeholder.metric(
                    "Position", 
                    f"{st.session_state.position:.6f} {symbol.split('/')[0]}",
                    help=f"Cash: ${st.session_state.balance:.2f}"
                )
                
                # Calculate and display metrics
                metrics = calculate_metrics(st.session_state.trades, st.session_state.balance_history)
                metrics_placeholder.json(metrics)
            
            # Update trading log
            if st.session_state.trades:
                recent_trades = st.session_state.trades[-10:]  # Show last 10 trades
                trades_df = pd.DataFrame(recent_trades)
                trades_df['time'] = trades_df['time'].dt.strftime('%H:%M:%S')
                trades_df = trades_df[['time', 'action', 'qty', 'price', 'pnl', 'balance_after']]
                trades_df.columns = ['Time', 'Action', 'Quantity', 'Price', 'P&L', 'Balance']
                trades_placeholder.dataframe(trades_df, use_container_width=True)
            
            # Update balance history chart
            if len(st.session_state.balance_history) > 1:
                balance_chart_data = pd.DataFrame(
                    st.session_state.balance_history, 
                    columns=['Time', 'Portfolio Value']
                ).set_index('Time')
                balance_chart_placeholder.line_chart(balance_chart_data, height=300)
            
            # Add some visual feedback for trade execution
            if trade_executed:
                if signal == 'buy':
                    st.toast(f"✅ Bought {qty:.6f} at ${price:.2f}", icon="📈")
                else:
                    st.toast(f"✅ Sold {trade_record['qty']:.6f} at ${price:.2f} (P&L: ${pnl:.2f})", icon="📉")
            
            # Sleep for the specified interval
            time.sleep(update_interval)
            
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")
            break
    
    st.session_state.simulation_running = False
    
    if not stop_button:
        st.success("🎉 Simulation completed!")
        
        # Final summary
        final_portfolio_value = st.session_state.balance + (st.session_state.position * price if 'price' in locals() else 0)
        total_return = final_portfolio_value - initial_balance
        return_percentage = (total_return / initial_balance) * 100
        
        st.metric(
            "Final Result", 
            f"${final_portfolio_value:.2f}",
            delta=f"{return_percentage:+.2f}%"
        )

elif run_simulation and not strategy:
    st.warning("Please upload a strategy script to begin simulation.")
elif not run_simulation:
    st.info("Upload a strategy script and check 'Run Simulation' to start.")

# Strategy examples section
if st.sidebar.button("Show Strategy Examples"):
    st.sidebar.markdown("""
    **Example Strategy Structure:**
    ```
    strategy_name = "My Strategy"
    
    def generate_signal(current_data, positions):
        price = current_data['price']
        historical_df = current_data['historical_df']
        
        # Your strategy logic here
        if some_buy_condition:
            return 'buy'
        elif some_sell_condition:
            return 'sell'
        else:
            return 'hold'
    ```
    """)
