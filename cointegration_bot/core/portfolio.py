def simulate_trade(df, signals, spread_ret, fee=0.001):
    df['position'] = signals.shift(1).fillna(0)
    df['spread_ret'] = spread_ret
    df['pnl'] = df['position'] * df['spread_ret'] - abs(df['position'].diff()) * fee
    df['cumulative_pnl'] = df['pnl'].cumsum()
    return df
