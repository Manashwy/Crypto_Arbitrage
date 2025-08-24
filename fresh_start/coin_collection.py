import ccxt

def load_markets():
    print("Start\n-------------")
    exchange = ccxt.poloniex()
    exchange.fetch_ohlcv_ws()
    markets = exchange.load_markets(True)
    symbols = list(markets.keys())
    vs_curr = set([str(s.split("/")[-1]) for s in symbols])
    skip = input("Skip to trading?(y/n)")
    if skip.lower().strip()=='y': 
        print("Exchange Currencies Available:\n",vs_curr)
        exchange_currency  = input("Enter Exchange Currency ('all' to include everything):\n").upper()
        print("Selected Exchange Currency: ", exchange_currency)
        if exchange_currency == 'ALL':
            exch_pairs = symbols
        else:
            exch_pairs = [s for s in symbols if f'/{exchange_currency}' in s]
        print(f"Found - {len(exch_pairs)}")
    
    pairs =  ['QTUM/USDT', 'XRP/USDT']
    print(pairs)
        

if __name__ == '__main__':
    load_markets()