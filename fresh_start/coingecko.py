from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

class CoinGeckoHandler:
    def __init__(self):
        self.cg = CoinGeckoAPI()

    # ------------------- Core Info ------------------- #
    def get_basic_info(self, coin_id):
        """
        Full metadata from /coins/{id}
        """
        return self.cg.get_coin_by_id(id=coin_id)

    def get_coin_description(self, coin_id):
        data = self.get_basic_info(coin_id)
        return data.get('description', {}).get('en', '')

    def get_coin_links(self, coin_id):
        data = self.get_basic_info(coin_id)
        return data.get('links', {})

    def get_coin_image(self, coin_id):
        data = self.get_basic_info(coin_id)
        return data.get('image', {})

    def get_coin_scores(self, coin_id):
        data = self.get_basic_info(coin_id)
        return {
            "market_cap_rank": data.get("market_cap_rank"),
            "coingecko_rank": data.get("coingecko_rank"),
            "coingecko_score": data.get("coingecko_score"),
            "developer_score": data.get("developer_score"),
            "community_score": data.get("community_score"),
            "liquidity_score": data.get("liquidity_score")
        }

    # ------------------- Market Data ------------------- #
    def get_current_market_data(self, coin_ids, vs_currency='usd'):
        """
        /coins/markets for multiple coins
        """
        data = self.cg.get_coins_markets(vs_currency=vs_currency, ids=coin_ids)
        return pd.DataFrame(data)

    def get_price(self, coin_id, vs_currency='usd'):
        return self.cg.get_price(ids=coin_id, vs_currencies=vs_currency)

    # ------------------- Historical Data ------------------- #
    def get_price_history(self, coin_id, vs_currency='usd', days=365):
        """
        Returns price + return DataFrame
        """
        data = self.cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['return'] = df['price'].pct_change().fillna(0)
        return df

    def get_ohlc(self, coin_id, vs_currency='usd', days=30):
        """
        OHLC candlestick data (1d, 7d, 14d, 30d, 90d, 180d, 365d, max)
        """
        data = self.cg.get_coin_ohlc_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_historical_data_on_date(self, coin_id, date_str):
        """
        /coins/{id}/history on a given date (format: 'dd-mm-yyyy')
        """
        return self.cg.get_coin_history_by_id(id=coin_id, date=date_str)

    # ------------------- Ticker / Exchange Data ------------------- #
    def get_tickers(self, coin_id):
        """
        Get exchange tickers (trading pairs) for a coin
        """
        data = self.cg.get_coin_ticker_by_id(id=coin_id)
        return pd.DataFrame(data.get('tickers', []))

    # ------------------- Categories ------------------- #
    def get_coin_categories(self):
        return self.cg.get_coins_categories()

    # ------------------- ADF Test ------------------- #
    def adf_test(self, series):
        result = adfuller(series)
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary (p < 0.05)': result[1] < 0.05
        }
