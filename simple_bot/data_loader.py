"""
Data Loader Module
Loads historical data from Binance or CSV files for backtesting
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import time
import requests


class DataLoader:
    """
    Data loader for cryptocurrency OHLCV data
    Supports Binance API and CSV files
    """

    BINANCE_API_URL = "https://api.binance.com/api/v3"

    INTERVALS = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
    }

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader

        Args:
            data_dir: Directory to store cached data
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _get_binance_klines(self, symbol: str, interval: str,
                            start_time: int, end_time: int,
                            limit: int = 1000) -> List:
        """
        Fetch klines from Binance API

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            limit: Max number of candles per request

        Returns:
            List of klines
        """
        url = f"{self.BINANCE_API_URL}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_historical_data(self, symbol: str, interval: str = '1h',
                              start_date: str = None, end_date: str = None,
                              days: int = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            days: Number of days to fetch (alternative to start_date)

        Returns:
            DataFrame with OHLCV data
        """
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        if start_date:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        elif days:
            start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
        else:
            start_ts = end_ts - (365 * 24 * 60 * 60 * 1000)  # Default 1 year

        all_klines = []
        current_start = start_ts
        interval_ms = self.INTERVALS.get(interval, 60 * 60 * 1000)

        print(f"Fetching {symbol} data from {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}")

        while current_start < end_ts:
            try:
                klines = self._get_binance_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_ts,
                    limit=1000
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Move to next batch
                current_start = klines[-1][0] + interval_ms

                # Rate limiting
                time.sleep(0.1)

                print(f"  Fetched {len(all_klines)} candles...")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                time.sleep(1)
                continue

        # Convert to DataFrame
        df = self._klines_to_dataframe(all_klines)
        print(f"Total: {len(df)} candles loaded")

        return df

    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """
        Convert Binance klines to DataFrame

        Args:
            klines: List of klines from Binance API

        Returns:
            DataFrame with OHLCV data
        """
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        return df.sort_index()

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        Save data to CSV file

        Args:
            df: DataFrame to save
            symbol: Symbol name
            interval: Interval
        """
        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")

    def load_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file

        Args:
            symbol: Symbol name
            interval: Interval

        Returns:
            DataFrame or None if file doesn't exist
        """
        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return None

        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        print(f"Data loaded from {filepath}: {len(df)} candles")
        return df

    def get_data(self, symbol: str, interval: str = '1h',
                 start_date: str = None, end_date: str = None,
                 days: int = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Get data with caching support

        Args:
            symbol: Trading pair
            interval: Candle interval
            start_date: Start date
            end_date: End date
            days: Number of days
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        if use_cache:
            cached = self.load_data(symbol, interval)
            if cached is not None and len(cached) > 0:
                # Check if cached data covers the requested range
                if start_date:
                    start = pd.Timestamp(start_date)
                    if cached.index.min() <= start:
                        print("Using cached data")
                        return cached

        # Fetch new data
        df = self.fetch_historical_data(symbol, interval, start_date, end_date, days)

        if len(df) > 0 and use_cache:
            self.save_data(df, symbol, interval)

        return df

    def get_multiple_symbols(self, symbols: List[str], interval: str = '1h',
                             start_date: str = None, end_date: str = None,
                             days: int = None) -> dict:
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of trading pairs
            interval: Candle interval
            start_date: Start date
            end_date: End date
            days: Number of days

        Returns:
            Dict with symbol -> DataFrame mapping
        """
        data = {}
        for symbol in symbols:
            print(f"\nFetching {symbol}...")
            try:
                df = self.get_data(symbol, interval, start_date, end_date, days)
                data[symbol] = df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return data


def generate_sample_data(days: int = 365 * 4, interval_hours: int = 1) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing when API is not available

    Args:
        days: Number of days of data
        interval_hours: Hours per candle

    Returns:
        DataFrame with synthetic OHLCV data
    """
    n_candles = (days * 24) // interval_hours

    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_candles)

    # Generate realistic price movements using geometric Brownian motion
    np.random.seed(42)

    # Initial price
    initial_price = 30000.0

    # Parameters
    mu = 0.0001  # Drift (small positive trend)
    sigma = 0.02  # Volatility

    # Generate returns
    returns = np.random.normal(mu, sigma, n_candles)

    # Add some trends and mean reversion
    trend = np.sin(np.linspace(0, 8 * np.pi, n_candles)) * 0.001
    returns = returns + trend

    # Calculate prices
    prices = initial_price * np.cumprod(1 + returns)

    # Generate OHLC from prices
    open_prices = prices * (1 + np.random.uniform(-0.005, 0.005, n_candles))
    high_prices = np.maximum(prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.01, n_candles)))
    low_prices = np.minimum(prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.01, n_candles)))
    close_prices = prices

    # Generate volume
    base_volume = 1000
    volume = base_volume * np.abs(np.random.normal(1, 0.5, n_candles)) * (1 + np.abs(returns) * 10)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=timestamps)

    return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(data_dir="data")

    # Fetch 4 years of BTC data
    print("Fetching BTC/USDT 4-year hourly data...")
    btc_data = loader.get_data(
        symbol='BTCUSDT',
        interval='1h',
        days=365 * 4
    )

    print(f"\nData shape: {btc_data.shape}")
    print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")
    print(f"\nFirst 5 rows:\n{btc_data.head()}")
    print(f"\nLast 5 rows:\n{btc_data.tail()}")
