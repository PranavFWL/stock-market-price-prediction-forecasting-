
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from config import DATA_CONFIG

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self.api_url = DATA_CONFIG['api_url']
        self.symbol = DATA_CONFIG['symbol']
        self.resolution = DATA_CONFIG['resolution']
        
    def fetch_live_data(self, days_back=30):
        """Fetch live data from MoneyControl API"""
        try:
            to_timestamp = int(datetime.now().timestamp())
            from_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            url = f"{self.api_url}?symbol={self.symbol}&resolution={self.resolution}&from={from_timestamp}&to={to_timestamp}&countback=20000&currencyCode=INR"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            logger.info(f"Fetching data from {days_back} days back...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = pd.DataFrame(response.json())
            data.rename(columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume', 't': 'timestamp'}, inplace=True)
            
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
            data.set_index('datetime', inplace=True)
            data.sort_index(inplace=True)

            # ADD DEBUG HERE
            print(f"Data shape before validation: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            print(f"First 3 rows:\n{data.head(3)}")
            print(f"Volume stats: min={data['volume'].min()}, max={data['volume'].max()}, mean={data['volume'].mean()}")

            
            df = self._validate_data(data)
            logger.info(f"Successfully fetched {len(df)} candles")
            
            return df
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def _validate_data(self, df):
        """
        Validate and clean fetched data
        
        Args:
            df: Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            df.dropna(subset=required_cols, inplace=True)
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
            df = df[~invalid_ohlc]
        
        
        # Check for outliers (price changes > 10%)
        returns = df['close'].pct_change()
        outliers = abs(returns) > 0.10
        if outliers.any():
            logger.warning(f"Found {outliers.sum()} potential outliers (>10% price change)")
        
        return df
    
    def load_from_csv(self, filepath):
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            df = pd.read_csv(filepath)
            
            # Try to parse datetime
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
            
            df = self._validate_data(df)
            logger.info(f"Loaded {len(df)} candles from {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def filter_market_hours(self, df):
        """
        Filter data to keep only market hours (9:15 AM - 3:30 PM IST)
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        market_hours = (
            (df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 15))
        ) & (
            (df.index.hour < 15) | ((df.index.hour == 15) & (df.index.minute <= 30))
        )
        
        df_filtered = df[market_hours].copy()
        logger.info(f"Filtered to market hours: {len(df_filtered)} candles")
        
        return df_filtered
    
    def resample_data(self, df, timeframe='5T'):
        """
        Resample data to different timeframe
        
        Args:
            df: DataFrame with datetime index
            timeframe: Target timeframe (e.g., '5T' for 5 minutes)
            
        Returns:
            pd.DataFrame: Resampled dataframe
        """
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        logger.info(f"Resampled to {timeframe}: {len(resampled)} candles")
        
        return resampled