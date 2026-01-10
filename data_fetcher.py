"""
Data Fetcher for Nifty 50 Intraday Data
Handles API calls and data validation
"""

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
        """
        Fetch live data from MoneyControl API
        
        Args:
            days_back: Number of days of historical data to fetch
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        try:
            # Calculate time range
            to_timestamp = int(datetime.now().timestamp())
            from_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            # Build URL
            params = {
                'symbol': self.symbol,
                'resolution': self.resolution,
                'from': from_timestamp,
                'to': to_timestamp,
                'countback': 20000,
                'currencyCode': 'INR'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            logger.info(f"Fetching data from {days_back} days back...")
            response = requests.get(self.api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns
            df.rename(columns={
                'c': 'close',
                'h': 'high',
                'l': 'low',
                'o': 'open',
                'v': 'volume',
                't': 'timestamp'
            }, inplace=True)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Sort by datetime
            df.sort_index(inplace=True)
            
            # Validate data
            df = self._validate_data(df)
            
            logger.info(f"Successfully fetched {len(df)} candles")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
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
        required_cols = ['open', 'high', 'low', 'close', 'volume']
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
        
        # Remove zero volume candles
        zero_volume = df['volume'] == 0
        if zero_volume.any():
            logger.warning(f"Removing {zero_volume.sum()} zero-volume candles")
            df = df[~zero_volume]
        
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
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Resampled to {timeframe}: {len(resampled)} candles")
        
        return resampled