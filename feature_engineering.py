"""
Production-Grade Feature Engineering for Intraday Trading
NO DATA LEAKAGE - All features use only past data
Using 'ta' library (compatible with all Python versions including 3.14)
"""

import pandas as pd
import numpy as np
# Using 'ta' library instead of pandas_ta for better compatibility
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from config import FEATURE_CONFIG, MARKET_HOURS, DATA_CONFIG
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_config = FEATURE_CONFIG
        self.candles_per_day = DATA_CONFIG['candles_per_day']
        
    def create_all_features(self, df):
        """
        Create all features for intraday trading
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        logger.info("Creating features...")
        df = df.copy()
        
        # Add date and time features
        df = self._add_datetime_features(df)
        
        # Add session-based features
        df = self._add_session_features(df)
        
        # Add previous day features
        df = self._add_previous_day_features(df)
        
        # Add price action features
        df = self._add_price_action_features(df)
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add volume features
        df = self._add_volume_features(df)
        
        # Add volatility features
        df = self._add_volatility_features(df)
        
        # Add momentum features
        df = self._add_momentum_features(df)
        
        # Add support/resistance features
        df = self._add_support_resistance(df)
        
        # Add VWAP (daily reset)
        df = self._add_vwap(df)
        
        logger.info(f"Created {len(df.columns)} total columns")
        
        return df
    
    def _add_datetime_features(self, df):
        """Add time-based features"""
        df['date'] = df.index.date
        df['time'] = df.index.time
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Minutes since market open (9:15 AM)
        df['minutes_since_open'] = (df['hour'] - 9) * 60 + (df['minute'] - 15)
        df['minutes_since_open'] = df['minutes_since_open'].clip(lower=0)
        
        return df
    
    def _add_session_features(self, df):
        """Add session-specific features (opening, lunch, closing)"""
        # Opening session (first 30 minutes)
        df['is_opening_session'] = (
            df['minutes_since_open'] <= MARKET_HOURS['opening_range_minutes']
        ).astype(int)
        
        # Closing session (last 30 minutes)
        closing_start = (6 * 60 + 15) - MARKET_HOURS['closing_range_minutes']  # 3:00 PM
        df['is_closing_session'] = (
            df['minutes_since_open'] >= closing_start
        ).astype(int)
        
        # Lunch hour
        df['is_lunch_hour'] = (
            (df['minutes_since_open'] >= MARKET_HOURS['lunch_start']) &
            (df['minutes_since_open'] <= MARKET_HOURS['lunch_end'])
        ).astype(int)
        
        # Mid-session (11:00 AM - 2:00 PM)
        df['is_mid_session'] = (
            (df['minutes_since_open'] >= 105) &
            (df['minutes_since_open'] <= 285)
        ).astype(int)
        
        return df
    
    def _add_previous_day_features(self, df):
        """Add previous day's key levels - CRITICAL for intraday"""
        # Group by date to get daily statistics
        daily_stats = df.groupby('date').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'open': 'first',
            'volume': 'sum'
        })
        
        # Shift to get previous day
        daily_stats = daily_stats.shift(1)
        daily_stats.columns = ['prev_day_' + col for col in daily_stats.columns]
        
        # Merge back to main dataframe
        df = df.merge(daily_stats, left_on='date', right_index=True, how='left')
        
        # Distance from previous day's levels
        df['dist_from_prev_high'] = (df['close'] - df['prev_day_high']) / df['close']
        df['dist_from_prev_low'] = (df['close'] - df['prev_day_low']) / df['close']
        df['dist_from_prev_close'] = (df['close'] - df['prev_day_close']) / df['close']
        
        # Opening gap (gap between today's open and yesterday's close)
        df['opening_gap'] = df.groupby('date')['open'].transform('first') - df['prev_day_close']
        df['opening_gap_pct'] = df['opening_gap'] / df['prev_day_close']
        
        # Check if gap was filled
        first_candle_close = df.groupby('date')['close'].transform('first')
        df['gap_filled'] = (
            ((df['opening_gap'] > 0) & (df['low'] <= df['prev_day_close'])) |
            ((df['opening_gap'] < 0) & (df['high'] >= df['prev_day_close']))
        ).astype(int)
        
        return df
    
    def _add_price_action_features(self, df):
        """Add candlestick and price action features"""
        # Basic candle features
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Ratios
        df['body_to_range'] = df['body'] / (df['range'] + 1e-10)
        df['upper_wick_to_range'] = df['upper_wick'] / (df['range'] + 1e-10)
        df['lower_wick_to_range'] = df['lower_wick'] / (df['range'] + 1e-10)
        
        # Bullish/Bearish
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['body_to_range'] < 0.1).astype(int)
        
        # High/Low position
        df['close_position_in_range'] = (df['close'] - df['low']) / (df['range'] + 1e-10)
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add technical indicators using 'ta' library"""
        # RSI
        for period in [14, 30]:
            rsi_indicator = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi_indicator.rsi()
        
        # MACD
        macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Moving Averages
        for period in self.feature_config['ma_periods']:
            # SMA
            sma_indicator = SMAIndicator(close=df['close'], window=period)
            df[f'sma_{period}'] = sma_indicator.sma_indicator()
            df[f'price_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / (df[f'sma_{period}'] + 1e-10)
            
            # EMA
            ema_indicator = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema_indicator.ema_indicator()
            df[f'price_to_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / (df[f'ema_{period}'] + 1e-10)
        
        # EMA crossovers
        if 'ema_10' in df.columns and 'ema_20' in df.columns:
            df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(int)
        
        return df
    
    def _add_volume_features(self, df):
        """Add volume-based features"""
        # Volume moving averages
        for window in self.feature_config['volume_windows']:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_sma_{window}'] + 1e-10)
        
        # Volume spike detection
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # OBV (On-Balance Volume)
        obv_indicator = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv_indicator.on_balance_volume()
        
        # OBV EMA
        obv_ema_indicator = EMAIndicator(close=df['obv'], window=20)
        df['obv_ema'] = obv_ema_indicator.ema_indicator()
        df['obv_to_ema'] = df['obv'] / (df['obv_ema'] + 1e-10)
        
        # Session average volume
        df['session_avg_volume'] = df.groupby('date')['volume'].transform(
            lambda x: x.expanding().mean()
        )
        df['volume_vs_session'] = df['volume'] / (df['session_avg_volume'] + 1e-10)
        
        return df
    
    def _add_volatility_features(self, df):
        """Add volatility features"""
        # True Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # ATR (Average True Range)
        for window in self.feature_config['volatility_windows']:
            atr_indicator = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=window
            )
            df[f'atr_{window}'] = atr_indicator.average_true_range()
            df[f'atr_pct_{window}'] = df[f'atr_{window}'] / df['close']
        
        # Historical Volatility (log returns)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        for window in self.feature_config['volatility_windows']:
            df[f'hist_vol_{window}'] = df['log_returns'].rolling(window=window).std()
        
        # Intraday range
        df['intraday_range'] = df['range'] / df['open']
        
        return df
    
    def _add_momentum_features(self, df):
        """Add momentum indicators"""
        # Returns at different timeframes
        for window in self.feature_config['momentum_windows']:
            df[f'returns_{window}'] = df['close'].pct_change(window)
            df[f'returns_{window}_abs'] = abs(df[f'returns_{window}'])
        
        # Rate of Change (ROC)
        for window in [10, 20]:
            roc_indicator = ROCIndicator(close=df['close'], window=window)
            df[f'roc_{window}'] = roc_indicator.roc()
        
        # Momentum (simple price difference)
        for window in [10, 20]:
            df[f'momentum_{window}'] = df['close'].diff(window)
        
        # Stochastic Oscillator
        stoch_indicator = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()
        
        return df
    
    def _add_support_resistance(self, df):
        """Add support/resistance levels (proper lookback only)"""
        for window in self.feature_config['support_resistance_windows']:
            # Swing high/low (NO center=True to avoid lookahead bias)
            df[f'swing_high_{window}'] = df['high'].rolling(window=window, center=False).max()
            df[f'swing_low_{window}'] = df['low'].rolling(window=window, center=False).min()
            
            # Distance to swing levels
            df[f'dist_to_high_{window}'] = (df[f'swing_high_{window}'] - df['close']) / df['close']
            df[f'dist_to_low_{window}'] = (df['close'] - df[f'swing_low_{window}']) / df['close']
            
            # Near swing levels (within 0.2%)
            df[f'near_high_{window}'] = (df[f'dist_to_high_{window}'] < 0.002).astype(int)
            df[f'near_low_{window}'] = (df[f'dist_to_low_{window}'] < 0.002).astype(int)
        
        return df
    
    def _add_vwap(self, df):
        """Add VWAP (Volume Weighted Average Price) - reset daily"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['typical_price_volume'] = typical_price * df['volume']
        
        # Calculate cumulative sums per session (reset daily)
        df['cumsum_volume'] = df.groupby('date')['volume'].cumsum()
        df['cumsum_tp_volume'] = df.groupby('date')['typical_price_volume'].cumsum()
        
        # VWAP
        df['vwap'] = df['cumsum_tp_volume'] / (df['cumsum_volume'] + 1e-10)
        df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        # Above/below VWAP
        df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
        
        # Drop intermediate columns
        df.drop(['typical_price_volume', 'cumsum_volume', 'cumsum_tp_volume'], axis=1, inplace=True)
        
        return df
    
    def get_feature_columns(self, df):
        """
        Get list of feature columns (exclude OHLCV and datetime)
        
        Args:
            df: DataFrame with features
            
        Returns:
            list: Feature column names
        """
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'timestamp', 'date', 'time', 'datetime'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return feature_cols