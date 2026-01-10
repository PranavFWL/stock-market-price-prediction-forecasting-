"""
Configuration settings for Intraday Nifty 50 LSTM Prediction System
Production-Grade Implementation
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# MODEL SETTINGS
# ============================================================================
MODEL_CONFIG = {
    'lookback_periods': 60,      # 60 * 5min = 5 hours of data
    'prediction_horizon': 1,     # Predict 1 step ahead (5 minutes)
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dense_units': 32,
    'dropout_rate': 0.2,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 1,
    'patience': 7,
    'reduce_lr_patience': 3,
}

# ============================================================================
# DATA SETTINGS
# ============================================================================
DATA_CONFIG = {
    'api_url': "https://priceapi.moneycontrol.com/techCharts/indianMarket/index/history",
    'symbol': "in;NSX",
    'resolution': 5,  # 5-minute candles
    'candles_per_day': 75,  # 9:15 AM - 3:30 PM = 375 minutes / 5 = 75 candles
    'min_data_points': 5000,  # Minimum candles needed for training
}

# ============================================================================
# TRADING SETTINGS
# ============================================================================
TRADING_CONFIG = {
    'transaction_cost': 0.0005,  # 0.05% per trade (both ways)
    'slippage': 0.0002,          # 0.02% slippage
    'position_size': 1.0,        # Full capital per trade (modify based on risk)
    'stop_loss_pct': 0.005,      # 0.5% stop loss
    'take_profit_pct': 0.01,     # 1% take profit
}

# ============================================================================
# MARKET HOURS (IST)
# ============================================================================
MARKET_HOURS = {
    'open_time': '09:15',
    'close_time': '15:30',
    'opening_range_minutes': 30,   # First 30 minutes
    'closing_range_minutes': 30,   # Last 30 minutes
    'lunch_start': 135,            # Minutes since market open
    'lunch_end': 195,
}

# ============================================================================
# FEATURE ENGINEERING SETTINGS
# ============================================================================
FEATURE_CONFIG = {
    'volatility_windows': [10, 20, 40],
    'ma_periods': [10, 20, 50],
    'momentum_windows': [1, 3, 6, 12],  # 5min, 15min, 30min, 1hour
    'volume_windows': [10, 20],
    'support_resistance_windows': [20, 40, 60],
}

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================
VALIDATION_CONFIG = {
    'n_folds': 5,
    'validation_split': 0.2,
    'test_split': 0.1,
    'walk_forward': True,
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'trading.log',
}

# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================
RANDOM_SEED = 42