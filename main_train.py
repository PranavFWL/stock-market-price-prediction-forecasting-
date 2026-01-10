"""
Main Training Script for Intraday Nifty 50 LSTM Prediction
Complete production-ready implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.preprocessing import RobustScaler
from datetime import datetime
import sys

# Import custom modules (ensure they're in the same directory)
from config import *
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from model_architecture import IntradayLSTM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_sequences(df, feature_cols, lookback, horizon):
    """
    Create sequences for LSTM training
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns
        lookback: Number of timesteps to look back
        horizon: Number of steps ahead to predict
        
    Returns:
        X, y: Training sequences and targets
    """
    logger.info(f"Creating sequences: lookback={lookback}, horizon={horizon}")
    
    # Get feature values
    X_data = df[feature_cols].values
    
    # Create target (price change percentage)
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = (df['future_close'] - df['close']) / df['close']
    
    # Remove rows with NaN targets
    df = df.dropna(subset=['target'])
    y_data = df['target'].values
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_data) - lookback):
        X_sequences.append(X_data[i:i + lookback])
        y_sequences.append(y_data[i + lookback])
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
    
    return X, y

def train_test_split_temporal(X, y, test_size=0.2):
    """
    Temporal train/test split (no shuffling)
    
    Args:
        X: Features
        y: Targets
        test_size: Test set proportion
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    logger.info(f"Split: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main training pipeline"""
    
    print("=" * 80)
    print("INTRADAY NIFTY 50 LSTM - TRAINING PIPELINE")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # ========================================================================
    # STEP 1: FETCH DATA
    # ========================================================================
    print("\n[1/6] Fetching data...")
    fetcher = DataFetcher()
    
    try:
        df_raw = fetcher.fetch_live_data(days_back=60)  # Get 60 days of data
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)
    
    # Filter to market hours only
    df_raw = fetcher.filter_market_hours(df_raw)
    
    logger.info(f"Raw data: {len(df_raw)} candles from {df_raw.index[0]} to {df_raw.index[-1]}")
    
    # ========================================================================
    # STEP 2: CREATE FEATURES
    # ========================================================================
    print("\n[2/6] Engineering features...")
    engineer = FeatureEngineer()
    
    df_features = engineer.create_all_features(df_raw)
    feature_cols = engineer.get_feature_columns(df_features)
    
    # Remove NaN rows (from indicator calculations)
    df_features = df_features.dropna()
    logger.info(f"After feature engineering: {len(df_features)} candles, {len(feature_cols)} features")
    
    # ========================================================================
    # STEP 3: CREATE SEQUENCES
    # ========================================================================
    print("\n[3/6] Creating sequences...")
    lookback = MODEL_CONFIG['lookback_periods']
    horizon = MODEL_CONFIG['prediction_horizon']
    
    X, y = prepare_sequences(df_features, feature_cols, lookback, horizon)
    
    # ========================================================================
    # STEP 4: TRAIN/TEST SPLIT & SCALING
    # ========================================================================
    print("\n[4/6] Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split_temporal(
        X, y, test_size=VALIDATION_CONFIG['validation_split']
    )
    
    # Scale features using RobustScaler (handles outliers better)
    scaler_X = RobustScaler()
    
    # Reshape for scaling
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    X_train_scaled = scaler_X.fit_transform(X_train_2d)
    X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Scale test set
    n_samples_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_features)
    X_test_scaled = scaler_X.transform(X_test_2d)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    # Scale targets (for better training stability)
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    logger.info(f"Scaled data - X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")
    logger.info(f"Target range - Train: [{y_train.min():.4f}, {y_train.max():.4f}]")
    logger.info(f"Target range - Test: [{y_test.min():.4f}, {y_test.max():.4f}]")
    
    # ========================================================================
    # STEP 5: BUILD AND TRAIN MODEL
    # ========================================================================
    print("\n[5/6] Building and training model...")
    
    lstm_model = IntradayLSTM(MODEL_CONFIG)
    model = lstm_model.build_model(lookback, n_features)
    lstm_model.summary()
    
    # Define model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"intraday_lstm_{timestamp}.keras"
    
    # Get callbacks
    callbacks = lstm_model.get_callbacks(str(model_path))
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================================================
    # STEP 6: EVALUATE AND SAVE
    # ========================================================================
    print("\n[6/6] Evaluating and saving...")
    
    # Evaluate on test set
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test MAE: {test_mae:.6f}")
    logger.info(f"Test MSE: {test_mse:.6f}")
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate metrics in percentage terms
    mae_pct = np.mean(np.abs(y_pred - y_test)) * 100
    rmse_pct = np.sqrt(np.mean((y_pred - y_test)**2)) * 100
    direction_accuracy = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(int)) * 100
    
    logger.info(f"\n=== Model Performance ===")
    logger.info(f"MAE: {mae_pct:.4f}%")
    logger.info(f"RMSE: {rmse_pct:.4f}%")
    logger.info(f"Direction Accuracy: {direction_accuracy:.2f}%")
    
    # Save artifacts
    artifacts = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'lookback_periods': lookback,
        'prediction_horizon': horizon,
        'model_config': MODEL_CONFIG,
        'training_date': timestamp,
        'test_metrics': {
            'mae_pct': mae_pct,
            'rmse_pct': rmse_pct,
            'direction_accuracy': direction_accuracy
        }
    }
    
    artifacts_path = MODEL_DIR / f"artifacts_{timestamp}.pkl"
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    logger.info(f"\nModel saved: {model_path}")
    logger.info(f"Artifacts saved: {artifacts_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel: {model_path.name}")
    print(f"MAE: {mae_pct:.4f}% | Direction Accuracy: {direction_accuracy:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()