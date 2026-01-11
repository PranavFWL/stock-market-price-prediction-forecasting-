"""
Live Prediction Script for Intraday Trading
"""

import pandas as pd
import numpy as np
import pickle
import logging
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from config import *
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_artifacts(artifacts_path):
    """Load model artifacts"""
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

def make_prediction(model, df, scaler_X, scaler_y, feature_cols, lookback):
    """
    Make next-step prediction
    
    Returns:
        float: Predicted price change percentage
    """
    # Get latest data
    X_latest = df[feature_cols].iloc[-lookback:].values
    
    # Scale
    X_latest_scaled = scaler_X.transform(X_latest)
    X_latest_seq = X_latest_scaled.reshape(1, lookback, -1)
    
    # Predict
    y_pred_scaled = model.predict(X_latest_seq, verbose=0)[0][0]
    y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
    
    return y_pred

def main():
    """Main prediction pipeline"""
    
    print("=" * 80)
    print("INTRADAY NIFTY 50 LSTM - LIVE PREDICTION")
    print("=" * 80)
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n[1/4] Loading model...")
    
    # Specify your model files here
    model_path = MODEL_DIR / "intraday_lstm_20260111_205059.keras"  
    artifacts_path = MODEL_DIR / "artifacts_20260111_205059.pkl"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info(f"Available models in {MODEL_DIR}:")
        for f in MODEL_DIR.glob("*.keras"):
            print(f"  - {f.name}")
        sys.exit(1)
    
    model = load_model(model_path, safe_mode=False)
    artifacts = load_artifacts(artifacts_path)
    
    scaler_X = artifacts['scaler_X']
    scaler_y = artifacts['scaler_y']
    feature_cols = artifacts['feature_cols']
    lookback = artifacts['lookback_periods']
    
    logger.info(f"Model loaded: {model_path.name}")
    logger.info(f"Lookback: {lookback}, Features: {len(feature_cols)}")
    
    # ========================================================================
    # FETCH LIVE DATA
    # ========================================================================
    print("\n[2/4] Fetching live data...")
    
    fetcher = DataFetcher()
    df_raw = fetcher.fetch_live_data(days_back=5)
    df_raw = fetcher.filter_market_hours(df_raw)
    
    logger.info(f"Fetched {len(df_raw)} candles")
    
    # ========================================================================
    # CREATE FEATURES
    # ========================================================================
    print("\n[3/4] Creating features...")
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_raw)
    df_features = df_features.dropna()
    
    # ========================================================================
    # MAKE PREDICTION
    # ========================================================================
    print("\n[4/4] Making prediction...")
    
    current_price = df_features['close'].iloc[-1]
    current_time = df_features.index[-1]
    
    # Predict
    price_change_pct = make_prediction(
        model, df_features, scaler_X, scaler_y, feature_cols, lookback
    )
    
    predicted_price = current_price * (1 + price_change_pct)
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"Current Time: {current_time}")
    print(f"Current Price: ₹{current_price:,.2f}")
    print(f"Predicted Change: {price_change_pct*100:+.3f}%")
    print(f"Predicted Price (5min): ₹{predicted_price:,.2f}")
    
    if price_change_pct > 0:
        print(f"Signal: 🟢 BUY (Bullish)")
    else:
        print(f"Signal: 🔴 SELL (Bearish)")
    
    print("=" * 80)
    
    # ========================================================================
    # PLOT RECENT PRICE ACTION
    # ========================================================================
    plt.figure(figsize=(12, 6))
    
    # Plot last 100 candles
    recent_data = df_features.iloc[-100:]
    plt.plot(range(len(recent_data)), recent_data['close'], 'b-', linewidth=2, label='Actual Price')
    
    # Plot prediction
    plt.scatter([len(recent_data)], [predicted_price], color='red', s=100, marker='*', 
                label=f'Predicted: ₹{predicted_price:,.2f}', zorder=5)
    
    plt.axhline(y=current_price, color='gray', linestyle='--', alpha=0.5, label=f'Current: ₹{current_price:,.2f}')
    
    plt.title(f'Nifty 50 - Last 100 Candles + Prediction ({current_time})', fontsize=14, fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()