"""
LSTM Model Architecture for Intraday Prediction
NO BIDIRECTIONAL - Uses only past data
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Multiply, Permute, RepeatVector, Activation, Flatten, Lambda
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
import logging
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class IntradayLSTM:
    def __init__(self, config=None):
        """
        Initialize LSTM model for intraday prediction
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.model = None
        
    def build_model(self, lookback_periods, n_features):
        """
        Build LSTM model with attention mechanism
        
        Args:
            lookback_periods: Number of timesteps to look back
            n_features: Number of features per timestep
            
        Returns:
            keras.Model: Compiled model
        """
        logger.info(f"Building model: lookback={lookback_periods}, features={n_features}")
        
        # Input layer
        inputs = Input(shape=(lookback_periods, n_features), name='input')
        
        # LSTM Layer 1 (NOT Bidirectional - no future data)
        lstm1 = LSTM(
            self.config['lstm_units_1'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            name='lstm_1'
        )(inputs)
        lstm1 = BatchNormalization(name='bn_1')(lstm1)
        
        # Attention Mechanism (focuses on recent important patterns)
        attention = Dense(1, activation='tanh', name='attention_dense')(lstm1)
        attention = Flatten(name='attention_flatten')(attention)
        attention = Activation('softmax', name='attention_softmax')(attention)
        attention = RepeatVector(self.config['lstm_units_1'], name='attention_repeat')(attention)
        attention = Permute([2, 1], name='attention_permute')(attention)
        
        # Apply attention
        context = Multiply(name='attention_multiply')([lstm1, attention])
        context = Lambda(lambda x: K.sum(x, axis=1), name='attention_sum')(context)
        
        # Dense layers for final prediction
        dense1 = Dense(
            self.config['dense_units'],
            activation='relu',
            name='dense_1'
        )(context)
        dense1 = BatchNormalization(name='bn_2')(dense1)
        dense1 = Dropout(self.config['dropout_rate'], name='dropout_1')(dense1)
        
        # Output layer - predict price change percentage
        # Using linear activation for regression
        output = Dense(1, activation='linear', name='output')(dense1)
        
        # Build model
        model = Model(inputs=inputs, outputs=output, name='intraday_lstm')
        
        # Compile with Huber loss (robust to outliers)
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def get_callbacks(self, model_path):
        """
        Get training callbacks
        
        Args:
            model_path: Path to save best model
            
        Returns:
            list: Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet")


class SimpleIntradayLSTM:
    """Simpler alternative model without attention"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG
        self.model = None
    
    def build_model(self, lookback_periods, n_features):
        """
        Build simple LSTM model
        
        Args:
            lookback_periods: Number of timesteps to look back
            n_features: Number of features per timestep
            
        Returns:
            keras.Model: Compiled model
        """
        logger.info(f"Building SIMPLE model: lookback={lookback_periods}, features={n_features}")
        
        inputs = Input(shape=(lookback_periods, n_features))
        
        # LSTM layers
        x = LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        x = BatchNormalization()(x)
        
        x = LSTM(32, return_sequences=False, dropout=0.2)(x)
        x = BatchNormalization()(x)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("Simple model built successfully")
        
        return model
    
    def get_callbacks(self, model_path):
        """Get training callbacks"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]