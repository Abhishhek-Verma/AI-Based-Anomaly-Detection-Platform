"""
Autoencoder Neural Network for Anomaly Detection
Learns normal physiological patterns from training data
Detects anomalies through reconstruction error
"""

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detection model.
    Learns reconstruction of normal vital signs patterns.
    Anomalies detected by high reconstruction error (learned variance from data).
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 4, 
                 threshold_percentile: float = 95.0,
                 epochs: int = 50, batch_size: int = 32):
        """
        Initialize Autoencoder for anomaly detection.
        
        Args:
            input_dim: Dimension of input features
            encoding_dim: Dimension of encoded representation
            threshold_percentile: Percentile for anomaly threshold (90-99)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.is_trained = False
        self.reconstruction_threshold = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Build autoencoder architecture.
        Encoder: input_dim -> encoding_dim (2 layers)
        Decoder: encoding_dim -> input_dim (2 layers)
        
        Returns:
            Compiled Keras model
        """
        # Encoder
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(8, activation='relu', name='encoder_dense1')(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoder_dense2')(encoded)
        
        # Decoder
        decoded = layers.Dense(8, activation='relu', name='decoder_dense1')(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear', name='decoder_output')(decoded)
        
        # Complete autoencoder
        self.model = keras.Model(input_layer, decoded, name='Autoencoder')
        
        # Compile with Adam optimizer and MSE loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Autoencoder built: {self.input_dim} -> {self.encoding_dim} -> {self.input_dim}")
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              validation_split: float = 0.2, verbose: int = 1) -> Dict:
        """
        Train autoencoder on normal (non-anomalous) data.
        
        Args:
            X_train: Training features (num_samples, input_dim)
            X_val: Validation features (optional)
            validation_split: Fraction of data for validation if X_val not provided
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Use provided validation set or split from training
        if X_val is None:
            val_data = None
            use_val_split = validation_split
        else:
            val_data = (X_val, X_val)
            use_val_split = None
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_data,
            validation_split=use_val_split,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Set anomaly threshold based on training data
        self._calibrate_threshold(X_train)
        
        logger.info(f"Autoencoder trained. Threshold: {self.reconstruction_threshold:.4f}")
        return self.history.history
    
    def _calibrate_threshold(self, X_train: np.ndarray) -> None:
        """
        Calculate reconstruction error threshold for anomaly detection.
        Uses percentile of training reconstruction errors.
        
        Args:
            X_train: Training features
        """
        training_predictions = self.model.predict(X_train, verbose=0)
        training_mse = np.mean(np.square(X_train - training_predictions), axis=1)
        
        self.reconstruction_threshold = np.percentile(
            training_mse, 
            self.threshold_percentile
        )
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error (Mean Squared Error).
        
        Args:
            X: Input features (num_samples, input_dim)
            
        Returns:
            Reconstruction error per sample
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        X_pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_pred), axis=1)
        
        return mse
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly score (0-1) based on reconstruction error.
        Scores normalized so threshold maps to ~0.5
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (0=normal, 1=severe anomaly)
        """
        if self.reconstruction_threshold is None:
            raise ValueError("Model must be trained")
        
        mse = self.get_reconstruction_error(X)
        
        # Normalize: threshold -> 0.5, higher errors -> closer to 1
        # Lower errors -> closer to 0
        anomaly_scores = np.clip(
            mse / (2 * self.reconstruction_threshold),
            0.0, 1.0
        )
        
        return anomaly_scores
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make complete predictions including anomaly scores and flags.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with:
                - reconstruction_error: MSE for each sample
                - anomaly_score: 0-1 anomaly scores
                - is_anomaly: boolean anomaly flags
        """
        mse = self.get_reconstruction_error(X)
        scores = self.predict_anomaly_score(X)
        anomaly_flags = mse > self.reconstruction_threshold
        
        return {
            'reconstruction_error': mse,
            'anomaly_score': scores,
            'is_anomaly': anomaly_flags,
            'threshold': self.reconstruction_threshold
        }
    
    def save(self, modelpath: str, configpath: Optional[str] = None) -> None:
        """
        Save trained model and configuration.
        
        Args:
            modelpath: Path to save model (.h5 or .keras)
            configpath: Path to save config (auto-generated if not provided)
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        Path(modelpath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(modelpath)
        
        if configpath is None:
            configpath = modelpath.replace('.h5', '_config.joblib').replace('.keras', '_config.joblib')
        
        config = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'threshold_percentile': self.threshold_percentile,
            'reconstruction_threshold': float(self.reconstruction_threshold),
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        
        joblib.dump(config, configpath)
        logger.info(f"Model saved to {modelpath}, config to {configpath}")
    
    def load(self, modelpath: str, configpath: Optional[str] = None) -> None:
        """
        Load trained model and configuration.
        
        Args:
            modelpath: Path to load model
            configpath: Path to load config
        """
        if not Path(modelpath).exists():
            raise FileNotFoundError(f"Model file not found: {modelpath}")
        
        self.model = keras.models.load_model(modelpath)
        self.is_trained = True
        
        if configpath is None:
            configpath = modelpath.replace('.h5', '_config.joblib').replace('.keras', '_config.joblib')
        
        if Path(configpath).exists():
            config = joblib.load(configpath)
            self.input_dim = config['input_dim']
            self.encoding_dim = config['encoding_dim']
            self.reconstruction_threshold = config['reconstruction_threshold']
            self.threshold_percentile = config['threshold_percentile']
        
        logger.info(f"Model loaded from {modelpath}")
    
    def plot_training_history(self) -> None:
        """Visualize training history (loss and MAE)"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('MSE Loss', fontweight='bold')
        axes[0].set_title('Autoencoder Training Loss', fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('MAE', fontweight='bold')
        axes[1].set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'threshold_percentile': self.threshold_percentile,
            'reconstruction_threshold': float(self.reconstruction_threshold) if self.reconstruction_threshold else None,
            'is_trained': self.is_trained,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
