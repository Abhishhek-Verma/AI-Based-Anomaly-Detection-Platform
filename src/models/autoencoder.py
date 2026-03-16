import numpy as np
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class AutoencoderAnomalyDetector:
    """Autoencoder model for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 8, epochs: int = 50, batch_size: int = 32):
        """
        Initialize Autoencoder
        
        Args:
            input_dim: Input dimension (number of features)
            encoding_dim: Dimension of encoded representation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        self.is_trained = False
        self.threshold = None
        
    def _build_model(self):
        """
        Build autoencoder architecture
        
        Returns:
            Compiled Keras model
        """
        # Encoder
        input_data = tf.keras.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(16, activation='relu')(input_data)
        encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = tf.keras.Model(input_data, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        logger.info("Autoencoder model built successfully")
        return autoencoder
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None, verbose: int = 0) -> None:
        """
        Train the autoencoder
        
        Args:
            X_train: Training data
            X_val: Validation data
            verbose: Training verbosity
        """
        if X_val is None:
            X_val = X_train
        
        history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, X_val),
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Autoencoder trained successfully")
        return history
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction error
        
        Args:
            X: Input data
            
        Returns:
            Array of reconstruction errors
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
        return mse
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Input data
            threshold: Anomaly threshold
            
        Returns:
            Array of predictions (0 for anomaly, 1 for normal)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        errors = self.get_reconstruction_error(X)
        
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Please provide threshold or set it during calibration")
            threshold = self.threshold
        
        predictions = (errors < threshold).astype(int)
        return predictions
    
    def calibrate_threshold(self, X: np.ndarray, percentile: float = 95) -> float:
        """
        Calibrate anomaly detection threshold
        
        Args:
            X: Calibration data
            percentile: Percentile for threshold
            
        Returns:
            Computed threshold
        """
        errors = self.get_reconstruction_error(X)
        self.threshold = np.percentile(errors, percentile)
        logger.info(f"Threshold calibrated to {self.threshold}")
        return self.threshold
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model
        
        Args:
            filepath: Path to load the model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
