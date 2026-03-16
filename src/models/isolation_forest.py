import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Isolation Forest model for anomaly detection"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest model
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_trained = False
        
    def train(self, X: np.ndarray) -> None:
        """
        Train the Isolation Forest model
        
        Args:
            X: Feature array for training
        """
        self.model.fit(X)
        self.is_trained = True
        logger.info("Isolation Forest model trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Feature array for prediction
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores
        
        Args:
            X: Feature array
            
        Returns:
            Array of anomaly scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get anomaly scores (negative means more anomalous)
        scores = self.model.score_samples(X)
        return scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to load the model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
