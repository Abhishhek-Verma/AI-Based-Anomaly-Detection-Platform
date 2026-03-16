"""
Isolation Forest for Anomaly Detection in High-Dimensional Vital Signs
Detects anomalies through isolation in random feature subspaces
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional
import logging

from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """
    Isolation Forest-based anomaly detection.
    Detects high-dimensional outliers by isolating anomalies in random subspaces.
    Complementary to reconstruction-based approaches like Autoencoders.
    """
    
    def __init__(self, contamination: float = 0.05, 
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies in training (0-1)
            n_estimators: Number of isolation trees in the ensemble
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X: np.ndarray) -> 'IsolationForestAnomalyDetector':
        """
        Train Isolation Forest on normal data.
        
        Args:
            X: Training features (num_samples, num_features)
            
        Returns:
            Self for method chaining
        """
        # Handle DataFrames
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model.fit(X)
        self.is_trained = True
        
        logger.info(f"IsolationForest trained on {X.shape[0]} samples, {X.shape[1]} features")
        return self
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict normalized anomaly scores (0-1).
        0 = normal, 1 = severe anomaly
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores in range [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get raw anomaly scores (-inf to 0, where smaller = more anomalous)
        raw_scores = self.model.score_samples(X)
        
        # Normalize to 0-1 scale
        # Typical range is approximately [-1, 0]
        # score of -1 -> anomaly_score of 0 (normal)
        # score of 0 -> anomaly_score of 1 (very normal)
        # We want: more negative = higher anomaly score
        # Formula: (1 + raw_score) / 1 = 1 + raw_score, then clip
        anomaly_scores = np.clip(1 + raw_scores, 0.0, 1.0)
        
        return anomaly_scores
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions including anomaly scores and binary flags.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with:
                - anomaly_score: Normalized scores (0-1)
                - is_anomaly: Boolean flags (True=anomaly, False=normal)
                - sklearn_prediction: Raw sklearn predctions (-1=anomaly, 1=normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained")
        
        # sklearn returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X)
        scores = self.predict_anomaly_score(X)
        
        return {
            'anomaly_score': scores,
            'is_anomaly': predictions == -1,
            'sklearn_prediction': predictions
        }
    
    def get_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw anomaly scores from the model.
        Lower scores indicate more anomalous samples.
        
        Args:
            X: Input features
            
        Returns:
            Raw anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return self.model.score_samples(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk using joblib.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"IsolationForest model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'IsolationForestAnomalyDetector':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self with loaded model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.contamination = model_data.get('contamination', 0.05)
        self.n_estimators = model_data.get('n_estimators', 100)
        self.random_state = model_data.get('random_state', 42)
        self.feature_names = model_data.get('feature_names')
        self.is_trained = True
        
        logger.info(f"IsolationForest model loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict:
        """Get model information and configuration"""
        return {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'num_features': self.model.n_features_in_ if self.is_trained else None,
            'feature_names': self.feature_names
        }
