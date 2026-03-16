"""
Combined Anomaly Detector and Severity Classifier
Merges Autoencoder and Isolation Forest scores for robust anomaly detection
Classifies abnormalities into LOW, MEDIUM, and HIGH severity levels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class CombinedAnomalyDetector:
    """
    Combines predictions from Autoencoder and Isolation Forest models.
    Uses ensemble approach for robust anomaly detection.
    """
    
    def __init__(self, autoencoder, isolation_forest,
                 method: str = 'weighted_average',
                 autoencoder_weight: float = 0.5,
                 isolation_weight: float = 0.5):
        """
        Initialize combined anomaly detector.
        
        Args:
            autoencoder: Trained AutoencoderAnomalyDetector instance
            isolation_forest: Trained IsolationForestAnomalyDetector instance
            method: Combination method ('weighted_average', 'max', 'min', 'voting')
            autoencoder_weight: Weight for autoencoder score (0-1)
            isolation_weight: Weight for isolation forest score (0-1)
        """
        self.autoencoder = autoencoder
        self.isolation_forest = isolation_forest
        self.method = method
        self.autoencoder_weight = autoencoder_weight
        self.isolation_weight = isolation_weight
        
        # Normalize weights
        total_weight = autoencoder_weight + isolation_weight
        self.autoencoder_weight /= total_weight
        self.isolation_weight /= total_weight
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get combined predictions from both models.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with combined anomaly scores and predictions
        """
        # Get predictions from both models
        ae_pred = self.autoencoder.predict(X)
        if_pred = self.isolation_forest.predict(X)
        
        ae_score = ae_pred['anomaly_score']
        if_score = if_pred['anomaly_score']
        
        # Combine scores based on method
        if self.method == 'weighted_average':
            combined_score = (self.autoencoder_weight * ae_score + 
                            self.isolation_weight * if_score)
        
        elif self.method == 'max':
            combined_score = np.maximum(ae_score, if_score)
        
        elif self.method == 'min':
            combined_score = np.minimum(ae_score, if_score)
        
        elif self.method == 'voting':
            # Both models agree if both score > 0.5
            ae_anomaly = (ae_score > 0.5).astype(int)
            if_anomaly = (if_score > 0.5).astype(int)
            vote_sum = ae_anomaly + if_anomaly
            combined_score = vote_sum / 2.0
        
        else:
            raise ValueError(f"Unknown combination method: {self.method}")
        
        return {
            'combined_score': combined_score,
            'autoencoder_score': ae_score,
            'isolation_forest_score': if_score,
            'autoencoder_error': ae_pred.get('reconstruction_error'),
            'method': self.method
        }


class SeverityClassifier:
    """
    Classify anomaly scores into severity levels.
    Enables prioritized health risk detection (LOW, MEDIUM, HIGH).
    """
    
    def __init__(self, low_threshold: float = 0.33,
                 medium_threshold: float = 0.67):
        """
        Initialize severity classifier.
        
        Args:
            low_threshold: Threshold between LOW and MEDIUM (0-1)
            medium_threshold: Threshold between MEDIUM and HIGH (0-1)
        """
        if not (0 < low_threshold < medium_threshold < 1):
            raise ValueError("Thresholds must satisfy: 0 < low < medium < 1")
        
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
    
    def classify(self, anomaly_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify anomaly scores into severity levels.
        
        Args:
            anomaly_scores: Array of anomaly scores (0-1)
            
        Returns:
            Tuple of (severity_levels, severity_codes)
            - severity_levels: 'LOW', 'MEDIUM', or 'HIGH'
            - severity_codes: 0 (LOW), 1 (MEDIUM), 2 (HIGH)
        """
        severity_codes = np.zeros(len(anomaly_scores), dtype=int)
        
        # Classify scores
        severity_codes[anomaly_scores >= self.medium_threshold] = 2  # HIGH
        severity_codes[(anomaly_scores >= self.low_threshold) & 
                      (anomaly_scores < self.medium_threshold)] = 1  # MEDIUM
        # Scores < low_threshold remain 0 (LOW)
        
        # Map codes to level names
        severity_names = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        severity_levels = np.array([severity_names[code] for code in severity_codes])
        
        return severity_levels, severity_codes
    
    def classify_with_logic(self, anomaly_scores: np.ndarray,
                           variance: Optional[np.ndarray] = None,
                           trend: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Classify with additional context (variance, trend).
        Accounts for sustained abnormalities (trends) and variability.
        
        Args:
            anomaly_scores: Base anomaly scores
            variance: Feature variance (optional)
            trend: Temporal trend indicator (optional)
            
        Returns:
            Dictionary with detailed severity classification
        """
        severity_levels, severity_codes = self.classify(anomaly_scores)
        
        # Boost severity if high variance + trend
        adjustment_factor = np.ones(len(anomaly_scores))
        
        if variance is not None:
            # High variance increases severity
            high_variance = variance > np.percentile(variance, 75)
            adjustment_factor[high_variance] *= 1.1
        
        if trend is not None:
            # Large trends (sustained changes) increase severity
            large_trend = np.abs(trend) > np.percentile(np.abs(trend), 75)
            adjustment_factor[large_trend] *= 1.15
        
        # Apply adjustments and reclassify
        adjusted_scores = np.clip(anomaly_scores * adjustment_factor, 0, 1)
        final_levels, final_codes = self.classify(adjusted_scores)
        
        return {
            'severity_level': final_levels,
            'severity_code': final_codes,
            'base_score': anomaly_scores,
            'adjusted_score': adjusted_scores,
            'adjustment_factor': adjustment_factor
        }
    
    def get_risk_distribution(self, anomaly_scores: np.ndarray) -> Dict[str, float]:
        """
        Get distribution of risk levels.
        
        Args:
            anomaly_scores: Array of anomaly scores
            
        Returns:
            Dictionary with counts and percentages for each severity level
        """
        _, codes = self.classify(anomaly_scores)
        
        total = len(codes)
        low_count = np.sum(codes == 0)
        med_count = np.sum(codes == 1)
        high_count = np.sum(codes == 2)
        
        return {
            'LOW': {'count': int(low_count), 'percentage': 100 * low_count / total},
            'MEDIUM': {'count': int(med_count), 'percentage': 100 * med_count / total},
            'HIGH': {'count': int(high_count), 'percentage': 100 * high_count / total},
            'total': int(total)
        }


class AnomalyDetectionPipeline:
    """
    Complete end-to-end anomaly detection pipeline.
    Combines preprocessing, windowing, model prediction, and severity classification.
    """
    
    def __init__(self, autoencoder, isolation_forest, 
                 scaler=None, temporal_builder=None):
        """
        Initialize anomaly detection pipeline.
        
        Args:
            autoencoder: Trained AutoencoderAnomalyDetector
            isolation_forest: Trained IsolationForestAnomalyDetector
            scaler: FeatureNormalizer for preprocessing
            temporal_builder: TemporalWindowBuilder for windowing
        """
        self.autoencoder = autoencoder
        self.isolation_forest = isolation_forest
        self.scaler = scaler
        self.temporal_builder = temporal_builder
        
        self.combined_detector = CombinedAnomalyDetector(
            autoencoder, isolation_forest
        )
        self.severity_classifier = SeverityClassifier()
    
    def process(self, X: np.ndarray,
                use_scaling: bool = True,
                use_windowing: bool = False) -> Dict:
        """
        Process data through complete pipeline.
        
        Args:
            X: Raw input features
            use_scaling: Whether to apply feature scaling
            use_windowing: Whether to apply temporal windowing
            
        Returns:
            Dictionary with complete analysis results
        """
        # Step 1: Scaling
        if use_scaling and self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X
        
        # Step 2: Get predictions
        predictions = self.combined_detector.predict(X_processed)
        
        # Step 3: Classify severity
        severity = self.severity_classifier.classify(
            predictions['combined_score']
        )
        severity_levels, severity_codes = severity
        
        # Step 4: Risk distribution
        risk_dist = self.severity_classifier.get_risk_distribution(
            predictions['combined_score']
        )
        
        return {
            'anomaly_scores': predictions['combined_score'],
            'autoencoder_scores': predictions['autoencoder_score'],
            'isolation_forest_scores': predictions['isolation_forest_score'],
            'severity_level': severity_levels,
            'severity_code': severity_codes,
            'risk_distribution': risk_dist,
            'num_samples': len(X),
            'num_anomalies': np.sum(severity_codes >= 1),
            'num_high_risk': np.sum(severity_codes == 2)
        }
    
    def save_all_models(self, directory: str) -> None:
        """Save all components of the pipeline"""
        from pathlib import Path
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.autoencoder.save(f"{directory}/autoencoder.h5")
        self.isolation_forest.save_model(f"{directory}/isolation_forest.joblib")
        
        if self.scaler is not None:
            self.scaler.save(f"{directory}/scaler.joblib")
        
        logger.info(f"All models saved to {directory}")
