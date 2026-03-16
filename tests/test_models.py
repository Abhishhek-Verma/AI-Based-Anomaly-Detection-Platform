"""
Unit tests for anomaly detection models
"""

import pytest
import numpy as np
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.autoencoder import AutoencoderAnomalyDetector


class TestIsolationForest:
    """Test cases for IsolationForestAnomalyDetector"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        return X
    
    def test_training(self, sample_data):
        """Test model training"""
        model = IsolationForestAnomalyDetector()
        model.train(sample_data)
        
        assert model.is_trained is True
    
    def test_prediction(self, sample_data):
        """Test model prediction"""
        model = IsolationForestAnomalyDetector()
        model.train(sample_data)
        
        predictions = model.predict(sample_data)
        
        assert predictions.shape[0] == sample_data.shape[0]
        assert all(p in [-1, 1] for p in predictions)
    
    def test_anomaly_scores(self, sample_data):
        """Test anomaly scores"""
        model = IsolationForestAnomalyDetector()
        model.train(sample_data)
        
        scores = model.predict_proba(sample_data)
        
        assert scores.shape[0] == sample_data.shape[0]


class TestAutoencoder:
    """Test cases for AutoencoderAnomalyDetector"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        return X
    
    def test_training(self, sample_data):
        """Test model training"""
        model = AutoencoderAnomalyDetector(input_dim=5, epochs=2)
        model.train(sample_data, sample_data, verbose=0)
        
        assert model.is_trained is True
    
    def test_reconstruction_error(self, sample_data):
        """Test reconstruction error calculation"""
        model = AutoencoderAnomalyDetector(input_dim=5, epochs=2)
        model.train(sample_data, sample_data, verbose=0)
        
        errors = model.get_reconstruction_error(sample_data)
        
        assert errors.shape[0] == sample_data.shape[0]
        assert all(e >= 0 for e in errors)
    
    def test_threshold_calibration(self, sample_data):
        """Test threshold calibration"""
        model = AutoencoderAnomalyDetector(input_dim=5, epochs=2)
        model.train(sample_data, sample_data, verbose=0)
        
        threshold = model.calibrate_threshold(sample_data, percentile=95)
        
        assert threshold > 0
        assert model.threshold == threshold


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
