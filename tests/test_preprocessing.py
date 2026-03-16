"""
Unit tests for data preprocessing
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.data_loader import HealthcareDataLoader
from src.preprocessing.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = {
            'feature_1': [1, 2, 3, 4, 5] * 10,
            'feature_2': [10, 20, 30, 40, 50] * 10,
            'feature_3': [100, 200, 300, 400, 500] * 10,
        }
        return pd.DataFrame(data)
    
    def test_split_features_target(self, sample_data):
        """Test feature-target splitting"""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.split_features_target(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == 3
        assert y is None
    
    def test_standardize(self, sample_data):
        """Test feature standardization"""
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.split_features_target(sample_data)
        
        X_scaled = preprocessor.standardize(X, fit=True)
        
        # Check if mean is close to 0 and std is close to 1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_remove_outliers(self, sample_data):
        """Test outlier removal"""
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.split_features_target(sample_data)
        
        # Add some outliers
        X_with_outliers = np.vstack([X, [[1000, 1000, 1000]]])
        
        X_cleaned, outliers = preprocessor.remove_outliers(X_with_outliers, method='iqr')
        
        assert len(X_cleaned) == len(X)
        assert len(outliers) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
