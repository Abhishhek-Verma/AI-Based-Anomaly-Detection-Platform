import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess healthcare data for model training"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def split_features_target(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Split features and target variable
        
        Args:
            df: Input dataframe
            target_col: Target column name (optional)
            
        Returns:
            Tuple of features array and target array
        """
        # Remove non-numeric columns if present
        numeric_df = df.select_dtypes(include=[np.number])
        self.feature_names = numeric_df.columns.tolist()
        
        X = numeric_df.values
        y = None
        
        if target_col and target_col in df.columns:
            y = df[target_col].values
        
        logger.info(f"Features extracted: {len(self.feature_names)} features")
        return X, y
    
    def standardize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Standardize features using StandardScaler
        
        Args:
            X: Feature array
            fit: Whether to fit the scaler
            
        Returns:
            Standardized feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        logger.info("Features standardized")
        return X_scaled
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range
        
        Args:
            X: Feature array
            
        Returns:
            Normalized feature array
        """
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        logger.info("Features normalized to [0, 1]")
        return X_normalized
    
    def remove_outliers(self, X: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from features
        
        Args:
            X: Feature array
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of cleaned features and outlier indices
        """
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
            outliers = np.where(~mask)[0]
        
        elif method == 'zscore':
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            mask = np.all(z_scores < threshold, axis=1)
            outliers = np.where(~mask)[0]
        
        X_cleaned = X[mask]
        logger.info(f"Removed {len(outliers)} outliers using {method} method")
        return X_cleaned, outliers
    
    def get_feature_names(self) -> list:
        """Get feature names"""
        return self.feature_names
