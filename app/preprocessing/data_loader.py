import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HealthcareDataLoader:
    """Load and validate healthcare dataset"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to the dataset file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load healthcare dataset from CSV
        
        Returns:
            DataFrame containing healthcare data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully with shape {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def validate_data(self) -> dict:
        """
        Validate dataset integrity and consistency
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        logger.info(f"Data validation complete: {validation_results}")
        return validation_results
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Get statistical summary of the dataset
        
        Returns:
            DataFrame containing statistical summary
        """
        return self.df.describe()
    
    def handle_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        if strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        logger.info(f"Missing values handled using {strategy} strategy")
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from dataset
        
        Returns:
            DataFrame with duplicates removed
        """
        duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
        return self.df
