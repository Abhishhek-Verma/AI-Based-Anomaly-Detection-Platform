"""
Dataset validation and analysis script
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config
from src.preprocessing.data_loader import HealthcareDataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def validate_dataset(data_path: str) -> dict:
    """
    Validate healthcare dataset
    
    Args:
        data_path: Path to dataset
        
    Returns:
        Validation results dictionary
    """
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION AND ANALYSIS")
    logger.info("=" * 60)
    
    # Load data
    loader = HealthcareDataLoader(data_path)
    df = loader.load_data()
    
    # Validate data
    validation_results = loader.validate_data()
    
    logger.info("\n[DATASET OVERVIEW]")
    logger.info(f"Shape: {validation_results['shape']}")
    logger.info(f"Duplicates: {validation_results['duplicates']}")
    logger.info(f"Numeric columns: {validation_results['numeric_columns']}")
    logger.info(f"Categorical columns: {validation_results['categorical_columns']}")
    
    logger.info("\n[MISSING VALUES]")
    for col, count in validation_results['missing_values'].items():
        if count > 0:
            logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Get statistics
    logger.info("\n[STATISTICAL SUMMARY]")
    stats = loader.get_statistics()
    logger.info(f"\n{stats}")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        logger.info("\nHandling missing values...")
        df = loader.handle_missing_values(strategy='mean')
    
    # Remove duplicates
    if validation_results['duplicates'] > 0:
        logger.info("Removing duplicate rows...")
        df = loader.remove_duplicates()
    
    # Check for data quality
    logger.info("\n[DATA QUALITY CHECK]")
    logger.info(f"All non-null numeric data: {df[validation_results['numeric_columns']].isnull().sum().sum() == 0}")
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    
    return validation_results


if __name__ == '__main__':
    config = get_config()
    data_path = os.path.join(config.DATA_PATH, config.DATASET_NAME)
    
    if os.path.exists(data_path):
        validate_dataset(data_path)
    else:
        logger.warning(f"Dataset not found at {data_path}")
        logger.info("Please add your healthcare dataset to the data directory")
