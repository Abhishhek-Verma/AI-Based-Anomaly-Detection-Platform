"""
Model training and evaluation script
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config
from src.preprocessing.data_loader import HealthcareDataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.autoencoder import AutoencoderAnomalyDetector
from src.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def train_isolation_forest(X_train: np.ndarray, X_test: np.ndarray) -> dict:
    """
    Train Isolation Forest model
    
    Args:
        X_train: Training data
        X_test: Test data
        
    Returns:
        Training results
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ISOLATION FOREST MODEL")
    logger.info("=" * 60)
    
    # Initialize and train model
    model = IsolationForestAnomalyDetector(contamination=0.1)
    model.train(X_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Get anomaly scores
    train_scores = model.predict_proba(X_train)
    test_scores = model.predict_proba(X_test)
    
    # Save model
    config = get_config()
    model.save_model(config.MODEL_ISOLATION_FOREST_PATH)
    
    results = {
        'model': model,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'train_anomalies': np.sum(train_predictions == -1),
        'test_anomalies': np.sum(test_predictions == -1)
    }
    
    logger.info(f"Training anomalies found: {results['train_anomalies']}")
    logger.info(f"Test anomalies found: {results['test_anomalies']}")
    
    return results


def train_autoencoder(X_train: np.ndarray, X_test: np.ndarray) -> dict:
    """
    Train Autoencoder model
    
    Args:
        X_train: Training data
        X_test: Test data
        
    Returns:
        Training results
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING AUTOENCODER MODEL")
    logger.info("=" * 60)
    
    # Initialize model
    model = AutoencoderAnomalyDetector(
        input_dim=X_train.shape[1],
        encoding_dim=8,
        epochs=50
    )
    
    # Train model
    logger.info("Training autoencoder...")
    model.train(X_train, X_test, verbose=1)
    
    # Calibrate threshold
    logger.info("Calibrating anomaly threshold...")
    threshold = model.calibrate_threshold(X_train, percentile=95)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Get reconstruction errors
    train_errors = model.get_reconstruction_error(X_train)
    test_errors = model.get_reconstruction_error(X_test)
    
    # Save model
    config = get_config()
    model.save_model(config.MODEL_AUTOENCODER_PATH)
    
    results = {
        'model': model,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'train_errors': train_errors,
        'test_errors': test_errors,
        'threshold': threshold,
        'train_anomalies': np.sum(train_predictions == 0),
        'test_anomalies': np.sum(test_predictions == 0)
    }
    
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Training anomalies found: {results['train_anomalies']}")
    logger.info(f"Test anomalies found: {results['test_anomalies']}")
    
    return results


def main():
    """Main training script"""
    logger.info("Starting model training pipeline")
    
    config = get_config()
    data_path = os.path.join(config.DATA_PATH, config.DATASET_NAME)
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please add your healthcare dataset to the data directory")
        return
    
    # Load data
    loader = HealthcareDataLoader(data_path)
    df = loader.load_data()
    
    # Validate data
    loader.validate_data()
    loader.handle_missing_values(strategy='mean')
    loader.remove_duplicates()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, _ = preprocessor.split_features_target(df)
    
    # Handle outliers
    X_cleaned, _ = preprocessor.remove_outliers(X, method='iqr')
    
    # Standardize features
    X_scaled = preprocessor.standardize(X_cleaned, fit=True)
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    # Train models
    if_results = train_isolation_forest(X_train, X_test)
    ae_results = train_autoencoder(X_train, X_test)
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
