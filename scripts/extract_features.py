"""
Feature Extraction Script
Demonstrates feature engineering pipeline for anomaly detection
"""

import os
import sys
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config
from utils.logger import setup_logging
from app.preprocessing.data_loader import HealthcareDataLoader
from app.preprocessing.feature_engineering import (
    FeatureEngineer, FeatureSelector, FeatureNormalizer
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Execute feature extraction pipeline"""
    
    logger.info("=" * 90)
    logger.info("FEATURE ENGINEERING & EXTRACTION PIPELINE")
    logger.info("=" * 90)
    
    config = get_config()
    data_path = os.path.join(config.DATA_PATH, config.DATASET_NAME)
    
    # Load data
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please run: python scripts/generate_sample_data.py")
        return
    
    logger.info("\n[Step 1] Loading Healthcare Data...")
    loader = HealthcareDataLoader(data_path)
    df = loader.load_data()
    logger.info(f"✓ Dataset loaded: {df.shape[0]} records × {df.shape[1]} features")
    
    # Initialize feature engineer
    logger.info("\n[Step 2] Extracting Features...")
    fe = FeatureEngineer(df, time_window=5)
    
    # Extract all features
    all_features = fe.get_anomaly_features()
    logger.info(f"✓ Features extracted: {all_features.shape[1]} features")
    logger.info(f"  Features: {', '.join(all_features.columns.tolist())}")
    
    # Get feature summary
    feature_summary = fe.get_feature_summary()
    logger.info("\nFeature Statistics:")
    for feature, stats in list(feature_summary.items())[:5]:
        logger.info(f"  {feature:20} Mean={stats['mean']:8.2f} ± {stats['std']:6.2f}")
    
    # Select features
    logger.info("\n[Step 3] Selecting Core Features...")
    fs = FeatureSelector(all_features)
    selected_features = fs.select_core_features()
    logger.info(f"✓ Core features selected: {len(fs.selected_features)} features")
    logger.info(f"  Selected: {', '.join(fs.selected_features)}")
    
    # Feature statistics
    feature_stats = fs.get_feature_stats()
    logger.info("\nSelected Feature Statistics:")
    for feature, stats in feature_stats.items():
        logger.info(f"  {feature:15} Mean={stats['mean']:8.2f} "
                   f"Range=[{stats['min']:6.2f}, {stats['max']:6.2f}]")
    
    # Normalization
    logger.info("\n[Step 4] Normalizing Features...")
    normalizer = FeatureNormalizer()
    normalizer.fit(selected_features)
    
    # Apply normalization
    normalized_features = normalizer.normalize(selected_features, method='zscore')
    logger.info("✓ Features normalized (Z-score)")
    logger.info(f"  Mean: {normalized_features.mean().mean():.4f}")
    logger.info(f"  Std: {normalized_features.std().mean():.4f}")
    
    # Save features
    logger.info("\n[Step 5] Saving Processed Features...")
    features_output = os.path.join(config.DATA_PATH, 'features_engineered.csv')
    normalized_output = os.path.join(config.DATA_PATH, 'features_normalized.csv')
    
    all_features.to_csv(features_output, index=False)
    normalized_features.to_csv(normalized_output, index=False)
    
    logger.info(f"✓ Engineered features saved: {features_output}")
    logger.info(f"✓ Normalized features saved: {normalized_output}")
    
    # Summary report
    logger.info("\n" + "=" * 90)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 90)
    logger.info(f"\nDataset: {df.shape[0]} records")
    logger.info(f"Base features: {', '.join(list(df.columns)[:7])}")
    logger.info(f"Engineered features: {all_features.shape[1]} total")
    logger.info(f"Core features selected: {len(fs.selected_features)}")
    logger.info(f"Features for anomaly detection:")
    
    for feature in fs.selected_features:
        feature_name = {
            'HR': 'Heart Rate',
            'SpO2': 'Oxygen Saturation',
            'Temp': 'Body Temperature',
            'SysBP': 'Systolic Blood Pressure',
            'DiaBP': 'Diastolic Blood Pressure',
            'HRV': 'Heart Rate Variability',
            'MAP': 'Mean Arterial Pressure'
        }.get(feature, feature)
        
        logger.info(f"  • {feature:10} - {feature_name}")
    
    logger.info("\n✓ Pipeline ready for model training")
    logger.info("=" * 90)


if __name__ == '__main__':
    main()
