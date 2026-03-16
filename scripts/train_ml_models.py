"""
Complete ML Models Training Pipeline
Demonstrates the full anomaly detection workflow with logging and model persistence
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.preprocessing.feature_engineering import FeatureNormalizer
from app.models.temporal_patterns import TemporalFeatureExtractor
from app.models.autoencoder import AutoencoderAnomalyDetector
from app.models.isolation_forest import IsolationForestAnomalyDetector
from app.models.combined_detector import CombinedAnomalyDetector, SeverityClassifier, AnomalyDetectionPipeline


def create_synthetic_vital_signs(n_samples: int = 1000, n_patients: int = 10, 
                                 anomaly_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic vital signs data for demonstration/testing.
    
    Args:
        n_samples: Total number of time steps
        n_patients: Number of virtual patients
        anomaly_rate: Fraction of samples to mark as anomalies
    
    Returns:
        X: Feature matrix (n_samples, 5 features)
        y_true: Binary labels (1=anomaly, 0=normal)
    """
    logger.info(f"Generating synthetic vital signs data: {n_samples} samples from {n_patients} patients")
    
    # Feature names and normal ranges (HR, SpO2, Temp, SysBP, DiaBP)
    features = ['HR', 'SpO2', 'Temperature', 'SysBP', 'DiaBP']
    normal_ranges = {
        'HR': (60, 100),
        'SpO2': (95, 100),
        'Temperature': (36.5, 37.5),
        'SysBP': (90, 120),
        'DiaBP': (60, 80)
    }
    
    X = np.zeros((n_samples, 5))
    y_true = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        patient_id = i % n_patients
        
        # Generate normal vital signs with patient-specific patterns
        if np.random.random() > anomaly_rate:
            # Normal pattern
            for j, (feature, (low, high)) in enumerate(normal_ranges.items()):
                # Add patient-specific baseline
                baseline = low + patient_id * (high - low) / n_patients
                X[i, j] = baseline + np.random.normal(0, (high - low) / 10)
        else:
            # Anomalous pattern
            y_true[i] = 1
            for j, (feature, (low, high)) in enumerate(normal_ranges.items()):
                # Generate out-of-range values
                X[i, j] = np.random.choice([
                    np.random.normal(low - 10, 5),  # Too low
                    np.random.normal(high + 10, 5)   # Too high
                ])
    
    # Clip to reasonable bounds
    X = np.clip(X, 0, 200)
    
    logger.info(f"✓ Generated data shape: {X.shape}")
    logger.info(f"  - Anomalies: {y_true.sum()} ({100*y_true.sum()/len(y_true):.1f}%)")
    
    return X, y_true


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, 
                       X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FeatureNormalizer]:
    """
    Normalize features using StandardScaler.
    
    Args:
        X_train, X_val, X_test: Feature matrices
    
    Returns:
        Normalized feature matrices and fitted scaler
    """
    logger.info("Normalizing features with StandardScaler...")
    
    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    X_test_norm = normalizer.transform(X_test)
    
    logger.info(f"✓ Features normalized")
    logger.info(f"  - Train mean: {X_train_norm.mean():.4f}, std: {X_train_norm.std():.4f}")
    logger.info(f"  - Val mean: {X_val_norm.mean():.4f}, std: {X_val_norm.std():.4f}")
    logger.info(f"  - Test mean: {X_test_norm.mean():.4f}, std: {X_test_norm.std():.4f}")
    
    return X_train_norm, X_val_norm, X_test_norm, normalizer


def create_temporal_windows(X: np.ndarray, window_size: int = 10, 
                           stride: int = 2) -> np.ndarray:
    """
    Create sliding window temporal features.
    
    Args:
        X: Feature matrix
        window_size: Number of timesteps per window
        stride: Step size between windows
    
    Returns:
        Windowed feature matrix (flattened)
    """
    logger.info(f"Creating temporal windows (size={window_size}, stride={stride})...")
    
    extractor = TemporalFeatureExtractor(window_size=window_size, stride=stride)
    X_windowed = extractor.extract_features(X)
    
    logger.info(f"✓ Windowed data shape: {X_windowed.shape}")
    
    return X_windowed


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, 
                     input_dim: int = 50, encoding_dim: int = 16) -> AutoencoderAnomalyDetector:
    """
    Build and train Autoencoder model.
    
    Args:
        X_train, X_val: Training and validation data
        input_dim: Input feature dimension
        encoding_dim: Bottleneck dimension
    
    Returns:
        Trained Autoencoder instance
    """
    logger.info(f"Training Autoencoder (input_dim={input_dim}, encoding_dim={encoding_dim})...")
    
    ae = AutoencoderAnomalyDetector(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        threshold_percentile=95.0,
        epochs=30,
        batch_size=32
    )
    
    ae.build_model()
    ae.train(X_train, X_val=X_val, verbose=0)
    
    logger.info(f"✓ Autoencoder trained successfully")
    logger.info(f"  - Reconstruction threshold: {ae.reconstruction_threshold:.4f}")
    logger.info(f"  - Training loss: {ae.history['loss'][-1]:.6f}")
    
    return ae


def train_isolation_forest(X_train: np.ndarray, 
                          contamination: float = 0.05) -> IsolationForestAnomalyDetector:
    """
    Train Isolation Forest model.
    
    Args:
        X_train: Training data
        contamination: Fraction of expected anomalies
    
    Returns:
        Trained Isolation Forest instance
    """
    logger.info(f"Training Isolation Forest (contamination={contamination})...")
    
    iso_forest = IsolationForestAnomalyDetector(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.train(X_train)
    
    logger.info(f"✓ Isolation Forest trained successfully")
    
    return iso_forest


def evaluate_models(ae: AutoencoderAnomalyDetector, 
                   iso_forest: IsolationForestAnomalyDetector,
                   X_test: np.ndarray, y_true: np.ndarray) -> None:
    """
    Evaluate and compare models on test set.
    
    Args:
        ae: Trained Autoencoder
        iso_forest: Trained Isolation Forest
        X_test: Test data
        y_true: True anomaly labels
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Get scores
    ae_scores = ae.predict_anomaly_score(X_test)
    if_scores = iso_forest.predict_anomaly_score(X_test)
    
    # Binary predictions
    ae_preds = (ae_scores >= 0.5).astype(int)
    if_preds = (if_scores >= 0.5).astype(int)
    
    # Statistics
    logger.info(f"\nAutoencoder Anomaly Scores:")
    logger.info(f"  - Min: {ae_scores.min():.4f}, Mean: {ae_scores.mean():.4f}, Max: {ae_scores.max():.4f}")
    logger.info(f"  - Detected anomalies: {ae_preds.sum()} / {len(ae_preds)} ({100*ae_preds.sum()/len(ae_preds):.1f}%)")
    
    logger.info(f"\nIsolation Forest Anomaly Scores:")
    logger.info(f"  - Min: {if_scores.min():.4f}, Mean: {if_scores.mean():.4f}, Max: {if_scores.max():.4f}")
    logger.info(f"  - Detected anomalies: {if_preds.sum()} / {len(if_preds)} ({100*if_preds.sum()/len(if_preds):.1f}%)")
    
    # Correlation
    correlation = np.corrcoef(ae_scores, if_scores)[0, 1]
    logger.info(f"\nModel Score Correlation: {correlation:.4f}")
    logger.info(f"  → Low correlation indicates complementary detection capabilities")


def save_models(ae: AutoencoderAnomalyDetector, 
               iso_forest: IsolationForestAnomalyDetector,
               normalizer: FeatureNormalizer,
               models_dir: str = "models") -> None:
    """
    Save trained models to disk.
    
    Args:
        ae: Trained Autoencoder
        iso_forest: Trained Isolation Forest
        normalizer: Fitted scaler
        models_dir: Directory to save models
    """
    logger.info(f"\nSaving models to {models_dir}/...")
    
    Path(models_dir).mkdir(exist_ok=True)
    
    # Save Autoencoder
    ae_path = f"{models_dir}/autoencoder_model.h5"
    ae.save(ae_path)
    logger.info(f"  ✓ Saved Autoencoder to {ae_path}")
    
    # Save Isolation Forest
    if_path = f"{models_dir}/isolation_forest_model.joblib"
    iso_forest.save_model(if_path)
    logger.info(f"  ✓ Saved Isolation Forest to {if_path}")
    
    # Save Normalizer
    normalizer_path = f"{models_dir}/feature_normalizer.joblib"
    normalizer.save(normalizer_path)
    logger.info(f"  ✓ Saved Feature Normalizer to {normalizer_path}")


def main():
    """
    Execute complete ML models training pipeline.
    """
    logger.info("=" * 60)
    logger.info("ML MODELS TRAINING PIPELINE - START")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        # 1. Generate/load data
        X, y_true = create_synthetic_vital_signs(n_samples=1000, n_patients=10, anomaly_rate=0.05)
        
        # 2. Split data
        train_idx = int(0.6 * len(X))
        val_idx = int(0.8 * len(X))
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y_true[:train_idx], y_true[train_idx:val_idx], y_true[val_idx:]
        logger.info(f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
        
        # 3. Normalize features
        X_train_norm, X_val_norm, X_test_norm, normalizer = normalize_features(X_train, X_val, X_test)
        
        # 4. Create temporal windows
        X_train_wind = create_temporal_windows(X_train_norm)
        X_val_wind = create_temporal_windows(X_val_norm)
        X_test_wind = create_temporal_windows(X_test_norm)
        
        # 5. Train models
        ae = train_autoencoder(X_train_wind, X_val_wind, 
                              input_dim=X_train_wind.shape[1], encoding_dim=16)
        iso_forest = train_isolation_forest(X_train_wind, contamination=0.05)
        
        # 6. Evaluate models
        evaluate_models(ae, iso_forest, X_test_wind, y_test)
        
        # 7. Generate combined scores (example)
        logger.info(f"\n{'='*60}")
        logger.info("ENSEMBLE PREDICTION (Example)")
        logger.info(f"{'='*60}")
        
        ae_scores = ae.predict_anomaly_score(X_test_wind)
        if_scores = iso_forest.predict_anomaly_score(X_test_wind)
        combined_scores = 0.5 * ae_scores + 0.5 * if_scores
        
        logger.info(f"Combined anomaly scores (test set):")
        logger.info(f"  - Min: {combined_scores.min():.4f}")
        logger.info(f"  - Mean: {combined_scores.mean():.4f}")
        logger.info(f"  - Max: {combined_scores.max():.4f}")
        
        # 8. Classify severity
        logger.info(f"\n{'='*60}")
        logger.info("SEVERITY CLASSIFICATION (Example)")
        logger.info(f"{'='*60}")
        
        severity_classifier = SeverityClassifier(low_threshold=0.33, medium_threshold=0.67)
        severity_labels = severity_classifier.classify(combined_scores)
        
        unique, counts = np.unique(severity_labels, return_counts=True)
        for severity, count in zip(unique, counts):
            pct = 100 * count / len(severity_labels)
            logger.info(f"  - {severity}: {count} samples ({pct:.1f}%)")
        
        # 9. Save models
        save_models(ae, iso_forest, normalizer, models_dir="models")
        
        logger.info(f"\n{'='*60}")
        logger.info("ML MODELS TRAINING PIPELINE - COMPLETE")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
