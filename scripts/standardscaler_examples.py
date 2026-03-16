"""
Example: Using StandardScaler Normalization for Feature Preprocessing

Demonstrates how to normalize numerical features using the FeatureNormalizer
class with scikit-learn's StandardScaler for anomaly detection models.
"""

import pandas as pd
import numpy as np
from app.preprocessing.feature_engineering import FeatureNormalizer

# ============================================================================
# EXAMPLE 1: Basic Normalization Workflow
# ============================================================================

def example_basic_normalization():
    """Demonstrate basic normalization workflow"""
    
    print("=" * 80)
    print("EXAMPLE 1: Basic StandardScaler Normalization")
    print("=" * 80)
    
    # Sample healthcare data
    X_train = pd.DataFrame({
        'HR': np.random.normal(75, 12, 100),      # Heart rate
        'SpO2': np.random.normal(97, 2, 100),     # Oxygen saturation
        'Temp': np.random.normal(37, 0.8, 100),   # Temperature
        'SysBP': np.random.normal(120, 15, 100),  # Systolic BP
        'DiaBP': np.random.normal(80, 10, 100)    # Diastolic BP
    })
    
    print("\nOriginal Features (first 5 rows):")
    print(X_train.head())
    print(f"\nOriginal Statistics:")
    print(f"  Means: {X_train.mean().round(2).to_dict()}")
    print(f"  Stds: {X_train.std().round(2).to_dict()}")
    
    # Initialize and fit normalizer
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train)
    
    # Transform data
    X_train_normalized = normalizer.transform(X_train)
    
    print("\n\nNormalized Features (first 5 rows):")
    print(X_train_normalized.head())
    print(f"\nNormalized Statistics:")
    print(f"  Means: {X_train_normalized.mean().round(6).to_dict()}")
    print(f"  Stds: {X_train_normalized.std().round(4).to_dict()}")
    
    print("\n✓ All features centered at 0 with unit variance")


# ============================================================================
# EXAMPLE 2: Train-Test Workflow
# ============================================================================

def example_train_test_workflow():
    """Demonstrate proper train-test normalization"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Train-Test Normalization Workflow")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'HR': np.random.normal(75, 12, 80),
        'SpO2': np.random.normal(97, 2, 80),
        'Temp': np.random.normal(37, 0.8, 80),
        'SysBP': np.random.normal(120, 15, 80),
        'DiaBP': np.random.normal(80, 10, 80)
    })
    
    X_test = pd.DataFrame({
        'HR': np.random.normal(73, 11, 20),  # Slightly different distribution
        'SpO2': np.random.normal(96.5, 2.5, 20),
        'Temp': np.random.normal(37.2, 0.9, 20),
        'SysBP': np.random.normal(118, 14, 20),
        'DiaBP': np.random.normal(79, 11, 20)
    })
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Fit on training data ONLY
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train)
    
    # Transform both sets with same parameters
    X_train_normalized = normalizer.transform(X_train)
    X_test_normalized = normalizer.transform(X_test)
    
    print(f"\nTrained on: Training set ({X_train.shape[0]} samples)")
    print(f"  Scaling parameters: {normalizer.get_scaling_params()}")
    
    print(f"\nTraining set normalized stats:")
    print(f"  Means: {X_train_normalized.mean().round(6).to_dict()}")
    print(f"  Stds: {X_train_normalized.std().round(4).to_dict()}")
    
    print(f"\nTest set normalized stats (using training params):")
    print(f"  Means: {X_test_normalized.mean().round(6).to_dict()}")
    print(f"  Stds: {X_test_normalized.std().round(4).to_dict()}")
    
    print("\n✓ Both sets use same scaling parameters from training data")


# ============================================================================
# EXAMPLE 3: Scaler Persistence
# ============================================================================

def example_scaler_persistence():
    """Demonstrate saving and loading scaler for production"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Scaler Persistence for Production")
    print("=" * 80)
    
    import os
    
    # Training data
    X_train = pd.DataFrame({
        'HR': np.random.normal(75, 12, 100),
        'SpO2': np.random.normal(97, 2, 100),
        'Temp': np.random.normal(37, 0.8, 100),
        'SysBP': np.random.normal(120, 15, 100),
        'DiaBP': np.random.normal(80, 10, 100)
    })
    
    # Fit and save
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train)
    
    scaler_path = 'models/scalers/example_scaler.joblib'
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    normalizer.save(scaler_path)
    
    print(f"\n✓ Scaler saved to: {scaler_path}")
    print(f"  File size: {os.path.getsize(scaler_path)} bytes")
    
    # Load and use
    normalizer_prod = FeatureNormalizer()
    normalizer_prod.load(scaler_path)
    
    print(f"\n✓ Scaler loaded successfully")
    
    # New data in production
    X_new = pd.DataFrame({
        'HR': [72, 78],
        'SpO2': [96.5, 98],
        'Temp': [37.1, 36.9],
        'SysBP': [118, 125],
        'DiaBP': [78, 82]
    })
    
    X_new_normalized = normalizer_prod.transform(X_new)
    print(f"\nNew production data normalized:")
    print(X_new_normalized)
    
    # Cleanup
    if os.path.exists(scaler_path):
        os.remove(scaler_path)
        os.rmdir(os.path.dirname(scaler_path))


# ============================================================================
# EXAMPLE 4: Missing Data Handling
# ============================================================================

def example_missing_data_handling():
    """Demonstrate automatic missing data handling"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Missing Data Handling")
    print("=" * 80)
    
    # Data with missing values
    X_with_missing = pd.DataFrame({
        'HR': [70, 75, np.nan, 80, 72],
        'SpO2': [97, np.nan, 96.5, 98, 97.2],
        'Temp': [37, 36.9, 37.1, np.nan, 37.2],
        'SysBP': [115, 120, 118, 125, np.nan],
        'DiaBP': [75, 78, 72, 80, 79]
    })
    
    print("\nData with missing values:")
    print(X_with_missing)
    print(f"\nMissing counts: {X_with_missing.isnull().sum().to_dict()}")
    
    # Normalize (missing values auto-imputed with mean)
    normalizer = FeatureNormalizer()
    normalizer.fit(X_with_missing)
    X_normalized = normalizer.transform(X_with_missing)
    
    print("\nNormalized data (missing values imputed):")
    print(X_normalized)
    print(f"\nMissing after normalization: {X_normalized.isnull().sum().sum()}")
    
    print("\n✓ Missing values automatically handled by mean imputation")


# ============================================================================
# EXAMPLE 5: Feature Scaling Parameters
# ============================================================================

def example_scaling_parameters():
    """Demonstrate accessing scaling parameters"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Accessing Scaling Parameters")
    print("=" * 80)
    
    X_train = pd.DataFrame({
        'HR': np.random.normal(75, 12, 1000),
        'SpO2': np.random.normal(97, 2, 1000),
        'Temp': np.random.normal(37, 0.8, 1000),
    })
    
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train)
    
    # Get scaling parameters
    params = normalizer.get_scaling_params()
    
    print("\nScaling Parameters (Mean & Std from training data):")
    for feature, param_dict in params.items():
        print(f"\n{feature}:")
        print(f"  Mean (μ): {param_dict['mean']:.4f}")
        print(f"  Std (σ): {param_dict['std']:.4f}")
        print(f"  Formula: z = (x - {param_dict['mean']:.4f}) / {param_dict['std']:.4f}")
    
    # Get arrays directly
    means = normalizer.get_feature_means()
    stds = normalizer.get_feature_stds()
    
    print(f"\n\nDirect arrays:")
    print(f"  Means: {means}")
    print(f"  Stds: {stds}")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == '__main__':
    example_basic_normalization()
    example_train_test_workflow()
    example_scaler_persistence()
    example_missing_data_handling()
    example_scaling_parameters()
    
    print("\n" + "=" * 80)
    print("✓ All examples completed successfully!")
    print("=" * 80)
