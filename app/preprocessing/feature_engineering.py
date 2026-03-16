"""
Feature Engineering for Anomaly Detection
Extracts and derives physiological features from raw vital signs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import pickle
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Extract and derive features from vital signs"""
    
    def __init__(self, df: pd.DataFrame, patient_id_col: str = 'patient_id', 
                 time_window: int = 5):
        """
        Initialize feature engineer
        
        Args:
            df: Healthcare dataset
            patient_id_col: Patient identifier column
            time_window: Window size for rolling calculations
        """
        self.df = df.copy()
        self.patient_id_col = patient_id_col
        self.time_window = time_window
        self.features = {}
    
    def calculate_hrv(self, heart_rate_series: pd.Series) -> float:
        """
        Calculate Heart Rate Variability (HRV)
        Uses standard deviation of RR intervals as HRV measure
        
        Args:
            heart_rate_series: Series of heart rate values
            
        Returns:
            HRV value (standard deviation of heart rate)
        """
        if len(heart_rate_series) < 2:
            return np.nan
        
        return float(heart_rate_series.std())
    
    def calculate_map(self, systolic: float, diastolic: float) -> float:
        """
        Calculate Mean Arterial Pressure (MAP)
        MAP = (Systolic + 2*Diastolic) / 3
        
        Args:
            systolic: Systolic blood pressure (mmHg)
            diastolic: Diastolic blood pressure (mmHg)
            
        Returns:
            MAP value
        """
        if pd.isna(systolic) or pd.isna(diastolic):
            return np.nan
        
        return (systolic + 2 * diastolic) / 3
    
    def calculate_pulse_pressure(self, systolic: float, diastolic: float) -> float:
        """
        Calculate Pulse Pressure (PP)
        PP = Systolic - Diastolic
        
        Args:
            systolic: Systolic blood pressure
            diastolic: Diastolic blood pressure
            
        Returns:
            Pulse pressure value
        """
        if pd.isna(systolic) or pd.isna(diastolic):
            return np.nan
        
        return systolic - diastolic
    
    def calculate_rate_pressure_product(self, heart_rate: float, 
                                       systolic: float) -> float:
        """
        Calculate Rate Pressure Product (RPP)
        RPP = Heart Rate * Systolic BP
        Indicates myocardial oxygen demand
        
        Args:
            heart_rate: Heart rate (bpm)
            systolic: Systolic blood pressure (mmHg)
            
        Returns:
            RPP value
        """
        if pd.isna(heart_rate) or pd.isna(systolic):
            return np.nan
        
        return heart_rate * systolic
    
    def extract_base_features(self) -> pd.DataFrame:
        """
        Extract base features (direct from dataset)
        
        Returns:
            DataFrame with base features
        """
        base_features = pd.DataFrame()
        
        # Core vital signs
        feature_map = {
            'heart_rate': 'HR',
            'oxygen_saturation': 'SpO2',
            'temperature': 'Temp',
            'blood_pressure_sys': 'SysBP',
            'blood_pressure_dia': 'DiaBP',
            'glucose_level': 'Glucose',
            'cholesterol': 'Cholesterol',
            'age': 'Age'
        }
        
        for col, alias in feature_map.items():
            if col in self.df.columns:
                base_features[alias] = self.df[col]
        
        self.features['base'] = base_features
        return base_features
    
    def derive_features(self) -> pd.DataFrame:
        """
        Derive complex features from base vital signs
        
        Returns:
            DataFrame with derived features
        """
        derived = pd.DataFrame(index=self.df.index)
        
        # MAP (Mean Arterial Pressure)
        if 'blood_pressure_sys' in self.df.columns and 'blood_pressure_dia' in self.df.columns:
            derived['MAP'] = self.df.apply(
                lambda row: self.calculate_map(row['blood_pressure_sys'], 
                                              row['blood_pressure_dia']),
                axis=1
            )
        
        # PP (Pulse Pressure)
        if 'blood_pressure_sys' in self.df.columns and 'blood_pressure_dia' in self.df.columns:
            derived['PP'] = self.df.apply(
                lambda row: self.calculate_pulse_pressure(row['blood_pressure_sys'],
                                                         row['blood_pressure_dia']),
                axis=1
            )
        
        # RPP (Rate Pressure Product)
        if 'heart_rate' in self.df.columns and 'blood_pressure_sys' in self.df.columns:
            derived['RPP'] = self.df.apply(
                lambda row: self.calculate_rate_pressure_product(row['heart_rate'],
                                                                row['blood_pressure_sys']),
                axis=1
            )
        
        # HRV (Heart Rate Variability) - rolling standard deviation
        if 'heart_rate' in self.df.columns:
            derived['HRV'] = self.df['heart_rate'].rolling(
                window=self.time_window, 
                center=True
            ).std()
        
        # Normalized SpO2 (distance from normal)
        if 'oxygen_saturation' in self.df.columns:
            normal_spo2 = 97.5  # Target SpO2
            derived['SpO2_deviation'] = normal_spo2 - self.df['oxygen_saturation']
        
        # Temperature deviation from normal
        if 'temperature' in self.df.columns:
            normal_temp = 37.0
            derived['Temp_deviation'] = np.abs(self.df['temperature'] - normal_temp)
        
        self.features['derived'] = derived
        return derived
    
    def get_anomaly_features(self) -> pd.DataFrame:
        """
        Get all features for anomaly detection
        
        Returns:
            Combined DataFrame with selected features
        """
        base = self.extract_base_features()
        derived = self.derive_features()
        
        # Combine features
        features_df = pd.concat([base, derived], axis=1)
        
        return features_df
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary statistics for all features
        
        Returns:
            Dictionary with feature statistics
        """
        features_df = self.get_anomaly_features()
        
        summary = {}
        for col in features_df.columns:
            data = features_df[col].dropna()
            summary[col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                'missing': int(data.isnull().sum())
            }
        
        return summary


class FeatureSelector:
    """Select most important features for anomaly detection"""
    
    # Define core physiological features for anomaly detection
    CORE_PHYSIOLOGICAL_FEATURES = {
        'HR': 'Heart Rate',
        'SpO2': 'Oxygen Saturation',
        'Temp': 'Body Temperature',
        'SysBP': 'Systolic Blood Pressure',
        'DiaBP': 'Diastolic Blood Pressure',
        'HRV': 'Heart Rate Variability',
        'MAP': 'Mean Arterial Pressure'
    }
    
    # Optional features for enhanced detection
    OPTIONAL_FEATURES = {
        'PP': 'Pulse Pressure',
        'RPP': 'Rate Pressure Product',
        'SpO2_deviation': 'SpO2 Deviation',
        'Temp_deviation': 'Temperature Deviation',
        'Age': 'Patient Age',
        'Glucose': 'Blood Glucose',
        'Cholesterol': 'Cholesterol'
    }
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize feature selector
        
        Args:
            features_df: DataFrame with all engineered features
        """
        self.features_df = features_df
        self.selected_features = []
        self.feature_importance = {}
    
    def select_core_features(self) -> pd.DataFrame:
        """
        Select core physiological features for anomaly detection
        
        Returns:
            DataFrame with selected core features
        """
        self.selected_features = []
        
        for feature_code, feature_name in self.CORE_PHYSIOLOGICAL_FEATURES.items():
            if feature_code in self.features_df.columns:
                self.selected_features.append(feature_code)
        
        return self.features_df[self.selected_features]
    
    def select_features_by_variance(self, threshold: float = 0.01) -> pd.DataFrame:
        """
        Select features with variance above threshold
        
        Args:
            threshold: Minimum variance threshold
            
        Returns:
            DataFrame with selected features
        """
        self.selected_features = []
        
        for col in self.features_df.columns:
            variance = self.features_df[col].var()
            if variance > threshold:
                self.selected_features.append(col)
        
        return self.features_df[self.selected_features]
    
    def select_features_by_correlation(self, max_correlation: float = 0.8) -> pd.DataFrame:
        """
        Select features with low inter-correlation
        Removes highly correlated redundant features
        
        Args:
            max_correlation: Maximum allowed correlation between features
            
        Returns:
            DataFrame with selected non-redundant features
        """
        selected = []
        numeric_df = self.features_df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        available_cols = list(numeric_df.columns)
        
        while available_cols:
            col = available_cols.pop(0)
            selected.append(col)
            
            # Remove highly correlated columns
            remaining = []
            for other_col in available_cols:
                if abs(correlation_matrix.loc[col, other_col]) < max_correlation:
                    remaining.append(other_col)
            available_cols = remaining
        
        self.selected_features = selected
        return self.features_df[selected]
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features"""
        return self.selected_features
    
    def get_feature_stats(self) -> Dict:
        """
        Get statistics for selected features
        
        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        for feature in self.selected_features:
            data = self.features_df[feature].dropna()
            stats[feature] = {
                'count': len(data),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'missing': int(self.features_df[feature].isnull().sum()),
                'missing_pct': 100 * int(self.features_df[feature].isnull().sum()) / len(self.features_df)
            }
        
        return stats


class FeatureNormalizer:
    """
    Normalize numerical features using StandardScaler for anomaly detection.
    Ensures all features contribute proportionally to the models.
    """
    
    def __init__(self):
        """Initialize feature normalizer with StandardScaler"""
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer_fitted = False
    
    def fit(self, features_df: pd.DataFrame) -> 'FeatureNormalizer':
        """
        Fit StandardScaler on training data.
        Handles missing values before fitting.
        
        Args:
            features_df: Training features DataFrame (numerical features only)
            
        Returns:
            Self for method chaining
        """
        # Select only numerical columns
        numeric_df = features_df.select_dtypes(include=[np.number]).copy()
        self.feature_names = numeric_df.columns.tolist()
        
        # Fit imputer for missing values
        self.imputer.fit(numeric_df)
        self.imputer_fitted = True
        
        # Impute missing values
        numeric_imputed = pd.DataFrame(
            self.imputer.transform(numeric_df),
            columns=self.feature_names,
            index=numeric_df.index
        )
        
        # Fit StandardScaler
        self.scaler.fit(numeric_imputed)
        self.is_fitted = True
        
        return self
    
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted StandardScaler.
        
        Args:
            features_df: Features to normalize
            
        Returns:
            Normalized DataFrame with same shape as input
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        
        # Select numerical columns
        numeric_df = features_df[self.feature_names].copy()
        
        # Impute missing values
        numeric_imputed = pd.DataFrame(
            self.imputer.transform(numeric_df),
            columns=self.feature_names,
            index=numeric_df.index
        )
        
        # Apply StandardScaler
        normalized_array = self.scaler.transform(numeric_imputed)
        
        # Convert back to DataFrame
        normalized_df = pd.DataFrame(
            normalized_array,
            columns=self.feature_names,
            index=features_df.index
        )
        
        return normalized_df
    
    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            features_df: Features to fit and transform
            
        Returns:
            Normalized DataFrame
        """
        return self.fit(features_df).transform(features_df)
    
    def get_scaling_params(self) -> Dict[str, Dict[str, float]]:
        """
        Get scaling parameters (mean and std) for each feature.
        
        Returns:
            Dictionary with scaling parameters
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first.")
        
        params = {}
        for feature, mean, std in zip(
            self.feature_names,
            self.scaler.mean_,
            self.scaler.scale_
        ):
            params[feature] = {
                'mean': float(mean),
                'std': float(std)
            }
        
        return params
    
    def save(self, filepath: str) -> None:
        """
        Save fitted scaler to disk using joblib.
        
        Args:
            filepath: Path to save the scaler
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'imputer_fitted': self.imputer_fitted
        }, filepath)
    
    def load(self, filepath: str) -> 'FeatureNormalizer':
        """
        Load fitted scaler from disk.
        
        Args:
            filepath: Path to load the scaler from
            
        Returns:
            Self with loaded scaler
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Scaler file not found at {filepath}")
        
        state = joblib.load(filepath)
        self.scaler = state['scaler']
        self.imputer = state['imputer']
        self.feature_names = state['feature_names']
        self.is_fitted = state['is_fitted']
        self.imputer_fitted = state['imputer_fitted']
        
        return self
    
    def get_feature_means(self) -> np.ndarray:
        """Get mean of each feature from training data"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first.")
        return self.scaler.mean_
    
    def get_feature_stds(self) -> np.ndarray:
        """Get standard deviation of each feature from training data"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first.")
        return self.scaler.scale_
