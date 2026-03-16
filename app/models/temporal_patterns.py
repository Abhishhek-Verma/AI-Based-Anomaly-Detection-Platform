"""
Temporal Pattern Processing for Anomaly Detection
Sliding window approach to capture sustained abnormalities
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TemporalWindow:
    """Represents a sliding window of temporal data"""
    data: np.ndarray
    timestamp: pd.Timestamp
    window_id: int
    patient_id: int


class TemporalWindowBuilder:
    """
    Create sliding windows from time series vital signs data.
    Captures temporal patterns to identify sustained abnormalities.
    """
    
    def __init__(self, window_size: int = 10, stride: int = 1):
        """
        Initialize temporal window builder.
        
        Args:
            window_size: Number of timesteps in each window
            stride: Number of timesteps to slide between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.windows = []
    
    def build_windows(self, df: pd.DataFrame, 
                     feature_cols: List[str],
                     patient_col: str = 'patient_id',
                     time_col: str = 'timestamp') -> List[TemporalWindow]:
        """
        Build sliding windows from time series data.
        
        Args:
            df: DataFrame with time series data
            feature_cols: List of feature column names
            patient_col: Patient identifier column
            time_col: Timestamp column
            
        Returns:
            List of TemporalWindow objects
        """
        self.windows = []
        window_id = 0
        
        # Group by patient to maintain temporal continuity
        for patient_id, patient_df in df.groupby(patient_col):
            # Sort by timestamp
            patient_df = patient_df.sort_values(time_col).reset_index(drop=True)
            
            # Create sliding windows
            for start_idx in range(0, len(patient_df) - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                
                window_data = patient_df.iloc[start_idx:end_idx][feature_cols].values
                window_time = patient_df.iloc[end_idx - 1][time_col]
                
                window = TemporalWindow(
                    data=window_data,
                    timestamp=window_time,
                    window_id=window_id,
                    patient_id=int(patient_id)
                )
                
                self.windows.append(window)
                window_id += 1
        
        return self.windows
    
    def get_windows_array(self) -> np.ndarray:
        """
        Get all windows as 3D array (num_windows, window_size, num_features)
        
        Returns:
            3D numpy array of windows
        """
        if not self.windows:
            return np.array([])
        
        return np.array([w.data for w in self.windows])
    
    def get_window_metadata(self) -> pd.DataFrame:
        """
        Get metadata for all windows.
        
        Returns:
            DataFrame with window metadata
        """
        metadata = []
        for window in self.windows:
            metadata.append({
                'window_id': window.window_id,
                'patient_id': window.patient_id,
                'timestamp': window.timestamp,
                'data_points': len(window.data)
            })
        
        return pd.DataFrame(metadata)
    
    def __len__(self) -> int:
        """Get number of windows"""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> TemporalWindow:
        """Get window by index"""
        return self.windows[idx]


class TemporalFeatureExtractor:
    """
    Extract temporal statistics from sliding windows.
    Captures patterns of sustained abnormalities.
    """
    
    def __init__(self):
        """Initialize temporal feature extractor"""
        pass
    
    @staticmethod
    def extract_window_stats(window: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from a window.
        
        Args:
            window: Window data (window_size, num_features)
            
        Returns:
            Dictionary with temporal statistics
        """
        stats = {}
        
        # Calculate per-feature statistics
        for feat_idx in range(window.shape[1]):
            feature_values = window[:, feat_idx]
            
            # Basic statistics
            stats[f'feat_{feat_idx}_mean'] = float(np.mean(feature_values))
            stats[f'feat_{feat_idx}_std'] = float(np.std(feature_values))
            stats[f'feat_{feat_idx}_min'] = float(np.min(feature_values))
            stats[f'feat_{feat_idx}_max'] = float(np.max(feature_values))
            stats[f'feat_{feat_idx}_range'] = float(np.max(feature_values) - np.min(feature_values))
            
            # Trend (slope)
            x = np.arange(len(feature_values))
            z = np.polyfit(x, feature_values, 1)
            stats[f'feat_{feat_idx}_trend'] = float(z[0])
            
            # Variability (coefficient of variation)
            mean_val = np.mean(feature_values)
            if mean_val != 0:
                stats[f'feat_{feat_idx}_cv'] = float(np.std(feature_values) / mean_val)
            else:
                stats[f'feat_{feat_idx}_cv'] = 0.0
        
        return stats
    
    def extract_all_windows(self, windows: List[TemporalWindow]) -> pd.DataFrame:
        """
        Extract features from all windows.
        
        Args:
            windows: List of TemporalWindow objects
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for window in windows:
            window_features = self.extract_window_stats(window.data)
            window_features['window_id'] = window.window_id
            window_features['patient_id'] = window.patient_id
            features_list.append(window_features)
        
        return pd.DataFrame(features_list)
