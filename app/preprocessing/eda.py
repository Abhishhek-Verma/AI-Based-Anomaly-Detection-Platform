"""
Exploratory Data Analysis (EDA) for Healthcare Vital Signs
Analyzes distribution, patterns, and identifies anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class VitalSignsEDA:
    """Exploratory Data Analysis for healthcare vital signs"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize EDA analyzer
        
        Args:
            dataframe: Healthcare dataset
        """
        self.df = dataframe
        self.stats = {}
        self.anomalies = {}
        
        # Define normal physiological ranges (clinical guidelines)
        self.normal_ranges = {
            'heart_rate': (60, 100),           # bpm
            'blood_pressure_sys': (90, 140),   # mmHg
            'blood_pressure_dia': (60, 90),    # mmHg
            'temperature': (36.5, 37.5),       # Celsius
            'oxygen_saturation': (95, 100),    # %
            'glucose_level': (70, 100),        # mg/dL (fasting)
            'cholesterol': (0, 200),           # mg/dL (desirable)
            'age': (0, 120)                    # years
        }
    
    def get_basic_statistics(self) -> Dict:
        """
        Get basic statistical summary
        
        Returns:
            Dictionary with descriptive statistics
        """
        logger.info("Computing basic statistics...")
        
        stats = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'numeric_stats': self.df.describe().to_dict()
        }
        
        self.stats = stats
        return stats
    
    def analyze_vital_sign(self, column: str) -> Dict:
        """
        Analyze distribution of a vital sign
        
        Args:
            column: Vital sign column name
            
        Returns:
            Dictionary with analysis results
        """
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return {}
        
        data = self.df[column].dropna()
        
        analysis = {
            'column': column,
            'count': len(data),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis())
        }
        
        logger.info(f"Analysis for {column}: Mean={analysis['mean']:.2f}, Std={analysis['std']:.2f}")
        return analysis
    
    def identify_outliers(self, column: str, method: str = 'iqr') -> List[int]:
        """
        Identify outliers in a vital sign
        
        Args:
            column: Column name
            method: 'iqr' or 'zscore'
            
        Returns:
            List of outlier indices
        """
        data = self.df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_mask = z_scores > 3
        
        else:
            return []
        
        outlier_indices = data[outlier_mask].index.tolist()
        logger.info(f"Found {len(outlier_indices)} outliers in {column} using {method}")
        return outlier_indices
    
    def identify_abnormal_ranges(self) -> Dict:
        """
        Identify readings outside normal physiological ranges
        
        Returns:
            Dictionary with abnormal readings
        """
        logger.info("Identifying abnormal readings...")
        abnormal = {}
        
        for column, (normal_min, normal_max) in self.normal_ranges.items():
            if column in self.df.columns:
                below_normal = (self.df[column] < normal_min).sum()
                above_normal = (self.df[column] > normal_max).sum()
                abnormal_count = below_normal + above_normal
                abnormal_pct = (abnormal_count / len(self.df)) * 100
                
                abnormal[column] = {
                    'below_normal': int(below_normal),
                    'above_normal': int(above_normal),
                    'abnormal_count': int(abnormal_count),
                    'abnormal_percentage': float(abnormal_pct),
                    'normal_range': (normal_min, normal_max)
                }
                
                if abnormal_count > 0:
                    logger.warning(
                        f"{column}: {abnormal_count} abnormal values ({abnormal_pct:.2f}%) "
                        f"[Normal: {normal_min}-{normal_max}]"
                    )
        
        self.anomalies = abnormal
        return abnormal
    
    def analyze_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between vital signs
        
        Returns:
            Correlation matrix
        """
        logger.info("Computing correlations...")
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        logger.info("Correlation analysis complete")
        return correlation_matrix
    
    def get_patient_stats(self, patient_id: str) -> Dict:
        """
        Get statistics for a specific patient
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with patient statistics
        """
        patient_data = self.df[self.df['patient_id'] == patient_id]
        
        if patient_data.empty:
            logger.warning(f"Patient {patient_id} not found")
            return {}
        
        stats = {}
        for col in patient_data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': float(patient_data[col].mean()),
                'std': float(patient_data[col].std()),
                'min': float(patient_data[col].min()),
                'max': float(patient_data[col].max()),
                'latest': float(patient_data[col].iloc[-1]) if len(patient_data) > 0 else None
            }
        
        return stats
    
    def detect_trends(self, column: str, window: int = 5) -> Dict:
        """
        Detect trends in vital signs
        
        Args:
            column: Column name
            window: Rolling window size
            
        Returns:
            Dictionary with trend information
        """
        if column not in self.df.columns:
            return {}
        
        data = self.df[column].dropna()
        rolling_mean = data.rolling(window=window).mean()
        
        # Calculate trend direction
        trend_direction = np.polyfit(range(len(data)), data, 1)[0]
        
        trend_info = {
            'column': column,
            'overall_trend': 'increasing' if trend_direction > 0 else 'decreasing',
            'trend_strength': float(abs(trend_direction)),
            'rolling_mean': rolling_mean.to_list(),
            'current_value': float(data.iloc[-1]),
            'mean_value': float(data.mean())
        }
        
        return trend_info
    
    def generate_report(self) -> str:
        """
        Generate comprehensive EDA report
        
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("HEALTHCARE VITAL SIGNS - EXPLORATORY DATA ANALYSIS (EDA)")
        report.append("=" * 80)
        
        # Dataset overview
        report.append("\n[1] DATASET OVERVIEW")
        report.append(f"   Shape: {self.stats['shape']}")
        report.append(f"   Duplicates: {self.stats['duplicates']}")
        missing = self.stats['missing_values']
        if any(missing.values()):
            report.append("   Missing Values:")
            for col, count in missing.items():
                if count > 0:
                    report.append(f"      {col}: {count}")
        
        # Vital signs analysis
        report.append("\n[2] VITAL SIGNS ANALYSIS")
        vital_signs = ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 
                       'temperature', 'oxygen_saturation', 'glucose_level', 'cholesterol']
        
        for vital_sign in vital_signs:
            if vital_sign in self.df.columns:
                stats = self.analyze_vital_sign(vital_sign)
                report.append(f"\n   {vital_sign.upper()}")
                report.append(f"      Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                report.append(f"      Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                report.append(f"      Median: {stats['median']:.2f}")
                report.append(f"      Skewness: {stats['skewness']:.2f}")
        
        # Abnormal ranges
        report.append("\n[3] ABNORMAL PHYSIOLOGICAL VALUES")
        abnormal = self.identify_abnormal_ranges()
        
        for column, info in abnormal.items():
            if info['abnormal_count'] > 0:
                report.append(f"\n   {column.upper()}")
                report.append(f"      Normal Range: {info['normal_range']}")
                report.append(f"      Below Normal: {info['below_normal']} readings")
                report.append(f"      Above Normal: {info['above_normal']} readings")
                report.append(f"      Abnormal %: {info['abnormal_percentage']:.2f}%")
        
        # Correlations
        report.append("\n[4] CORRELATIONS BETWEEN VITAL SIGNS")
        correlations = self.analyze_correlations()
        high_corr = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr_val = correlations.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr.append((correlations.columns[i], correlations.columns[j], corr_val))
        
        if high_corr:
            for col1, col2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                report.append(f"   {col1} <-> {col2}: {corr:.3f}")
        else:
            report.append("   No strong correlations found (|r| > 0.5)")
        
        report.append("\n" + "=" * 80)
        report.append("END OF EDA REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)


class VitalSignsVisualizer:
    """Create visualizations for vital signs analysis"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize visualizer"""
        self.df = dataframe
        sns.set_style("darkgrid")
    
    def plot_distributions(self, output_path: str = 'distribution_plots.png'):
        """Plot distributions of vital signs"""
        vital_signs = ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia',
                       'temperature', 'oxygen_saturation', 'glucose_level', 'cholesterol']
        
        available_signs = [vs for vs in vital_signs if vs in self.df.columns]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, vital_sign in enumerate(available_signs):
            axes[idx].hist(self.df[vital_sign].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{vital_sign.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(available_signs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {output_path}")
        plt.close()
    
    def plot_boxplots(self, output_path: str = 'boxplot.png'):
        """Plot boxplots for outlier detection"""
        vital_signs = ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia',
                       'temperature', 'oxygen_saturation', 'glucose_level']
        
        available_signs = [vs for vs in vital_signs if vs in self.df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, vital_sign in enumerate(available_signs):
            axes[idx].boxplot(self.df[vital_sign].dropna())
            axes[idx].set_title(f'{vital_sign.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Boxplot saved to {output_path}")
        plt.close()
    
    def plot_correlations(self, output_path: str = 'correlation_heatmap.png'):
        """Plot correlation heatmap"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix - Vital Signs', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {output_path}")
        plt.close()
    
    def plot_abnormal_ranges(self, normal_ranges: Dict, output_path: str = 'abnormal_ranges.png'):
        """Plot readings outside normal ranges"""
        vital_signs = [k for k in normal_ranges.keys() if k in self.df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, vital_sign in enumerate(vital_signs[:6]):
            normal_min, normal_max = normal_ranges[vital_sign]
            
            # Plot distribution with normal range highlighted
            axes[idx].hist(self.df[vital_sign].dropna(), bins=30, alpha=0.7, color='skyblue')
            axes[idx].axvline(normal_min, color='green', linestyle='--', linewidth=2, label='Normal Range')
            axes[idx].axvline(normal_max, color='green', linestyle='--', linewidth=2)
            axes[idx].axvspan(normal_min, normal_max, alpha=0.2, color='green')
            
            axes[idx].set_title(f'{vital_sign.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Abnormal ranges plot saved to {output_path}")
        plt.close()
