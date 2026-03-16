"""
EDA Analysis Script - Generate comprehensive healthcare data analysis
"""

import os
import sys
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config
from utils.logger import setup_logging
from app.preprocessing.data_loader import HealthcareDataLoader
from app.preprocessing.eda import VitalSignsEDA, VitalSignsVisualizer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Execute comprehensive EDA analysis"""
    
    logger.info("=" * 80)
    logger.info("HEALTHCARE VITAL SIGNS - EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 80)
    
    config = get_config()
    data_path = os.path.join(config.DATA_PATH, config.DATASET_NAME)
    
    # Load data
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please run: python scripts/generate_sample_data.py")
        return
    
    loader = HealthcareDataLoader(data_path)
    df = loader.load_data()
    
    # Initialize EDA analyzer
    eda = VitalSignsEDA(df)
    
    # Get basic statistics
    logger.info("\n[Step 1] Computing Basic Statistics...")
    basic_stats = eda.get_basic_statistics()
    logger.info(f"Dataset shape: {basic_stats['shape']}")
    logger.info(f"Missing values: {sum(basic_stats['missing_values'].values())}")
    
    # Analyze vital signs
    logger.info("\n[Step 2] Analyzing Vital Signs Distribution...")
    vital_signs = ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia',
                   'temperature', 'oxygen_saturation', 'glucose_level', 'cholesterol']
    
    for vital_sign in vital_signs:
        if vital_sign in df.columns:
            stats = eda.analyze_vital_sign(vital_sign)
    
    # Identify abnormal readings
    logger.info("\n[Step 3] Identifying Abnormal Physiological Values...")
    abnormal = eda.identify_abnormal_ranges()
    
    total_abnormal = sum(info['abnormal_count'] for info in abnormal.values())
    logger.info(f"Total abnormal readings found: {total_abnormal}")
    
    # Analyze correlations
    logger.info("\n[Step 4] Analyzing Correlations...")
    correlations = eda.analyze_correlations()
    
    # Detect trends
    logger.info("\n[Step 5] Detecting Trends...")
    for vital_sign in vital_signs[:3]:
        if vital_sign in df.columns:
            trend = eda.detect_trends(vital_sign)
            if trend:
                logger.info(f"{vital_sign}: {trend['overall_trend']} trend (strength: {trend['trend_strength']:.4f})")
    
    # Generate report
    logger.info("\n[Step 6] Generating Report...")
    report = eda.generate_report()
    print("\n" + report)
    
    # Save report
    report_path = 'eda_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Create visualizations
    logger.info("\n[Step 7] Creating Visualizations...")
    visualizer = VitalSignsVisualizer(df)
    
    # Create output directory if needed
    os.makedirs('plots', exist_ok=True)
    
    visualizer.plot_distributions('plots/distribution_plots.png')
    visualizer.plot_boxplots('plots/boxplot.png')
    visualizer.plot_correlations('plots/correlation_heatmap.png')
    visualizer.plot_abnormal_ranges(eda.normal_ranges, 'plots/abnormal_ranges.png')
    
    logger.info("\nVisualizations saved to 'plots/' directory")
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("EDA ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Report: {report_path}")
    logger.info("Visualizations: plots/")
    logger.info("  - distribution_plots.png")
    logger.info("  - boxplot.png")
    logger.info("  - correlation_heatmap.png")
    logger.info("  - abnormal_ranges.png")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
