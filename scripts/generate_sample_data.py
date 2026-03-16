"""
Sample healthcare data generator for testing
"""

import pandas as pd
import numpy as np
import os


def generate_sample_dataset(n_samples: int = 1000, output_path: str = 'data/healthcare_data.csv') -> pd.DataFrame:
    """
    Generate sample healthcare dataset
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the dataset
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(42)
    
    data = {
        'patient_id': [f'P{i:05d}' for i in range(n_samples)],
        'age': np.random.randint(18, 85, n_samples),
        'heart_rate': np.random.normal(75, 15, n_samples),
        'blood_pressure_sys': np.random.normal(120, 15, n_samples),
        'blood_pressure_dia': np.random.normal(80, 10, n_samples),
        'temperature': np.random.normal(98.6, 1, n_samples),
        'oxygen_saturation': np.random.normal(97, 2, n_samples),
        'glucose_level': np.random.normal(100, 20, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    df.to_csv(output_path, index=False)
    print(f"Sample dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    return df


if __name__ == '__main__':
    generate_sample_dataset()
