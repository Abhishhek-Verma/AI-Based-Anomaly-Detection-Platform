"""
Patient-Specific Tracking for Real-Time Anomaly Detection
Maintains individual sliding windows and baseline vitals per patient
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatientVitals:
    """Single vital signs reading for a patient"""
    patient_id: str
    timestamp: datetime
    hr: float          # Heart Rate
    spo2: float        # Blood Oxygen Saturation
    temperature: float # Body Temperature (Celsius)
    sys_bp: float      # Systolic Blood Pressure
    dia_bp: float      # Diastolic Blood Pressure
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array"""
        return np.array([self.hr, self.spo2, self.temperature, self.sys_bp, self.dia_bp])
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp.isoformat(),
            'HR': self.hr,
            'SpO2': self.spo2,
            'Temperature': self.temperature,
            'SysBP': self.sys_bp,
            'DiaBP': self.dia_bp
        }


@dataclass
class PatientBaseline:
    """Patient-specific baseline vitals for anomaly detection"""
    patient_id: str
    hr_mean: float = 75.0
    hr_std: float = 10.0
    spo2_mean: float = 97.0
    spo2_std: float = 2.0
    temperature_mean: float = 37.0
    temperature_std: float = 0.5
    sys_bp_mean: float = 110.0
    sys_bp_std: float = 10.0
    dia_bp_mean: float = 70.0
    dia_bp_std: float = 7.0
    
    def get_means(self) -> np.ndarray:
        """Get baseline means as array"""
        return np.array([
            self.hr_mean, self.spo2_mean, self.temperature_mean,
            self.sys_bp_mean, self.dia_bp_mean
        ])
    
    def get_stds(self) -> np.ndarray:
        """Get baseline standard deviations as array"""
        return np.array([
            self.hr_std, self.spo2_std, self.temperature_std,
            self.sys_bp_std, self.dia_bp_std
        ])
    
    def update_from_history(self, vitals_history: List[PatientVitals]) -> None:
        """Update baseline from patient's vital history"""
        if not vitals_history:
            return
        
        vitals_array = np.array([v.to_array() for v in vitals_history])
        
        self.hr_mean = float(np.mean(vitals_array[:, 0]))
        self.hr_std = float(np.std(vitals_array[:, 0]))
        self.spo2_mean = float(np.mean(vitals_array[:, 1]))
        self.spo2_std = float(np.std(vitals_array[:, 1]))
        self.temperature_mean = float(np.mean(vitals_array[:, 2]))
        self.temperature_std = float(np.std(vitals_array[:, 2]))
        self.sys_bp_mean = float(np.mean(vitals_array[:, 3]))
        self.sys_bp_std = float(np.std(vitals_array[:, 3]))
        self.dia_bp_mean = float(np.mean(vitals_array[:, 4]))
        self.dia_bp_std = float(np.std(vitals_array[:, 4]))


class PatientSlidingWindow:
    """
    Maintains individual sliding window for a patient.
    Used for temporal pattern analysis.
    """
    
    def __init__(self, patient_id: str, window_size: int = 10):
        """
        Initialize patient sliding window.
        
        Args:
            patient_id: Unique patient identifier
            window_size: Number of timesteps in window
        """
        self.patient_id = patient_id
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
    
    def add_vital(self, vital: PatientVitals) -> None:
        """Add new vital reading to window"""
        self.window.append(vital.to_array())
        self.timestamps.append(vital.timestamp)
    
    def get_window_array(self) -> Optional[np.ndarray]:
        """
        Get current window as flattened array.
        
        Returns:
            Flattened window (50,) if full, None if incomplete
        """
        if len(self.window) < self.window_size:
            return None
        
        window_array = np.array(list(self.window))
        return window_array.flatten()
    
    def is_full(self) -> bool:
        """Check if window has reached target size"""
        return len(self.window) >= self.window_size
    
    def get_window_size(self) -> int:
        """Get current window size (may be less than target)"""
        return len(self.window)


class PatientTracker:
    """
    Centralized tracker for all patients.
    Maintains per-patient sliding windows and baseline vitals.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize patient tracker.
        
        Args:
            window_size: Size of temporal window per patient
        """
        self.window_size = window_size
        self.patients: Dict[str, Dict] = {}
        self.retention_period = timedelta(hours=24)
    
    def register_patient(self, patient_id: str, baseline: Optional[PatientBaseline] = None) -> None:
        """
        Register a new patient.
        
        Args:
            patient_id: Unique patient identifier
            baseline: Optional pre-defined baseline vitals
        """
        if patient_id not in self.patients:
            self.patients[patient_id] = {
                'window': PatientSlidingWindow(patient_id, self.window_size),
                'baseline': baseline or PatientBaseline(patient_id),
                'vitals_history': deque(maxlen=1000),  # Keep last 1000 readings
                'last_update': None
            }
            logger.info(f"Registered patient: {patient_id}")
    
    def add_vital(self, vital: PatientVitals) -> None:
        """
        Add vital reading for a patient.
        
        Args:
            vital: PatientVitals reading
        """
        # Auto-register if not exists
        if vital.patient_id not in self.patients:
            self.register_patient(vital.patient_id)
        
        patient_data = self.patients[vital.patient_id]
        patient_data['window'].add_vital(vital)
        patient_data['vitals_history'].append(vital)
        patient_data['last_update'] = vital.timestamp
    
    def get_patient_window(self, patient_id: str) -> Optional[np.ndarray]:
        """Get current window for patient"""
        if patient_id not in self.patients:
            return None
        
        return self.patients[patient_id]['window'].get_window_array()
    
    def get_patient_baseline(self, patient_id: str) -> Optional[PatientBaseline]:
        """Get baseline vitals for patient"""
        if patient_id not in self.patients:
            return None
        
        return self.patients[patient_id]['baseline']
    
    def get_recent_vitals(self, patient_id: str, count: int = 10) -> List[PatientVitals]:
        """Get recent vitals for patient"""
        if patient_id not in self.patients:
            return []
        
        history = self.patients[patient_id]['vitals_history']
        return list(history)[-count:]
    
    def update_baseline(self, patient_id: str) -> None:
        """Update patient baseline from recent history"""
        if patient_id not in self.patients:
            return
        
        patient_data = self.patients[patient_id]
        patient_data['baseline'].update_from_history(
            list(patient_data['vitals_history'])
        )
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """Get comprehensive patient information"""
        if patient_id not in self.patients:
            return None
        
        patient_data = self.patients[patient_id]
        return {
            'patient_id': patient_id,
            'window_size': patient_data['window'].get_window_size(),
            'baseline': patient_data['baseline'],
            'last_update': patient_data['last_update'],
            'total_readings': len(patient_data['vitals_history'])
        }
    
    def get_all_active_patients(self) -> List[str]:
        """Get list of all active patients"""
        return list(self.patients.keys())
    
    def cleanup_stale_patients(self, days: int = 7) -> int:
        """
        Remove patients with no recent activity.
        
        Args:
            days: Remove patients inactive for this many days
        
        Returns:
            Number of patients removed
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        stale_patients = [
            pid for pid, data in self.patients.items()
            if data['last_update'] and data['last_update'] < cutoff_time
        ]
        
        for pid in stale_patients:
            del self.patients[pid]
            logger.info(f"Removed stale patient: {pid}")
        
        return len(stale_patients)
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        return {
            'total_patients': len(self.patients),
            'active_patients': sum(
                1 for data in self.patients.values()
                if data['window'].is_full()
            ),
            'total_readings': sum(
                len(data['vitals_history'])
                for data in self.patients.values()
            )
        }
