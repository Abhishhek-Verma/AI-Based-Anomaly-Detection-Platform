"""
Explainability Module for Anomaly Detection
Provides interpretable insights into why an anomaly was detected
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnomalyExplanation:
    """Detailed explanation of detected anomaly"""
    patient_id: str
    timestamp: str
    anomaly_score: float
    severity: str  # LOW, MEDIUM, HIGH
    ae_score: float  # Autoencoder score
    if_score: float  # Isolation Forest score
    abnormal_vitals: Dict[str, float]  # Vital name -> deviation percentage
    key_contributors: List[Tuple[str, float]]  # Top abnormal vitals (sorted)
    primary_contributor: str
    primary_percentage: float
    baseline_vitals: Dict[str, float]
    current_vitals: Dict[str, float]
    recommendation: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/reporting"""
        return {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp,
            'anomaly_score': float(self.anomaly_score),
            'severity': self.severity,
            'ae_score': float(self.ae_score),
            'if_score': float(self.if_score),
            'abnormal_vitals': {k: float(v) for k, v in self.abnormal_vitals.items()},
            'key_contributors': [(k, float(v)) for k, v in self.key_contributors],
            'primary_contributor': self.primary_contributor,
            'primary_percentage': float(self.primary_percentage),
            'baseline_vitals': {k: float(v) for k, v in self.baseline_vitals.items()},
            'current_vitals': {k: float(v) for k, v in self.current_vitals.items()},
            'recommendation': self.recommendation
        }


class AnomalyExplainer:
    """
    Generates explainable insights for detected anomalies.
    Shows which vitals are abnormal and their contribution percentages.
    """
    
    # Vital signs names
    VITAL_NAMES = ['HR', 'SpO2', 'Temperature', 'SysBP', 'DiaBP']
    
    # Normal ranges for vitals (used for context)
    NORMAL_RANGES = {
        'HR': (60, 100),
        'SpO2': (95, 100),
        'Temperature': (36.5, 37.5),
        'SysBP': (90, 120),
        'DiaBP': (60, 80)
    }
    
    def __init__(self, threshold_deviation: float = 0.5):
        """
        Initialize explainer.
        
        Args:
            threshold_deviation: Min deviation percentage to flag as abnormal
        """
        self.threshold_deviation = threshold_deviation
    
    def explain_anomaly(self,
                       patient_id: str,
                       timestamp: str,
                       anomaly_score: float,
                       ae_score: float,
                       if_score: float,
                       severity: str,
                       current_vitals: np.ndarray,
                       baseline_vitals: np.ndarray,
                       baseline_stds: np.ndarray) -> AnomalyExplanation:
        """
        Generate detailed explanation for detected anomaly.
        
        Args:
            patient_id: Patient identifier
            timestamp: Timestamp of anomaly detection
            anomaly_score: Combined anomaly score (0-1)
            ae_score: Autoencoder score contribution
            if_score: Isolation Forest score contribution
            severity: Severity classification (LOW/MEDIUM/HIGH)
            current_vitals: Current vital readings (5,)
            baseline_vitals: Patient baseline vitals (5,)
            baseline_stds: Patient baseline standard deviations (5,)
        
        Returns:
            AnomalyExplanation object
        """
        
        # Calculate deviations from baseline
        deviations = current_vitals - baseline_vitals
        deviation_percentages = (deviations / np.maximum(baseline_stds, 1.0)) * 100
        
        # Identify abnormal vitals (top 3)
        abnormal_vitals = {}
        for i, vital_name in enumerate(self.VITAL_NAMES):
            dev_pct = abs(deviation_percentages[i])
            if dev_pct > self.threshold_deviation:
                abnormal_vitals[vital_name] = dev_pct
        
        # Sort by absolute deviation
        sorted_abnormal = sorted(
            abnormal_vitals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3
        
        # Primary contributor (highest deviation)
        if sorted_abnormal:
            primary_contributor, primary_percentage = sorted_abnormal[0]
        else:
            # If no clear abnormal vital, use highest absolute deviation
            max_idx = np.argmax(np.abs(deviation_percentages))
            primary_contributor = self.VITAL_NAMES[max_idx]
            primary_percentage = abs(deviation_percentages[max_idx])
        
        # Generate abnormal vitals dict
        abnormal_dict = {k: v for k, v in sorted_abnormal}
        
        # Create baseline and current vitals dicts
        baseline_dict = {
            name: float(val)
            for name, val in zip(self.VITAL_NAMES, baseline_vitals)
        }
        
        current_dict = {
            name: float(val)
            for name, val in zip(self.VITAL_NAMES, current_vitals)
        }
        
        # Generate recommendation based on severity and primary contributor
        recommendation = self._generate_recommendation(
            severity, primary_contributor, primary_percentage
        )
        
        return AnomalyExplanation(
            patient_id=patient_id,
            timestamp=timestamp,
            anomaly_score=float(anomaly_score),
            severity=severity,
            ae_score=float(ae_score),
            if_score=float(if_score),
            abnormal_vitals=abnormal_dict,
            key_contributors=sorted_abnormal,
            primary_contributor=primary_contributor,
            primary_percentage=float(primary_percentage),
            baseline_vitals=baseline_dict,
            current_vitals=current_dict,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, severity: str, vital: str, 
                                deviation: float) -> str:
        """
        Generate clinical recommendation based on severity and vital.
        
        Args:
            severity: Anomaly severity level
            vital: Most abnormal vital sign
            deviation: Deviation percentage
        
        Returns:
            Recommendation string
        """
        recommendations = {
            'HR': {
                'HIGH': "⚠️ CRITICAL: Abnormal heart rate detected. Immediate physician review required. Check for arrhythmias or hemodynamic instability.",
                'MEDIUM': "⚠️ WARNING: Heart rate trending abnormal. Increase monitoring frequency and review cardiac history.",
                'LOW': "ℹ️ INFO: Minor heart rate deviation. Continue routine monitoring."
            },
            'SpO2': {
                'HIGH': "🚨 CRITICAL: Oxygen saturation critically low. Check oxygen delivery system immediately. Consider supplemental oxygen.",
                'MEDIUM': "⚠️ WARNING: Oxygen saturation below target. Verify SpO2 probe placement and consider oxygen supplementation.",
                'LOW': "ℹ️ INFO: SpO2 slightly low. Monitor closely and ensure adequate ventilation."
            },
            'Temperature': {
                'HIGH': "🔥 CRITICAL: Severe fever detected. Rule out infection. Consider antipyretic therapy and cultures.",
                'MEDIUM': "⚠️ WARNING: Elevated temperature. Monitor for signs of infection. Consider investigation.",
                'LOW': "❄️ CRITICAL: Hypothermia detected. Initiate external rewarming. Check ambient environment."
            },
            'SysBP': {
                'HIGH': "⚠️ WARNING: Elevated blood pressure. Monitor for end-organ damage symptoms. Consider hypertensive crisis protocol.",
                'MEDIUM': "⚠️ WARNING: Systolic BP elevated. Continue monitoring and review antihypertensive medications.",
                'LOW': "💔 CRITICAL: Hypotension detected. Assess perfusion and fluid status. Consider vasopressor support."
            },
            'DiaBP': {
                'HIGH': "⚠️ WARNING: Diastolic pressure elevated. Monitor closely and review medications.",
                'MEDIUM': "⚠️ WARNING: Diastolic pressure trending high. Continue routine monitoring.",
                'LOW': "💔 CRITICAL: Low diastolic pressure. Check for shock states or cardiovascular decompensation."
            }
        }
        
        return recommendations.get(vital, {}).get(
            severity,
            f"Monitor {vital} closely."
        )
    
    def format_for_display(self, explanation: AnomalyExplanation) -> str:
        """
        Format explanation for console display.
        
        Args:
            explanation: AnomalyExplanation object
        
        Returns:
            Formatted string for console output
        """
        
        output = []
        output.append("=" * 70)
        output.append("ANOMALY DETECTED - EXPLAINABILITY REPORT")
        output.append("=" * 70)
        output.append(f"Patient ID: {explanation.patient_id}")
        output.append(f"Timestamp: {explanation.timestamp}")
        output.append(f"Anomaly Score: {explanation.anomaly_score:.4f}")
        output.append(f"Severity: {explanation.severity}")
        output.append("")
        
        output.append("MODEL SCORES:")
        output.append(f"  - Autoencoder: {explanation.ae_score:.4f}")
        output.append(f"  - Isolation Forest: {explanation.if_score:.4f}")
        output.append(f"  - Combined: {explanation.anomaly_score:.4f}")
        output.append("")
        
        output.append("KEY ABNORMAL VITALS:")
        if explanation.key_contributors:
            for vital, deviation in explanation.key_contributors:
                current = explanation.current_vitals[vital]
                baseline = explanation.baseline_vitals[vital]
                output.append(f"  {vital:12s}: {current:7.2f} (baseline: {baseline:6.2f}, deviation: +{deviation:5.1f}%)")
        else:
            output.append("  No specific abnormal vitals identified.")
        output.append("")
        
        output.append("PRIMARY CONTRIBUTOR:")
        output.append(f"  {explanation.primary_contributor}: {explanation.primary_percentage:.1f}% above baseline")
        output.append("")
        
        output.append("RECOMMENDATION:")
        output.append(f"  {explanation.recommendation}")
        output.append("=" * 70)
        
        return "\n".join(output)
    
    def format_for_email(self, explanation: AnomalyExplanation) -> str:
        """
        Format explanation for email alert.
        
        Args:
            explanation: AnomalyExplanation object
        
        Returns:
            Formatted HTML string for email
        """
        
        severity_color = {
            'LOW': '#FFA500',      # Orange
            'MEDIUM': '#FF6347',   # Tomato
            'HIGH': '#DC143C'      # Crimson
        }.get(explanation.severity, '#808080')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .header {{ background-color: {severity_color}; color: white; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 15px 0; }}
                .metric {{ background-color: #f5f5f5; padding: 10px; margin: 5px 0; border-left: 4px solid {severity_color}; }}
                .vital {{ display: inline-block; margin: 5px 10px; }}
                .abnormal {{ color: #DC143C; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 Patient Anomaly Alert</h2>
                <p style="margin: 0;">Severity: <strong>{explanation.severity}</strong></p>
            </div>
            
            <div class="section">
                <h3>Patient Information</h3>
                <div class="metric">
                    <strong>Patient ID:</strong> {explanation.patient_id}<br>
                    <strong>Timestamp:</strong> {explanation.timestamp}<br>
                    <strong>Anomaly Score:</strong> <span class="abnormal">{explanation.anomaly_score:.4f}</span>
                </div>
            </div>
            
            <div class="section">
                <h3>Key Abnormal Vitals</h3>
                <div class="metric">
                    <strong>Primary Contributor:</strong> <span class="abnormal">{explanation.primary_contributor}</span>
                    ({explanation.primary_percentage:.1f}% deviation)<br><br>
                    <strong>Current Value:</strong> {explanation.current_vitals[explanation.primary_contributor]:.2f}<br>
                    <strong>Baseline Value:</strong> {explanation.baseline_vitals[explanation.primary_contributor]:.2f}
                </div>
        """
        
        if explanation.key_contributors:
            html += "<h3>All Abnormal Vitals</h3>"
            for vital, deviation in explanation.key_contributors:
                html += f"""
                <div class="metric">
                    <strong>{vital}:</strong> {explanation.current_vitals[vital]:.2f} 
                    (baseline: {explanation.baseline_vitals[vital]:.2f}, 
                    deviation: <span class="abnormal">+{deviation:.1f}%</span>)
                </div>
                """
        
        html += f"""
            </div>
            
            <div class="section">
                <h3>Clinical Recommendation</h3>
                <div class="metric">
                    {explanation.recommendation}
                </div>
            </div>
            
            <div class="section">
                <p style="font-size: 12px; color: #999;">
                    This is an automated alert from the AI-Based Anomaly Detection Platform.
                    Please verify with the patient before taking clinical action.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
