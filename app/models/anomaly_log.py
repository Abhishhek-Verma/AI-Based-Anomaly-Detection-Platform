"""Database models for anomaly detection"""
from datetime import datetime
from enum import Enum
from sqlalchemy.dialects.postgresql import JSON, ARRAY
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class SeverityEnum(str, Enum):
    """Severity levels for anomalies"""
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'


class AnomalyLog(db.Model):
    """
    Stores all detected anomalies with details for auditing and analysis
    """
    __tablename__ = 'anomaly_logs'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Identification
    patient_id = db.Column(db.String(50), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Anomaly scores
    anomaly_score = db.Column(db.Float, nullable=False)
    autoencoder_score = db.Column(db.Float, nullable=False)
    isolation_forest_score = db.Column(db.Float, nullable=False)
    
    # Severity
    severity = db.Column(
        db.Enum(SeverityEnum),
        nullable=False,
        default=SeverityEnum.LOW,
        index=True
    )
    
    # Current vital signs (JSON for flexibility)
    vital_signs = db.Column(JSON, nullable=False)  # {hr, spo2, temp, sysbp, diabp}
    
    # Baseline vital signs for comparison
    baseline_vitals = db.Column(JSON, nullable=False)
    
    # Deviations from baseline (percentage)
    vital_deviations = db.Column(JSON, nullable=False)  # {hr_dev, spo2_dev, ...}
    
    # Key abnormal vitals (top 3)
    abnormal_vitals = db.Column(ARRAY(db.String), nullable=True)  # ['HR', 'SysBP', ...]
    
    # Primary contributor and its percentage
    primary_contributor = db.Column(db.String(50), nullable=True)
    primary_contributor_percentage = db.Column(db.Float, nullable=True)
    
    # Full explanation text
    explanation = db.Column(db.Text, nullable=True)
    recommendation = db.Column(db.Text, nullable=True)
    
    # Alert information
    alert_sent = db.Column(db.Boolean, default=False, index=True)
    alert_timestamp = db.Column(db.DateTime, nullable=True)
    
    # Metadata
    window_data = db.Column(JSON, nullable=True)  # For debugging
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        db.Index('idx_patient_timestamp', 'patient_id', 'timestamp'),
        db.Index('idx_severity_timestamp', 'severity', 'timestamp'),
    )
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'anomaly_score': self.anomaly_score,
            'autoencoder_score': self.autoencoder_score,
            'isolation_forest_score': self.isolation_forest_score,
            'severity': self.severity.value if self.severity else None,
            'vital_signs': self.vital_signs,
            'baseline_vitals': self.baseline_vitals,
            'vital_deviations': self.vital_deviations,
            'abnormal_vitals': self.abnormal_vitals,
            'primary_contributor': self.primary_contributor,
            'primary_contributor_percentage': self.primary_contributor_percentage,
            'explanation': self.explanation,
            'recommendation': self.recommendation,
            'alert_sent': self.alert_sent,
            'alert_timestamp': self.alert_timestamp.isoformat() if self.alert_timestamp else None,
        }
    
    def __repr__(self):
        return f'<AnomalyLog {self.patient_id} {self.timestamp} {self.severity}>'


class PatientBaseline(db.Model):
    """
    Stores personalized baseline vitals for each patient
    Updated as new data is processed
    """
    __tablename__ = 'patient_baselines'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Patient identification
    patient_id = db.Column(db.String(50), nullable=False, unique=True, index=True)
    
    # Baseline statistics (mean and std for each vital)
    hr_mean = db.Column(db.Float, nullable=True)
    hr_std = db.Column(db.Float, nullable=True)
    
    spo2_mean = db.Column(db.Float, nullable=True)
    spo2_std = db.Column(db.Float, nullable=True)
    
    temperature_mean = db.Column(db.Float, nullable=True)
    temperature_std = db.Column(db.Float, nullable=True)
    
    sysbp_mean = db.Column(db.Float, nullable=True)
    sysbp_std = db.Column(db.Float, nullable=True)
    
    diabp_mean = db.Column(db.Float, nullable=True)
    diabp_std = db.Column(db.Float, nullable=True)
    
    # Metadata
    samples_count = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'patient_id': self.patient_id,
            'hr': {'mean': self.hr_mean, 'std': self.hr_std},
            'spo2': {'mean': self.spo2_mean, 'std': self.spo2_std},
            'temperature': {'mean': self.temperature_mean, 'std': self.temperature_std},
            'sysbp': {'mean': self.sysbp_mean, 'std': self.sysbp_std},
            'diabp': {'mean': self.diabp_mean, 'std': self.diabp_std},
            'samples_count': self.samples_count,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
        }
    
    def __repr__(self):
        return f'<PatientBaseline {self.patient_id}>'


class PatientStats(db.Model):
    """
    Aggregated statistics per patient for dashboard KPIs
    """
    __tablename__ = 'patient_stats'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(50), nullable=False, unique=True, index=True)
    
    # Anomaly counts
    total_anomalies = db.Column(db.Integer, default=0)
    high_severity_count = db.Column(db.Integer, default=0)
    medium_severity_count = db.Column(db.Integer, default=0)
    low_severity_count = db.Column(db.Integer, default=0)
    
    # Anomaly rate (%)
    anomaly_rate = db.Column(db.Float, default=0.0)
    
    # Latest vitals
    last_hr = db.Column(db.Float, nullable=True)
    last_spo2 = db.Column(db.Float, nullable=True)
    last_temperature = db.Column(db.Float, nullable=True)
    last_sysbp = db.Column(db.Float, nullable=True)
    last_diabp = db.Column(db.Float, nullable=True)
    last_vital_timestamp = db.Column(db.DateTime, nullable=True)
    
    # Severity of latest anomaly
    latest_severity = db.Column(db.Enum(SeverityEnum), nullable=True)
    latest_anomaly_timestamp = db.Column(db.DateTime, nullable=True)
    
    # Metadata
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'patient_id': self.patient_id,
            'total_anomalies': self.total_anomalies,
            'high_severity_count': self.high_severity_count,
            'medium_severity_count': self.medium_severity_count,
            'low_severity_count': self.low_severity_count,
            'anomaly_rate': self.anomaly_rate,
            'last_vitals': {
                'hr': self.last_hr,
                'spo2': self.last_spo2,
                'temperature': self.last_temperature,
                'sysbp': self.last_sysbp,
                'diabp': self.last_diabp,
                'timestamp': self.last_vital_timestamp.isoformat() if self.last_vital_timestamp else None,
            },
            'latest_severity': self.latest_severity.value if self.latest_severity else None,
            'latest_anomaly_timestamp': self.latest_anomaly_timestamp.isoformat() if self.latest_anomaly_timestamp else None,
        }
    
    def __repr__(self):
        return f'<PatientStats {self.patient_id}>'
