"""
Kafka Consumer - Real-Time ML Inference and Alerting
Consumes patient vitals, runs anomaly detection, and generates alerts
"""

import json
import logging
from typing import Optional, Dict, List
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import numpy as np
import pandas as pd
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Import ML components
from app.preprocessing.feature_engineering import FeatureNormalizer
from app.models.autoencoder import AutoencoderAnomalyDetector
from app.models.isolation_forest import IsolationForestAnomalyDetector
from app.models.combined_detector import SeverityClassifier
from app.streaming.patient_tracking import PatientTracker, PatientVitals, PatientBaseline
from app.streaming.explainability import AnomalyExplainer, AnomalyExplanation
from app.streaming.alerting import AlertManager, EmailAlertSender, AlertLogger


class RealtimeAnomalyDetector:
    """
    Real-time anomaly detection for streaming vital signs.
    Integrates all components: tracking, inference, explainability, alerting.
    """
    
    def __init__(self,
                 models_dir: str = 'models',
                 alert_severity_threshold: str = 'MEDIUM',
                 alert_cooldown_minutes: int = 30):
        """
        Initialize real-time detector.
        
        Args:
            models_dir: Directory with pre-trained models
            alert_severity_threshold: Minimum severity to trigger alerts
            alert_cooldown_minutes: Minutes between alerts per patient
        """
        self.models_dir = models_dir
        
        # Load models
        self.normalizer = FeatureNormalizer()
        self.autoencoder = AutoencoderAnomalyDetector(input_dim=50, encoding_dim=16)
        self.isolation_forest = IsolationForestAnomalyDetector()
        self.severity_classifier = SeverityClassifier()
        
        self._load_models()
        
        # Initialize components
        self.patient_tracker = PatientTracker(window_size=10)
        self.explainer = AnomalyExplainer(threshold_deviation=0.5)
        self.alert_manager = AlertManager(
            cooldown_minutes=alert_cooldown_minutes,
            severity_threshold=alert_severity_threshold
        )
        self.email_sender = EmailAlertSender()
        self.alert_logger = AlertLogger()
        
        # Statistics
        self.total_messages = 0
        self.total_anomalies = 0
        self.total_alerts_sent = 0
    
    def _load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            # Load normalizer
            normalizer_path = f"{self.models_dir}/feature_normalizer.joblib"
            self.normalizer.load(normalizer_path)
            logger.info(f"Loaded feature normalizer from {normalizer_path}")
            
            # Load autoencoder
            ae_path = f"{self.models_dir}/autoencoder_model.h5"
            self.autoencoder.load(ae_path)
            logger.info(f"Loaded autoencoder from {ae_path}")
            
            # Load isolation forest
            if_path = f"{self.models_dir}/isolation_forest_model.joblib"
            self.isolation_forest.load_model(if_path)
            logger.info(f"Loaded isolation forest from {if_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.warning("Using untrained models. Run training script first.")
            return False
    
    def process_vital(self, vital_dict: Dict) -> Optional[Dict]:
        """
        Process single vital reading with full anomaly detection pipeline.
        
        Args:
            vital_dict: Dictionary with vital signs
        
        Returns:
            Result dictionary with anomaly info or None
        """
        try:
            # Parse vital
            vital = PatientVitals(
                patient_id=vital_dict['patient_id'],
                timestamp=datetime.fromisoformat(vital_dict['timestamp']),
                hr=float(vital_dict['HR']),
                spo2=float(vital_dict['SpO2']),
                temperature=float(vital_dict['Temperature']),
                sys_bp=float(vital_dict['SysBP']),
                dia_bp=float(vital_dict['DiaBP'])
            )
            
            # Add to patient tracker
            self.patient_tracker.add_vital(vital)
            self.total_messages += 1
            
            # Get window for inference
            window = self.patient_tracker.get_patient_window(vital.patient_id)
            if window is None:
                # Not enough data yet
                return {
                    'patient_id': vital.patient_id,
                    'timestamp': vital.timestamp.isoformat(),
                    'status': 'buffering',
                    'window_progress': self.patient_tracker.patients[vital.patient_id]['window'].get_window_size()
                }
            
            # Normalize window using fitted scaler
            window_df = pd.DataFrame(
                window.reshape(-1, 5),
                columns=['HR', 'SpO2', 'Temperature', 'SysBP', 'DiaBP']
            )
            window_norm = self.normalizer.transform(window_df).values.flatten()
            
            # Get anomaly scores from both models
            ae_score = float(self.autoencoder.predict_anomaly_score(window_norm.reshape(1, -1))[0])
            if_score = float(self.isolation_forest.predict_anomaly_score(window_norm.reshape(1, -1))[0])
            
            # Combine scores
            combined_score = 0.5 * ae_score + 0.5 * if_score
            
            # Classify severity
            severity = self.severity_classifier.classify(np.array([combined_score]))[0]
            
            self.total_anomalies += 1
            
            # Generate explanation
            baseline = self.patient_tracker.get_patient_baseline(vital.patient_id)
            explanation = self.explainer.explain_anomaly(
                patient_id=vital.patient_id,
                timestamp=vital.timestamp.isoformat(),
                anomaly_score=combined_score,
                ae_score=ae_score,
                if_score=if_score,
                severity=severity,
                current_vitals=vital.to_array(),
                baseline_vitals=baseline.get_means(),
                baseline_stds=baseline.get_stds()
            )
            
            # Check if alert should be sent
            should_alert = self.alert_manager.should_send_alert(
                vital.patient_id,
                severity
            )
            
            result = {
                'patient_id': vital.patient_id,
                'timestamp': vital.timestamp.isoformat(),
                'status': 'anomaly_detected' if combined_score > 0.5 else 'normal',
                'anomaly_score': combined_score,
                'ae_score': ae_score,
                'if_score': if_score,
                'severity': severity,
                'should_alert': should_alert,
                'explanation': explanation.to_dict()
            }
            
            # Send alert if needed
            if should_alert and self._should_send_email_alert(severity):
                self._send_alert(vital, explanation)
                self.alert_manager.mark_alert_sent(vital.patient_id, severity)
                self.total_alerts_sent += 1
                result['alert_sent'] = True
            else:
                result['alert_sent'] = False
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing vital: {e}", exc_info=True)
            return None
    
    def _should_send_email_alert(self, severity: str) -> bool:
        """Check if severity warrants email alert"""
        return severity in ['MEDIUM', 'HIGH']
    
    def _send_alert(self, vital: PatientVitals, explanation: AnomalyExplanation) -> None:
        """Send alert (console + email)"""
        # Console output
        console_output = self.explainer.format_for_display(explanation)
        logger.info("\n" + console_output)
        
        # Email alert (if configured)
        if self.email_sender.enabled:
            email_html = self.explainer.format_for_email(explanation)
            # For demo purposes, log instead of actually sending
            logger.info(f"[EMAIL] Would send alert to clinical team for patient {vital.patient_id}")
        
        # Log alert
        self.alert_logger.log_alert(
            patient_id=vital.patient_id,
            severity=explanation.severity,
            anomaly_score=explanation.anomaly_score,
            primary_contributor=explanation.primary_contributor,
            explanation_dict=explanation.to_dict()
        )
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        tracker_stats = self.patient_tracker.get_statistics()
        
        return {
            'total_messages': self.total_messages,
            'total_anomalies': self.total_anomalies,
            'total_alerts_sent': self.total_alerts_sent,
            'anomaly_rate': self.total_anomalies / max(1, self.total_messages),
            'active_patients': tracker_stats['active_patients'],
            'total_patients': tracker_stats['total_patients']
        }


class KafkaVitalConsumer:
    """
    Kafka consumer for patient vital signs.
    Performs real-time ML inference and alerting.
    """
    
    def __init__(self,
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'patient_vitals',
                 group_id: str = 'anomaly_detection_consumer',
                 models_dir: str = 'models'):
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            models_dir: Directory with pre-trained models
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.detector = RealtimeAnomalyDetector(models_dir=models_dir)
        self.message_count = 0
    
    def connect(self) -> bool:
        """
        Connect to Kafka broker.
        
        Returns:
            True if connected successfully
        """
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True
            )
            logger.info(f"Connected to Kafka topic '{self.topic}'")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def consume(self, max_messages: Optional[int] = None) -> None:
        """
        Consume messages from Kafka topic.
        
        Args:
            max_messages: Max messages to consume (None = infinite)
        """
        if not self.consumer:
            logger.error("Consumer not connected. Call connect() first.")
            return
        
        try:
            logger.info("=" * 70)
            logger.info("KAFKA CONSUMER - REAL-TIME ANOMALY DETECTION")
            logger.info("=" * 70)
            
            for message in self.consumer:
                try:
                    vital_dict = message.value
                    
                    # Process vital
                    result = self.detector.process_vital(vital_dict)
                    
                    if result:
                        self.message_count += 1
                        
                        # Log processing
                        if result['status'] == 'buffering':
                            logger.info(f"[BUFFERING] {result['patient_id']}: {result['window_progress']}/10 samples")
                        else:
                            status_icon = "🚨" if result['alert_sent'] else "✓"
                            logger.info(
                                f"{status_icon} [{result['severity']}] {result['patient_id']}: "
                                f"Score={result['anomaly_score']:.4f}, "
                                f"Primary={result['explanation']['primary_contributor']}"
                            )
                        
                        # Log statistics periodically
                        if self.message_count % 100 == 0:
                            stats = self.detector.get_statistics()
                            logger.info(f"\nStatistics: {stats}\n")
                    
                    # Check max messages
                    if max_messages and self.message_count >= max_messages:
                        logger.info(f"Reached max messages ({max_messages}). Stopping consumer.")
                        break
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        
        finally:
            self.close()
    
    def close(self) -> None:
        """Close Kafka consumer"""
        if self.consumer:
            self.consumer.close()
            
            # Print final statistics
            stats = self.detector.get_statistics()
            logger.info("\n" + "=" * 70)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 70)
            logger.info(f"Total messages consumed: {self.message_count}")
            logger.info(f"Total anomalies detected: {stats['total_anomalies']}")
            logger.info(f"Total alerts sent: {stats['total_alerts_sent']}")
            logger.info(f"Anomaly rate: {stats['anomaly_rate']:.2%}")
            logger.info(f"Active patients: {stats['active_patients']}")
            logger.info("=" * 70)


def run_consumer_demo(bootstrap_servers: str = 'localhost:9092',
                     max_messages: Optional[int] = None):
    """
    Run consumer demo.
    
    Args:
        bootstrap_servers: Kafka broker addresses
        max_messages: Max messages to consume
    """
    consumer = KafkaVitalConsumer(bootstrap_servers=bootstrap_servers)
    
    if not consumer.connect():
        logger.error("Failed to connect to Kafka. Is the broker running?")
        logger.info("Start Kafka with: docker-compose up -d kafka")
        return
    
    consumer.consume(max_messages=max_messages)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    max_messages = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_consumer_demo(max_messages=max_messages)
