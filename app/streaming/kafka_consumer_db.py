"""
Kafka Consumer with PostgreSQL Storage
Extends the Kafka consumer to store anomalies in PostgreSQL database
"""

import json
import logging
from typing import Optional, Dict
from datetime import datetime
from app.streaming.kafka_consumer import RealtimeAnomalyDetector, KafkaVitalConsumer
from app.models.anomaly_log import AnomalyLog, PatientBaseline, PatientStats, SeverityEnum, db

logger = logging.getLogger(__name__)


class DatabaseStoringAnomalyDetector:
    """
    Wraps RealtimeAnomalyDetector to also store results in PostgreSQL
    """
    
    def __init__(self, flask_app=None, **kwargs):
        """
        Initialize detector with database support
        
        Args:
            flask_app: Flask app instance for database context
            **kwargs: Arguments for RealtimeAnomalyDetector
        """
        self.flask_app = flask_app
        self.detector = RealtimeAnomalyDetector(**kwargs)
    
    def process_vital_with_storage(self, vital_dict: Dict) -> Optional[Dict]:
        """
        Process vital and store anomaly in database
        
        Args:
            vital_dict: Dictionary with vital signs
        
        Returns:
            Result dictionary with anomaly info
        """
        # Run normal processing
        result = self.detector.process_vital(vital_dict)
        
        if result and self.flask_app:
            # Store in database if anomaly detected
            if result.get('status') == 'anomaly_detected':
                try:
                    with self.flask_app.app_context():
                        self._store_anomaly_to_db(vital_dict, result)
                        # Update patient stats
                        self._update_patient_stats(vital_dict, result)
                except Exception as e:
                    logger.error(f"Error storing to database: {e}")
        
        return result
    
    def _store_anomaly_to_db(self, vital_dict: Dict, result: Dict) -> None:
        """
        Store anomaly log entry in PostgreSQL
        """
        try:
            explanation = result.get('explanation', {})
            
            anomaly_log = AnomalyLog(
                patient_id=vital_dict['patient_id'],
                timestamp=datetime.fromisoformat(result['timestamp']),
                anomaly_score=result['anomaly_score'],
                autoencoder_score=result['ae_score'],
                isolation_forest_score=result['if_score'],
                severity=result['severity'],
                vital_signs={
                    'HR': vital_dict['HR'],
                    'SpO2': vital_dict['SpO2'],
                    'Temperature': vital_dict['Temperature'],
                    'SysBP': vital_dict['SysBP'],
                    'DiaBP': vital_dict['DiaBP'],
                },
                baseline_vitals=explanation.get('baseline_vitals', {}),
                vital_deviations=explanation.get('vital_deviations', {}),
                abnormal_vitals=explanation.get('abnormal_vitals', []),
                primary_contributor=explanation.get('primary_contributor'),
                primary_contributor_percentage=explanation.get('primary_contributor_percentage'),
                explanation=explanation.get('explanation_text'),
                recommendation=explanation.get('recommendation'),
                alert_sent=result.get('alert_sent', False),
                alert_timestamp=datetime.utcnow() if result.get('alert_sent') else None,
            )
            
            db.session.add(anomaly_log)
            db.session.commit()
            
            logger.debug(f"Stored anomaly for patient {vital_dict['patient_id']} in database")
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to store anomaly: {e}")
    
    def _update_patient_stats(self, vital_dict: Dict, result: Dict) -> None:
        """
        Update patient statistics in database
        """
        try:
            patient_id = vital_dict['patient_id']
            
            # Get or create patient stats
            stats = PatientStats.query.filter_by(patient_id=patient_id).first()
            if not stats:
                stats = PatientStats(patient_id=patient_id)
            
            # Update vital info
            stats.last_hr = vital_dict['HR']
            stats.last_spo2 = vital_dict['SpO2']
            stats.last_temperature = vital_dict['Temperature']
            stats.last_sysbp = vital_dict['SysBP']
            stats.last_diabp = vital_dict['DiaBP']
            stats.last_vital_timestamp = datetime.fromisoformat(result['timestamp'])
            
            # Update latest anomaly info
            stats.latest_severity = result['severity']
            stats.latest_anomaly_timestamp = datetime.fromisoformat(result['timestamp'])
            
            db.session.merge(stats)
            db.session.commit()
            
            logger.debug(f"Updated stats for patient {patient_id}")
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to update stats: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics from original detector"""
        return self.detector.get_statistics()


class DatabaseStoringKafkaConsumer:
    """
    Extends KafkaVitalConsumer to store anomalies in PostgreSQL
    """
    
    def __init__(self, flask_app=None, **kwargs):
        """
        Initialize Kafka consumer with database support
        
        Args:
            flask_app: Flask app instance for database context
            **kwargs: Arguments for KafkaVitalConsumer
        """
        self.flask_app = flask_app
        self.consumer = KafkaVitalConsumer(**kwargs)
        # Replace detector with database-storing version
        self.consumer.detector = DatabaseStoringAnomalyDetector(
            flask_app=flask_app,
            models_dir=kwargs.get('models_dir', 'models')
        )
        self.message_count = 0
    
    def connect(self) -> bool:
        """Connect to Kafka"""
        return self.consumer.connect()
    
    def consume(self, max_messages: Optional[int] = None) -> None:
        """
        Consume messages from Kafka with database storage
        
        Args:
            max_messages: Max messages to consume
        """
        if not self.consumer.consumer:
            logger.error("Consumer not connected. Call connect() first.")
            return
        
        try:
            logger.info(f"Starting Kafka consumer... (max_messages={max_messages})")
            
            for message in self.consumer.consumer:
                try:
                    vital_dict = message.value
                    
                    # Process with database storage
                    if isinstance(self.consumer.detector, DatabaseStoringAnomalyDetector):
                        result = self.consumer.detector.process_vital_with_storage(vital_dict)
                    else:
                        result = self.consumer.detector.process_vital(vital_dict)
                    
                    # Log result
                    if result:
                        status_icon = '✓' if result.get('status') == 'normal' else '🚨'
                        if result.get('status') != 'buffering':
                            logger.info(
                                f"{status_icon} [{result['status'].upper()}] "
                                f"{result['patient_id']}: Score={result['anomaly_score']:.4f}, "
                                f"Severity={result['severity']}"
                            )
                    
                    self.message_count += 1
                    
                    # Check max_messages limit
                    if max_messages and self.message_count >= max_messages:
                        logger.info(f"Reached max_messages limit: {max_messages}")
                        break
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            if self.consumer.consumer:
                self.consumer.consumer.close()
            logger.info(f"Consumer stopped. Processed {self.message_count} messages")
            
            # Print statistics
            stats = self.get_statistics()
            logger.info("\n===== CONSUMER STATISTICS =====")
            logger.info(f"Total Messages: {stats['total_messages']}")
            logger.info(f"Total Anomalies: {stats['total_anomalies']}")
            logger.info(f"Total Alerts: {stats['total_alerts_sent']}")
            logger.info(f"Anomaly Rate: {stats['anomaly_rate']:.2%}")
            logger.info(f"Active Patients: {stats['active_patients']}")
            logger.info("================================")
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        return self.consumer.detector.get_statistics()


def run_consumer_with_db(app, max_messages: Optional[int] = None) -> None:
    """
    Run Kafka consumer with database storage
    
    Args:
        app: Flask application instance
        max_messages: Max messages to process
    """
    consumer = DatabaseStoringKafkaConsumer(
        flask_app=app,
        bootstrap_servers='localhost:9092',
        topic='patient_vitals',
        group_id='anomaly_detection_consumer',
        models_dir='models'
    )
    
    if consumer.connect():
        consumer.consume(max_messages=max_messages)
    else:
        logger.error("Failed to connect to Kafka")


if __name__ == '__main__':
    # This would be run with Flask app context
    print("Use run_consumer_with_db(app) to start consumer with database storage")
