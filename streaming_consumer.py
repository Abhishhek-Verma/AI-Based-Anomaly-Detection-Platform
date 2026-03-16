"""
Streaming Consumer for Real-time Anomaly Detection
"""

import json
import logging
import numpy as np
from config.config import get_config
from utils.logger import setup_logging
from app.streaming.kafka_client import KafkaStreamConsumer, KafkaStreamProducer
from app.models.isolation_forest import IsolationForestAnomalyDetector
from app.models.autoencoder import AutoencoderAnomalyDetector

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class StreamingAnomalyDetector:
    """Real-time anomaly detection consumer"""
    
    def __init__(self):
        """Initialize streaming detector"""
        config = get_config()
        
        self.consumer = KafkaStreamConsumer(
            broker_address=config.KAFKA_BROKER,
            topic=config.KAFKA_TOPIC_INPUT
        )
        
        self.producer = KafkaStreamProducer(
            broker_address=config.KAFKA_BROKER,
            topic=config.KAFKA_TOPIC_ANOMALIES
        )
        
        # Initialize models
        self.if_model = IsolationForestAnomalyDetector()
        self.ae_model = None
        
        self.config = config
        self.message_count = 0
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            logger.info("Loading pre-trained models...")
            
            if os.path.exists(self.config.MODEL_ISOLATION_FOREST_PATH):
                self.if_model.load_model(self.config.MODEL_ISOLATION_FOREST_PATH)
                logger.info("Isolation Forest model loaded")
            else:
                logger.warning("Isolation Forest model not found")
            
            if os.path.exists(self.config.MODEL_AUTOENCODER_PATH):
                # Initialize autoencoder with expected input dimension
                self.ae_model = AutoencoderAnomalyDetector(input_dim=8)
                self.ae_model.load_model(self.config.MODEL_AUTOENCODER_PATH)
                logger.info("Autoencoder model loaded")
            else:
                logger.warning("Autoencoder model not found")
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def process_message(self, message):
        """
        Process incoming message and detect anomalies
        
        Args:
            message: Dictionary containing health data
        """
        try:
            self.message_count += 1
            
            # Extract features
            features = self._extract_features(message)
            if features is None:
                return
            
            X = np.array(features).reshape(1, -1)
            
            # Run Isolation Forest
            if_prediction = self.if_model.predict(X)[0]
            if_score = self.if_model.predict_proba(X)[0]
            
            # Prepare result
            result = {
                'timestamp': message.get('timestamp'),
                'patient_id': message.get('patient_id'),
                'isolation_forest_anomaly': int(if_prediction == -1),
                'isolation_forest_score': float(if_score),
                'messages_processed': self.message_count
            }
            
            # Run Autoencoder if available
            if self.ae_model:
                try:
                    ae_errors = self.ae_model.get_reconstruction_error(X)
                    ae_anomaly = int(ae_errors[0] > self.ae_model.threshold) if self.ae_model.threshold else 0
                    result['autoencoder_anomaly'] = ae_anomaly
                    result['autoencoder_error'] = float(ae_errors[0])
                except Exception as e:
                    logger.debug(f"Autoencoder error: {str(e)}")
            
            # Log and publish if anomaly detected
            if result.get('isolation_forest_anomaly', 0):
                logger.warning(f"ANOMALY DETECTED: {result}")
                self.producer.send_message(result)
            
            if self.message_count % 100 == 0:
                logger.info(f"Processed {self.message_count} messages")
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _extract_features(self, message):
        """Extract features from message"""
        try:
            features = [
                message.get('heart_rate', 0),
                message.get('blood_pressure_sys', 0),
                message.get('blood_pressure_dia', 0),
                message.get('temperature', 0),
                message.get('oxygen_saturation', 0),
                message.get('glucose_level', 0),
                message.get('cholesterol', 0),
                message.get('timestamp', 0)
            ]
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def start(self):
        """Start consuming messages"""
        logger.info("=" * 60)
        logger.info("Streaming Consumer - Real-time Anomaly Detection")
        logger.info("=" * 60)
        
        try:
            self.consumer.connect()
            self.load_models()
            
            logger.info("Starting to consume messages...")
            logger.info("Press Ctrl+C to stop")
            
            self.consumer.consume_messages(self.process_message)
        
        except KeyboardInterrupt:
            logger.info("\nConsumer stopped by user")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
        finally:
            self.consumer.disconnect()
            self.producer.disconnect()
            logger.info("Consumer disconnected")


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    detector = StreamingAnomalyDetector()
    detector.start()
