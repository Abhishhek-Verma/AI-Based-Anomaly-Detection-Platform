"""
Kafka Producer - Publish healthcare data to Kafka topic
"""

import json
import time
import logging
import numpy as np
from config.config import get_config
from utils.logger import setup_logging
from app.streaming.kafka_client import KafkaStreamProducer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def generate_health_data():
    """Generate sample health data"""
    data = {
        'patient_id': f'P{np.random.randint(1000, 9999)}',
        'heart_rate': float(np.random.normal(75, 15)),
        'blood_pressure_sys': float(np.random.normal(120, 15)),
        'blood_pressure_dia': float(np.random.normal(80, 10)),
        'temperature': float(np.random.normal(98.6, 1)),
        'oxygen_saturation': float(np.random.normal(97, 2)),
        'glucose_level': float(np.random.normal(100, 20)),
        'cholesterol': float(np.random.normal(200, 40)),
        'timestamp': time.time()
    }
    return data


def main():
    """Main producer function"""
    logger.info("=" * 60)
    logger.info("Kafka Producer - Healthcare Data Publisher")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Initialize producer
    producer = KafkaStreamProducer(
        broker_address=config.KAFKA_BROKER,
        topic=config.KAFKA_TOPIC_INPUT
    )
    producer.connect()
    
    logger.info(f"Publishing data to topic: {config.KAFKA_TOPIC_INPUT}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        counter = 0
        while True:
            # Generate and send health data
            data = generate_health_data()
            producer.send_message(data)
            
            counter += 1
            if counter % 10 == 0:
                logger.info(f"Published {counter} messages")
            
            time.sleep(2)  # Publish every 2 seconds
    
    except KeyboardInterrupt:
        logger.info("\nProducer stopped by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        producer.disconnect()
        logger.info("Producer disconnected")


if __name__ == '__main__':
    main()
