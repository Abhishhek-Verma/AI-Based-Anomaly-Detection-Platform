"""
Kafka Consumer - Consume and process healthcare data
"""

import json
import logging
import numpy as np
from config.config import get_config
from utils.logger import setup_logging
from app.streaming.kafka_client import KafkaStreamConsumer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def process_message(message):
    """
    Process incoming health data
    
    Args:
        message: Dictionary containing health data
    """
    try:
        logger.info(f"Processing message - Patient: {message.get('patient_id')}")
        logger.debug(f"Data: {message}")
        
        # TODO: Implement anomaly detection logic
        # 1. Extract features from message
        # 2. Run through ML models
        # 3. Publish results if anomaly detected
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")


def main():
    """Main consumer function"""
    logger.info("=" * 60)
    logger.info("Kafka Consumer - Healthcare Data Processor")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Initialize consumer
    consumer = KafkaStreamConsumer(
        broker_address=config.KAFKA_BROKER,
        topic=config.KAFKA_TOPIC_INPUT,
        group_id='anomaly-detector'
    )
    consumer.connect()
    
    logger.info(f"Consuming from topic: {config.KAFKA_TOPIC_INPUT}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Start consuming messages
        consumer.consume_messages(process_message)
    
    except KeyboardInterrupt:
        logger.info("\nConsumer stopped by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        consumer.disconnect()
        logger.info("Consumer disconnected")


if __name__ == '__main__':
    main()
