"""
Start all components of the system
"""

import os
import sys
import subprocess
import logging
from config.config import get_config
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Start all system components"""
    logger.info("=" * 60)
    logger.info("Starting AI-Driven Healthcare Anomaly Detection System")
    logger.info("=" * 60)
    
    processes = []
    
    try:
        # Start Kafka producer
        logger.info("\nStarting Kafka Producer...")
        producer_proc = subprocess.Popen([sys.executable, 'start_producer.py'])
        processes.append(('Producer', producer_proc))
        
        # Start Kafka consumer
        logger.info("Starting Kafka Consumer...")
        consumer_proc = subprocess.Popen([sys.executable, 'start_consumer.py'])
        processes.append(('Consumer', consumer_proc))
        
        # Start Flask API server
        logger.info("Starting Flask API Server...")
        api_proc = subprocess.Popen([sys.executable, 'main.py'])
        processes.append(('API Server', api_proc))
        
        logger.info("\nAll components started successfully!")
        logger.info("Press Ctrl+C to stop all services")
        
        # Wait for all processes
        for name, proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        logger.info("\nStopping all components...")
        for name, proc in processes:
            logger.info(f"Stopping {name}...")
            proc.terminate()
            proc.wait()
        logger.info("All components stopped")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        for name, proc in processes:
            proc.terminate()
            proc.wait()


if __name__ == '__main__':
    main()
