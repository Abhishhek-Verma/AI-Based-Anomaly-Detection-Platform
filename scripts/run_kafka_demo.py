#!/usr/bin/env python3
"""
Kafka Streaming Demo - Producer and Consumer
Demonstrates real-time anomaly detection with patient vitals
"""

import sys
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_kafka_running(bootstrap_servers: str = 'localhost:9092') -> bool:
    """Check if Kafka broker is running"""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        producer.close()
        return True
    except:
        return False


def start_kafka_docker() -> bool:
    """Start Kafka using Docker Compose"""
    logger.info("Starting Kafka with Docker Compose...")
    
    # Create docker-compose.yml if it doesn't exist
    docker_compose_content = """version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
"""
    
    # Write docker-compose.yml
    docker_compose_path = Path("docker-compose.yml")
    with open(docker_compose_path, 'w') as f:
        f.write(docker_compose_content)
    
    logger.info("Created docker-compose.yml")
    
    # Start services
    try:
        result = subprocess.run(['docker-compose', 'up', '-d'], cwd=Path.cwd())
        if result.returncode != 0:
            logger.error("Failed to start Docker Compose")
            return False
        
        logger.info("Kafka started. Waiting for broker to be ready...")
        
        # Wait for Kafka to be ready
        for i in range(30):
            if check_kafka_running():
                logger.info("Kafka is ready!")
                return True
            time.sleep(1)
        
        logger.error("Kafka did not become ready in time")
        return False
    
    except FileNotFoundError:
        logger.error("Docker Compose not found. Install Docker Desktop or Docker Compose.")
        return False


def run_producer(duration: int = 60, num_patients: int = 5) -> subprocess.Popen:
    """Start Kafka producer in subprocess"""
    logger.info(f"Starting Kafka Producer (duration: {duration}s, patients: {num_patients})")
    
    cmd = [
        sys.executable,
        '-m', 'app.streaming.kafka_producer',
        str(duration),
        str(num_patients)
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process


def run_consumer(max_messages: int = 200) -> subprocess.Popen:
    """Start Kafka consumer in subprocess"""
    logger.info(f"Starting Kafka Consumer (max messages: {max_messages})")
    
    cmd = [
        sys.executable,
        '-m', 'app.streaming.kafka_consumer',
        str(max_messages)
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process


def main(duration: int = 60, num_patients: int = 5, max_messages: int = 200):
    """
    Run complete demo with Kafka producer and consumer.
    
    Args:
        duration: Producer duration in seconds
        num_patients: Number of simulated patients
        max_messages: Max messages for consumer
    """
    
    logger.info("=" * 70)
    logger.info("KAFKA STREAMING DEMO - Real-Time Anomaly Detection")
    logger.info("=" * 70)
    
    # Check if Kafka is running
    if not check_kafka_running():
        logger.warning("Kafka broker not detected at localhost:9092")
        logger.info("Attempting to start Kafka with Docker Compose...")
        
        if not start_kafka_docker():
            logger.error("Could not start Kafka. Please start manually:")
            logger.error("  1. Install Docker: https://www.docker.com/products/docker-desktop")
            logger.error("  2. Run: docker-compose up -d")
            return
    else:
        logger.info("Kafka broker already running")
    
    # Start producer
    logger.info("\n" + "-" * 70)
    producer_process = run_producer(duration=duration, num_patients=num_patients)
    
    # Give producer time to start
    time.sleep(3)
    
    # Start consumer
    logger.info("-" * 70)
    consumer_process = run_consumer(max_messages=max_messages)
    
    # Wait for both processes
    logger.info("\nProducer and Consumer running. Press Ctrl+C to stop.\n")
    
    try:
        producer_process.wait()
        logger.info("Producer finished")
        
        # Give consumer time to process remaining messages
        time.sleep(5)
        
        consumer_process.terminate()
        consumer_process.wait(timeout=10)
        logger.info("Consumer finished")
    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        producer_process.terminate()
        consumer_process.terminate()
        
        try:
            producer_process.wait(timeout=5)
            consumer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            producer_process.kill()
            consumer_process.kill()
    
    logger.info("\n" + "=" * 70)
    logger.info("Demo complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Kafka Streaming Demo - Real-Time Anomaly Detection'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Producer duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--patients',
        type=int,
        default=5,
        help='Number of simulated patients (default: 5)'
    )
    parser.add_argument(
        '--messages',
        type=int,
        default=200,
        help='Max messages for consumer (default: 200)'
    )
    
    args = parser.parse_args()
    
    main(duration=args.duration, num_patients=args.patients, max_messages=args.messages)
