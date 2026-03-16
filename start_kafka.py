"""
Kafka Broker Setup and Management
Instructions for starting Kafka locally
"""

import logging
import os
import platform

logger = logging.getLogger(__name__)


def print_kafka_setup_instructions():
    """Print instructions for starting Kafka"""
    
    print("\n" + "=" * 60)
    print("KAFKA SETUP INSTRUCTIONS")
    print("=" * 60)
    
    system = platform.system()
    
    if system == "Windows":
        print("\n[Windows Instructions]")
        print("1. Download Kafka from: https://kafka.apache.org/downloads")
        print("2. Extract to a directory (e.g., C:\\kafka)")
        print("3. Start ZooKeeper:")
        print("   cd C:\\kafka")
        print("   .\\bin\\windows\\zookeeper-server-start.bat .\\config\\zookeeper.properties")
        print("\n4. In a new terminal, start Kafka broker:")
        print("   cd C:\\kafka")
        print("   .\\bin\\windows\\kafka-server-start.bat .\\config\\server.properties")
    
    elif system == "Darwin":  # macOS
        print("\n[macOS Instructions]")
        print("1. Install Kafka via Homebrew:")
        print("   brew install kafka")
        print("\n2. Start ZooKeeper:")
        print("   zookeeper-server-start /usr/local/etc/kafka/zookeeper.properties")
        print("\n3. In a new terminal, start Kafka broker:")
        print("   kafka-server-start /usr/local/etc/kafka/server.properties")
    
    else:  # Linux
        print("\n[Linux Instructions]")
        print("1. Download Kafka from: https://kafka.apache.org/downloads")
        print("2. Extract to a directory (e.g., ~/kafka)")
        print("3. Start ZooKeeper:")
        print("   cd ~/kafka")
        print("   bin/zookeeper-server-start.sh config/zookeeper.properties")
        print("\n4. In a new terminal, start Kafka broker:")
        print("   cd ~/kafka")
        print("   bin/kafka-server-start.sh config/server.properties")
    
    print("\n5. Verify Kafka is running:")
    print("   - ZooKeeper should be running on port 2181")
    print("   - Kafka broker should be running on port 9092")
    
    print("\n6. Create topics (optional):")
    print("   kafka-topics --create --topic healthcare-data --bootstrap-server localhost:9092")
    print("   kafka-topics --create --topic anomalies-detected --bootstrap-server localhost:9092")
    
    print("\n" + "=" * 60)
    print("Configuration file location: config/.env")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    print_kafka_setup_instructions()
