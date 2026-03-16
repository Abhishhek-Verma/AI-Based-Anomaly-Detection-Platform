import json
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class KafkaStreamConsumer:
    """Consume real-time healthcare data from Kafka"""
    
    def __init__(self, broker_address: str, topic: str, group_id: str = 'anomaly-detector'):
        """
        Initialize Kafka consumer
        
        Args:
            broker_address: Kafka broker address
            topic: Topic to consume from
            group_id: Consumer group ID
        """
        self.broker_address = broker_address
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        
    def connect(self) -> None:
        """Connect to Kafka broker"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.broker_address.split(','),
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            logger.info(f"Connected to Kafka Topic: {self.topic}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise
    
    def consume_messages(self, callback: Callable, timeout: Optional[int] = None) -> None:
        """
        Consume messages from Kafka topic
        
        Args:
            callback: Function to process each message
            timeout: Timeout in milliseconds
        """
        if self.consumer is None:
            self.connect()
        
        try:
            for message in self.consumer:
                logger.debug(f"Received message: {message.value}")
                callback(message.value)
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except KafkaError as e:
            logger.error(f"Kafka error: {str(e)}")
    
    def disconnect(self) -> None:
        """Disconnect from Kafka"""
        if self.consumer:
            self.consumer.close()
            logger.info("Disconnected from Kafka")


class KafkaStreamProducer:
    """Produce anomaly detection results to Kafka"""
    
    def __init__(self, broker_address: str, topic: str):
        """
        Initialize Kafka producer
        
        Args:
            broker_address: Kafka broker address
            topic: Topic to produce to
        """
        self.broker_address = broker_address
        self.topic = topic
        self.producer = None
        
    def connect(self) -> None:
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.broker_address.split(','),
                value_serializer=lambda m: json.dumps(m).encode('utf-8')
            )
            logger.info(f"Connected to Kafka producer for topic: {self.topic}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise
    
    def send_message(self, message: dict) -> None:
        """
        Send message to Kafka topic
        
        Args:
            message: Message dictionary to send
        """
        if self.producer is None:
            self.connect()
        
        try:
            future = self.producer.send(self.topic, message)
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to {record_metadata.topic} partition {record_metadata.partition}")
        except KafkaError as e:
            logger.error(f"Failed to send message: {str(e)}")
    
    def flush(self) -> None:
        """Flush pending messages"""
        if self.producer:
            self.producer.flush()
    
    def disconnect(self) -> None:
        """Disconnect from Kafka"""
        if self.producer:
            self.producer.close()
            logger.info("Disconnected from Kafka producer")
