"""
Kafka Producer - Simulates Live Patient Vital Signs
Generates realistic patient data and sends to Kafka topic
"""

import json
import random
import time
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

logger = logging.getLogger(__name__)


class PatientVitalSimulator:
    """
    Simulates realistic patient vital signs with normal and anomalous patterns.
    """
    
    def __init__(self, num_patients: int = 5):
        """
        Initialize vital simulator.
        
        Args:
            num_patients: Number of simulated patients
        """
        self.num_patients = num_patients
        self.patient_ids = [f"PAT_{str(i+1).zfill(4)}" for i in range(num_patients)]
        
        # Initialize patient baselines (individualized)
        self.baselines = {
            patient_id: {
                'HR': random.randint(60, 90),
                'SpO2': random.uniform(96, 99),
                'Temperature': random.uniform(36.5, 37.2),
                'SysBP': random.randint(100, 130),
                'DiaBP': random.randint(60, 85)
            }
            for patient_id in self.patient_ids
        }
        
        # Anomaly injection probability per patient
        self.anomaly_probability = 0.05  # 5% chance per reading
        self.escalation_mode = False
        self.escalation_patient = None
    
    def generate_vital(self, patient_id: str, inject_anomaly: bool = False) -> Dict:
        """
        Generate single vital signs reading for a patient.
        
        Args:
            patient_id: Patient identifier
            inject_anomaly: Force anomaly injection
        
        Returns:
            Dictionary with vital signs values
        """
        baseline = self.baselines[patient_id]
        
        # Generate normal variation
        vital = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'HR': baseline['HR'] + np.random.normal(0, 2),
            'SpO2': baseline['SpO2'] + np.random.normal(0, 0.5),
            'Temperature': baseline['Temperature'] + np.random.normal(0, 0.1),
            'SysBP': baseline['SysBP'] + np.random.normal(0, 3),
            'DiaBP': baseline['DiaBP'] + np.random.normal(0, 2)
        }
        
        # Decide anomaly injection
        if inject_anomaly or random.random() < self.anomaly_probability:
            anomaly_type = random.choice(['high_hr', 'low_spo2', 'fever', 'hypertension', 'hypotension'])
            
            if anomaly_type == 'high_hr':
                vital['HR'] = random.uniform(130, 180)  # Tachycardia
                logger.info(f"Injecting anomaly (high HR) for patient {patient_id}")
            
            elif anomaly_type == 'low_spo2':
                vital['SpO2'] = random.uniform(85, 92)  # Hypoxemia
                logger.info(f"Injecting anomaly (low SpO2) for patient {patient_id}")
            
            elif anomaly_type == 'fever':
                vital['Temperature'] = random.uniform(38.5, 40.5)  # High fever
                logger.info(f"Injecting anomaly (fever) for patient {patient_id}")
            
            elif anomaly_type == 'hypertension':
                vital['SysBP'] = random.uniform(150, 180)  # Hypertensions
                vital['DiaBP'] = random.uniform(100, 120)
                logger.info(f"Injecting anomaly (hypertension) for patient {patient_id}")
            
            elif anomaly_type == 'hypotension':
                vital['SysBP'] = random.uniform(70, 85)  # Hypotension
                vital['DiaBP'] = random.uniform(40, 55)
                logger.info(f"Injecting anomaly (hypotension) for patient {patient_id}")
        
        # Ensure realistic bounds
        vital['HR'] = np.clip(vital['HR'], 40, 200)
        vital['SpO2'] = np.clip(vital['SpO2'], 80, 100)
        vital['Temperature'] = np.clip(vital['Temperature'], 35, 41)
        vital['SysBP'] = np.clip(vital['SysBP'], 60, 200)
        vital['DiaBP'] = np.clip(vital['DiaBP'], 30, 130)
        
        return vital
    
    def generate_batch(self, batch_size: int = 100, anomaly_rate: float = 0.1) -> List[Dict]:
        """
        Generate batch of vitals from all patients.
        
        Args:
            batch_size: Number of total readings to generate
            anomaly_rate: Fraction of readings to inject anomalies
        
        Returns:
            List of vital readings
        """
        vitals = []
        num_anomalies = max(1, int(batch_size * anomaly_rate))
        anomaly_indices = set(random.sample(range(batch_size), num_anomalies))
        
        for i in range(batch_size):
            patient_id = self.patient_ids[i % self.num_patients]
            inject_anomaly = i in anomaly_indices
            vital = self.generate_vital(patient_id, inject_anomaly=inject_anomaly)
            vitals.append(vital)
        
        return vitals


class KafkaVitalProducer:
    """
    Kafka producer for patient vital signs.
    Sends real-time simulated or actual patient data to Kafka topic.
    """
    
    def __init__(self,
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'patient_vitals'):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Kafka topic for vitals
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.message_count = 0
    
    def connect(self) -> bool:
        """
        Connect to Kafka broker.
        
        Returns:
            True if connected successfully
        """
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                acks='all',  # Wait for all replicas
                retries=3
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def send_vital(self, vital: Dict) -> bool:
        """
        Send single vital reading to Kafka.
        
        Args:
            vital: Vital signs dictionary
        
        Returns:
            True if sent successfully
        """
        if not self.producer:
            logger.error("Producer not connected. Call connect() first.")
            return False
        
        try:
            # Use patient_id as key for partitioning
            future = self.producer.send(
                self.topic,
                value=vital,
                key=vital['patient_id'].encode('utf-8')
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            self.message_count += 1
            
            if self.message_count % 50 == 0:
                logger.info(f"Sent {self.message_count} messages to {self.topic}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send vital to Kafka: {e}")
            return False
    
    def send_batch(self, vitals: List[Dict]) -> int:
        """
        Send batch of vitals to Kafka.
        
        Args:
            vitals: List of vital readings
        
        Returns:
            Number of successfully sent messages
        """
        sent_count = 0
        for vital in vitals:
            if self.send_vital(vital):
                sent_count += 1
        
        return sent_count
    
    def stream_vitals(self,
                     simulator: PatientVitalSimulator,
                     interval: float = 1.0,
                     duration: Optional[float] = None) -> None:
        """
        Stream vitals continuously from simulator.
        
        Args:
            simulator: PatientVitalSimulator instance
            interval: Seconds between vital readings
            duration: Total duration in seconds (None = infinite)
        """
        if not self.producer:
            logger.error("Producer not connected. Call connect() first.")
            return
        
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    logger.info(f"Streaming duration reached. Stopping producer.")
                    break
                
                # Generate vitals for all patients
                for patient_id in simulator.patient_ids:
                    vital = simulator.generate_vital(patient_id)
                    self.send_vital(vital)
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Producer interrupted by user")
        
        finally:
            self.close()
    
    def close(self) -> None:
        """Close Kafka producer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info(f"Producer closed. Total messages sent: {self.message_count}")


def run_producer_demo(duration: int = 60,
                     num_patients: int = 5,
                     bootstrap_servers: str = 'localhost:9092'):
    """
    Run producer demo with simulated data.
    
    Args:
        duration: Duration in seconds
        num_patients: Number of simulated patients
        bootstrap_servers: Kafka broker addresses
    """
    logger.info("=" * 70)
    logger.info("KAFKA PRODUCER - PATIENT VITALS STREAMING")
    logger.info("=" * 70)
    
    # Initialize simulator and producer
    simulator = PatientVitalSimulator(num_patients=num_patients)
    producer = KafkaVitalProducer(bootstrap_servers=bootstrap_servers)
    
    # Connect to Kafka
    if not producer.connect():
        logger.error("Failed to connect to Kafka. Is the broker running?")
        logger.info("Start Kafka with: docker-compose up -d kafka")
        return
    
    # Stream vitals
    logger.info(f"Streaming {num_patients} patient vitals for {duration} seconds...")
    logger.info("(Press Ctrl+C to stop)")
    
    producer.stream_vitals(simulator, interval=1.0, duration=duration)


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    num_patients = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    run_producer_demo(duration=duration, num_patients=num_patients)
