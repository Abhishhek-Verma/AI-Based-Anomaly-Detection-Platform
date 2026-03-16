import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    
    # Database
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', 5432)
    DB_NAME = os.getenv('DB_NAME', 'healthcare_db')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # Connection string
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Kafka
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
    KAFKA_TOPIC_INPUT = os.getenv('KAFKA_TOPIC_INPUT', 'healthcare-data')
    KAFKA_TOPIC_ANOMALIES = os.getenv('KAFKA_TOPIC_ANOMALIES', 'anomalies-detected')
    
    # Flask
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', True)
    API_PORT = os.getenv('API_PORT', 5000)
    
    # Models
    MODEL_ISOLATION_FOREST_PATH = os.getenv('MODEL_ISOLATION_FOREST_PATH', 'models/isolation_forest.pkl')
    MODEL_AUTOENCODER_PATH = os.getenv('MODEL_AUTOENCODER_PATH', 'models/autoencoder.h5')
    ANOMALY_THRESHOLD = float(os.getenv('ANOMALY_THRESHOLD', 0.7))
    
    # Data
    DATA_PATH = os.getenv('DATA_PATH', 'data/')
    DATASET_NAME = os.getenv('DATASET_NAME', 'healthcare_data.csv')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'


def get_config(env=None):
    """Get configuration object based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'development':
        return DevelopmentConfig
    elif env == 'production':
        return ProductionConfig
    elif env == 'testing':
        return TestingConfig
    else:
        return DevelopmentConfig
