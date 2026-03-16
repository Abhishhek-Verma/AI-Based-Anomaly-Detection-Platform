"""Flask application configuration"""
import os
from datetime import timedelta

# Flask configuration
class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    TESTING = False
    
    # SQLAlchemy configuration
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:postgres@localhost:5432/anomaly_detection'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = DEBUG
    
    # Flask-CORS configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Dashboard configuration
    DASHBOARD_REFRESH_INTERVAL = 5000  # milliseconds
    DASHBOARD_MAX_HISTORICAL_POINTS = 100
    DASHBOARD_KPI_LOOKBACK_HOURS = 24
    
    # Anomaly thresholds for severity colors
    SEVERITY_THRESHOLD_HIGH = 0.67
    SEVERITY_THRESHOLD_MEDIUM = 0.33
    
    # Alert configuration
    ALERT_RETENTION_DAYS = 30
    
    # Pagination
    ITEMS_PER_PAGE = 50


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Ensure DATABASE_URL is set in production
    if not os.getenv('DATABASE_URL'):
        raise ValueError('DATABASE_URL environment variable must be set in production')


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    config_map = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
        'production': ProductionConfig,
    }
    return config_map.get(env, DevelopmentConfig)
