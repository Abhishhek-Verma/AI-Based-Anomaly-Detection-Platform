import logging
import os
from datetime import datetime


def setup_logging(log_file: str = 'logs/app.log', log_level: str = 'INFO') -> None:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        log_level: Logging level
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Format for logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()
