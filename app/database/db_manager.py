import psycopg2
from psycopg2 import sql
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage PostgreSQL database connections"""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """
        Initialize database manager
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
    def connect(self) -> None:
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connected to database: {self.database}")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from database")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> None:
        """
        Execute INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query
            params: Query parameters
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            logger.info(f"Query executed successfully")
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def fetch_data(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """
        Fetch data from database
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of dictionaries containing fetched data
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Fetch all rows and convert to list of dicts
            rows = cursor.fetchall()
            result = [dict(zip(column_names, row)) for row in rows]
            
            return result
        except psycopg2.Error as e:
            logger.error(f"Fetch operation failed: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def create_tables(self) -> None:
        """Create required database tables"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS healthcare_records (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            feature_1 FLOAT,
            feature_2 FLOAT,
            feature_3 FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS anomalies (
            id SERIAL PRIMARY KEY,
            record_id INTEGER REFERENCES healthcare_records(id),
            anomaly_score FLOAT,
            is_anomaly BOOLEAN,
            model_type VARCHAR(50),
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_tables_sql)
            self.connection.commit()
            logger.info("Database tables created successfully")
        except psycopg2.Error as e:
            logger.error(f"Failed to create tables: {str(e)}")
        finally:
            cursor.close()
