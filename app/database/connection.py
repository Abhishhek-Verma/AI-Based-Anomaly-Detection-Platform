"""Database initialization and utilities"""
from flask import Flask
from sqlalchemy import event
from sqlalchemy.pool import Pool
from app.models.anomaly_log import db, AnomalyLog, PatientBaseline, PatientStats


def init_db(app: Flask):
    """Initialize database with app context"""
    with app.app_context():
        db.create_all()
        print("✓ Database tables created")


def reset_db(app: Flask):
    """Drop all tables and recreate (for testing/dev)"""
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("✓ Database reset complete")


def get_db_stats(app: Flask) -> dict:
    """Get database statistics"""
    with app.app_context():
        anomaly_count = AnomalyLog.query.count()
        patient_count = PatientBaseline.query.count()
        high_severity_count = AnomalyLog.query.filter_by(severity='HIGH').count()
        
        return {
            'total_anomalies': anomaly_count,
            'total_patients': patient_count,
            'high_severity_anomalies': high_severity_count,
        }


def configure_db_pool(app: Flask):
    """Configure SQLAlchemy connection pool"""
    @event.listens_for(Pool, "connect")
    def receive_connect(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        # Enable UUID extension for PostgreSQL
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            dbapi_conn.commit()
            cursor.close()
        except:
            pass
