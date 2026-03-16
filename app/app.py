"""Main Flask application factory"""
from flask import Flask, jsonify
from flask_cors import CORS
from app.config import get_config
from app.models.anomaly_log import db
from app.database.connection import init_db, configure_db_pool

# Import blueprints
from app.api.anomalies import anomalies_bp
from app.api.patients import patients_bp
from app.api.baselines import baselines_bp


def create_app(config=None):
    """
    Application factory function
    
    Args:
        config: Configuration object or None to use environment
    
    Returns:
        Flask app instance
    """
    app = Flask(__name__)
    
    # Load configuration
    if config is None:
        config = get_config()
    
    app.config.from_object(config)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Configure database pool
    configure_db_pool(app)
    
    # Register blueprints
    app.register_blueprint(anomalies_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(baselines_bp)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'Anomaly Detection Platform API'
        }), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            'message': 'Anomaly Detection Platform API',
            'version': '1.0.0',
            'endpoints': {
                'api': {
                    'anomalies': '/api/anomalies',
                    'patients': '/api/patients',
                    'baselines': '/api/baselines',
                }
            }
        }), 200
    
    # Initialize database
    with app.app_context():
        init_db(app)
    
    print("✓ Flask API initialized successfully")
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
