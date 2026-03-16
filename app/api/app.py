from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from config.config import get_config

logger = logging.getLogger(__name__)


def create_app(env: str = 'development') -> Flask:
    """
    Application factory for Flask app
    
    Args:
        env: Environment type
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(env)
    app.config.from_object(config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from app.api.routes import health_bp, prediction_bp, model_bp
    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(model_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500
    
    logger.info(f"Flask app created with {env} configuration")
    return app
