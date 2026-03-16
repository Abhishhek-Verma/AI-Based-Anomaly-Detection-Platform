# AI-Driven Healthcare Anomaly Detection System

## Project Overview
This project implements an AI-driven healthcare anomaly detection system using machine learning models to identify unusual patterns in healthcare datasets in real-time.

## Key Features
- **Data Preprocessing**: Complete pipeline for data cleaning, validation, and normalization
- **Anomaly Detection Models**: 
  - Isolation Forest for unsupervised anomaly detection
  - Autoencoder for deep learning-based detection
- **Real-time Streaming**: Kafka integration for streaming healthcare data
- **Backend API**: Flask-based REST API for predictions and model management
- **Database Integration**: PostgreSQL for data persistence
- **Visualization Support**: Matplotlib for data visualization

## Project Structure

```
.
├── config/                      # Configuration files
│   ├── config.py               # Main configuration
│   └── __init__.py
├── src/                         # Source code
│   ├── preprocessing/          # Data preprocessing modules
│   │   ├── data_loader.py     # Dataset loading and validation
│   │   ├── preprocessor.py    # Data preprocessing utilities
│   │   └── __init__.py
│   ├── models/                # Anomaly detection models
│   │   ├── isolation_forest.py # Isolation Forest implementation
│   │   ├── autoencoder.py     # Autoencoder implementation
│   │   └── __init__.py
│   ├── streaming/             # Real-time data streaming
│   │   ├── kafka_client.py    # Kafka producer/consumer
│   │   └── __init__.py
│   ├── api/                   # Backend API endpoints
│   │   ├── app.py             # Flask app factory
│   │   ├── routes.py          # API routes
│   │   └── __init__.py
│   ├── database/              # Database connectivity
│   │   ├── db_manager.py      # PostgreSQL manager
│   │   └── __init__.py
│   ├── utils/                 # Utility functions
│   │   ├── logger.py          # Logging setup
│   │   └── __init__.py
│   └── __init__.py
├── data/                        # Dataset directory
├── models/                      # Trained models
├── tests/                       # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── __init__.py
├── scripts/                     # Utility scripts
│   ├── validate_dataset.py     # Dataset validation script
│   └── train_models.py         # Model training script
├── notebooks/                   # Jupyter notebooks
├── main.py                      # Application entry point
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file

```

## Dependencies

All required dependencies are listed in `requirements.txt`:

- **Data Processing**: pandas, numpy, scikit-learn
- **Deep Learning**: tensorflow, keras
- **Real-time Streaming**: kafka-python
- **Database**: psycopg2-binary
- **Backend API**: flask, flask-cors, flask-restx
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, pytest-cov
- **Utilities**: python-dotenv, jupyter, scipy, requests

## Installation

### 1. Create a Python Virtual Environment
```bash
python -m venv venv
source venv\Scripts\activate  # On Windows
source venv/bin/activate      # On macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Validate Dataset
```bash
python scripts/validate_dataset.py
```

## Usage

### Start the Application
```bash
python main.py
```

### Train Models
```bash
python scripts/train_models.py
```

### Run Tests
```bash
pytest tests/ -v
```

## API Endpoints

- `GET /api/health/` - Health check
- `POST /api/predict/anomaly` - Predict anomalies
- `GET /api/model/status` - Get model status
- `POST /api/model/retrain` - Trigger model retraining

## Configuration

Configuration is managed through environment variables and the `config/config.py` file. Key settings:

- Database credentials
- Kafka broker address
- Flask API port
- Model paths and thresholds
- Logging configuration

## Development

### Setting Up for Development
1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`
4. Start the dev server: `python main.py`

### Project Standards
- Python 3.9+
- PEP 8 code style
- Comprehensive logging
- Unit test coverage

## License
[Add your license information]

## Contact
For questions or issues, please contact the development team.