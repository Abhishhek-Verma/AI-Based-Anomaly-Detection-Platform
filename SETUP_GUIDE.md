# AI-Driven Healthcare Anomaly Detection System - Setup & Usage Guide

## Project Overview

This is a complete AI-driven healthcare anomaly detection system built with Python 3.9+, featuring real-time streaming with Kafka, machine learning models (Isolation Forest & Autoencoder), PostgreSQL database integration, and a Flask REST API.

## Final Project Structure

```
AI-Based-Anomaly-Detection-Platform/
├── app/                           # Main application code
│   ├── __init__.py
│   ├── preprocessing/             # Data preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Dataset loading & validation
│   │   └── preprocessor.py       # Feature preprocessing
│   ├── models/                    # ML models
│   │   ├── __init__.py
│   │   ├── isolation_forest.py   # Isolation Forest detector
│   │   └── autoencoder.py        # Autoencoder detector
│   ├── streaming/                 # Real-time Kafka integration
│   │   ├── __init__.py
│   │   └── kafka_client.py       # Kafka producer/consumer
│   ├── api/                       # Flask REST API
│   │   ├── __init__.py
│   │   ├── app.py                # Flask app factory
│   │   └── routes.py             # API endpoints
│   └── database/                  # Database management
│       ├── __init__.py
│       └── db_manager.py         # PostgreSQL manager
├── utils/                         # Utility modules
│   ├── __init__.py
│   └── logger.py                 # Logging setup
├── config/                        # Configuration files
│   ├── __init__.py
│   └── config.py                 # Environment-based config
├── data/                          # Datasets
│   └── healthcare_data.csv       # Sample dataset
├── models/                        # Trained model files
│   ├── isolation_forest.pkl
│   └── autoencoder.h5
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── venv/                          # Virtual environment
├── main.py                        # Flask API server
├── start_all.py                   # Start all components
├── start_producer.py              # Kafka data producer
├── start_consumer.py              # Kafka data consumer
├── start_kafka.py                 # Kafka setup instructions
├── streaming_consumer.py           # Real-time anomaly detector
├── setup.py                       # Setup script
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .env                           # Environment variables (local)
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

## Installation & Setup

### Step 1: Clone the Repository
```bash
cd AI-Based-Anomaly-Detection-Platform
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env with your database and Kafka configurations
```

### Step 5: Validate Setup
```bash
python setup.py
```

### Step 6: Generate Sample Data
```bash
python scripts/generate_sample_data.py
```

## Usage

### Option 1: Run Everything Together
```bash
python start_all.py
```

This starts:
- Flask API server (port 5000)
- Kafka producer (publishes health data)
- Kafka consumer (processes messages)

### Option 2: Run Individual Components

**Start Flask API Server:**
```bash
python main.py
```
API available at: `http://localhost:5000`

**Start Kafka Producer (publish data):**
```bash
python start_producer.py
```

**Start Kafka Consumer (consume data):**
```bash
python start_consumer.py
```

**Start Real-time Anomaly Detector:**
```bash
python streaming_consumer.py
```

**Setup Kafka (instructions):**
```bash
python start_kafka.py
```

## API Endpoints

### Health Check
```bash
GET /api/health/
```

### Predict Anomalies
```bash
POST /api/predict/anomaly
Content-Type: application/json

{
  "features": [75.5, 120, 80, 98.6, 97.2, 100, 200]
}
```

### Model Status
```bash
GET /api/model/status
```

### Retrain Model
```bash
POST /api/model/retrain
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.10.0 | Deep learning (Autoencoder) |
| scikit-learn | ≥1.0.0 | Isolation Forest model |
| kafka-python | ≥2.0.2 | Real-time data streaming |
| flask | ≥2.3.0 | REST API server |
| psycopg2-binary | ≥2.9.0 | PostgreSQL connectivity |
| pandas | ≥1.5.0 | Data processing |
| numpy | ≥1.21.0 | Numerical computing |
| pytest | ≥7.0.0 | Testing framework |

## Configuration

### Environment Variables (.env)

```ini
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=healthcare_db
DB_USER=postgres
DB_PASSWORD=your_password

# Kafka
KAFKA_BROKER=localhost:9092
KAFKA_TOPIC_INPUT=healthcare-data
KAFKA_TOPIC_ANOMALIES=anomalies-detected

# Flask API
FLASK_ENV=development
API_PORT=5000

# Models
MODEL_ISOLATION_FOREST_PATH=models/isolation_forest.pkl
MODEL_AUTOENCODER_PATH=models/autoencoder.h5
ANOMALY_THRESHOLD=0.7

# Data
DATA_PATH=data/
DATASET_NAME=healthcare_data.csv

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Data Format

Healthcare messages should have the following structure:

```json
{
  "patient_id": "P12345",
  "heart_rate": 75.5,
  "blood_pressure_sys": 120,
  "blood_pressure_dia": 80,
  "temperature": 98.6,
  "oxygen_saturation": 97.2,
  "glucose_level": 100,
  "cholesterol": 200,
  "timestamp": 1234567890.5
}
```

## Database Setup

### PostgreSQL Tables

The system automatically creates:
- `healthcare_records`: Stores incoming health data
- `anomalies`: Stores detected anomalies

### Manual Table Creation

```sql
CREATE TABLE healthcare_records (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_1 FLOAT,
    feature_2 FLOAT,
    feature_3 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE anomalies (
    id SERIAL PRIMARY KEY,
    record_id INTEGER REFERENCES healthcare_records(id),
    anomaly_score FLOAT,
    is_anomaly BOOLEAN,
    model_type VARCHAR(50),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Models

### Isolation Forest
- **Purpose**: Unsupervised anomaly detection
- **Algorithm**: Isolation-based outlier detection
- **Contamination**: 10% (expected anomaly rate)
- **Location**: `models/isolation_forest.pkl`

### Autoencoder
- **Purpose**: Deep learning-based anomaly detection
- **Architecture**: 
  - Input: 8 features
  - Encoder: Dense(16) → Dense(8)
  - Decoder: Dense(16) → Dense(8)
- **Threshold**: 95th percentile of reconstruction error
- **Location**: `models/autoencoder.h5`

## Troubleshooting

### Kafka Connection Issues
1. Ensure Kafka is running on `localhost:9092`
2. Check `.env` KAFKA_BROKER setting
3. Run `python start_kafka.py` for setup instructions

### TensorFlow Import Errors
```bash
pip install --upgrade tensorflow protobuf==6.31.1
```

### PostgreSQL Connection Issues
1. Verify PostgreSQL is running
2. Check database credentials in `.env`
3. Ensure database exists: `createdb healthcare_db`

### Port Already in Use
```bash
# Change API_PORT in .env
API_PORT=5001
```

## Performance Optimization

1. **Batch Processing**: Process messages in batches for better throughput
2. **Model Caching**: Load models once, reuse for multiple predictions
3. **Database Indexing**: Add indexes on frequently queried columns
4. **Kafka Partitioning**: Use multiple partitions for parallel processing

## Security Considerations

1. Change default database passwords
2. Use environment variables for sensitive data
3. Enable HTTPS for production API
4. Implement authentication/authorization
5. Validate all input data

## Logging

Logs are written to:
- Console (INFO level and above)
- File: `logs/app.log` (configurable in `.env`)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Development

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings for all functions

### Adding New Features
1. Create feature branch: `git checkout -b feature/your-feature`
2. Write tests in `tests/`
3. Update documentation
4. Submit pull request

## License

[Add your license]

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in `logs/app.log`
3. Check Kafka broker status
4. Verify database connectivity

## Contact

Development Team
Last Updated: March 2024
