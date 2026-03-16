-- Initialize Anomaly Detection Database
-- This script is run when the PostgreSQL container starts

-- Create enum types
CREATE TYPE severity_enum AS ENUM ('LOW', 'MEDIUM', 'HIGH');

-- Create anomaly_logs table
CREATE TABLE IF NOT EXISTS anomaly_logs (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Anomaly scores
    anomaly_score FLOAT NOT NULL,
    autoencoder_score FLOAT NOT NULL,
    isolation_forest_score FLOAT NOT NULL,
    
    -- Severity
    severity severity_enum NOT NULL DEFAULT 'LOW',
    
    -- Vital signs (JSON)
    vital_signs JSONB NOT NULL,
    baseline_vitals JSONB NOT NULL,
    vital_deviations JSONB NOT NULL,
    
    -- Explanation data
    abnormal_vitals TEXT[],
    primary_contributor VARCHAR(50),
    primary_contributor_percentage FLOAT,
    explanation TEXT,
    recommendation TEXT,
    
    -- Alert information
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_timestamp TIMESTAMP,
    
    -- Metadata
    window_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    CONSTRAINT pk_anomaly_logs PRIMARY KEY (id)
);

CREATE INDEX idx_patient_timestamp ON anomaly_logs(patient_id, timestamp DESC);
CREATE INDEX idx_severity_timestamp ON anomaly_logs(severity, timestamp DESC);
CREATE INDEX idx_alert_sent ON anomaly_logs(alert_sent);
CREATE INDEX idx_timestamp ON anomaly_logs(timestamp DESC);

-- Create patient_baselines table
CREATE TABLE IF NOT EXISTS patient_baselines (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL UNIQUE,
    
    -- Heart Rate
    hr_mean FLOAT,
    hr_std FLOAT,
    
    -- SpO2
    spo2_mean FLOAT,
    spo2_std FLOAT,
    
    -- Temperature
    temperature_mean FLOAT,
    temperature_std FLOAT,
    
    -- Systolic BP
    sysbp_mean FLOAT,
    sysbp_std FLOAT,
    
    -- Diastolic BP
    diabp_mean FLOAT,
    diabp_std FLOAT,
    
    -- Metadata
    samples_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT pk_patient_baselines PRIMARY KEY (id),
    CONSTRAINT uk_patient_id UNIQUE (patient_id)
);

CREATE INDEX idx_patient_baseline_id ON patient_baselines(patient_id);

-- Create patient_stats table
CREATE TABLE IF NOT EXISTS patient_stats (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL UNIQUE,
    
    -- Anomaly counts
    total_anomalies INTEGER DEFAULT 0,
    high_severity_count INTEGER DEFAULT 0,
    medium_severity_count INTEGER DEFAULT 0,
    low_severity_count INTEGER DEFAULT 0,
    
    -- Anomaly rate
    anomaly_rate FLOAT DEFAULT 0.0,
    
    -- Latest vitals
    last_hr FLOAT,
    last_spo2 FLOAT,
    last_temperature FLOAT,
    last_sysbp FLOAT,
    last_diabp FLOAT,
    last_vital_timestamp TIMESTAMP,
    
    -- Latest anomaly
    latest_severity severity_enum,
    latest_anomaly_timestamp TIMESTAMP,
    
    -- Metadata
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT pk_patient_stats PRIMARY KEY (id),
    CONSTRAINT uk_patient_stats_id UNIQUE (patient_id)
);

CREATE INDEX idx_patient_stats_id ON patient_stats(patient_id);
CREATE INDEX idx_patient_stats_severity ON patient_stats(latest_severity);

-- Create views for easier querying

-- View: Recent anomalies (last 24 hours)
CREATE OR REPLACE VIEW v_recent_anomalies AS
SELECT 
    id,
    patient_id,
    timestamp,
    severity,
    anomaly_score,
    primary_contributor,
    alert_sent
FROM anomaly_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- View: High severity anomalies
CREATE OR REPLACE VIEW v_high_severity_anomalies AS
SELECT 
    id,
    patient_id,
    timestamp,
    anomaly_score,
    primary_contributor,
    explanation
FROM anomaly_logs
WHERE severity = 'HIGH'
ORDER BY timestamp DESC;

-- View: Anomaly statistics by patient
CREATE OR REPLACE VIEW v_anomaly_stats_by_patient AS
SELECT 
    patient_id,
    COUNT(*) as total_anomalies,
    COUNT(*) FILTER (WHERE severity = 'HIGH') as high_count,
    COUNT(*) FILTER (WHERE severity = 'MEDIUM') as medium_count,
    COUNT(*) FILTER (WHERE severity = 'LOW') as low_count,
    COUNT(*) FILTER (WHERE alert_sent = TRUE) as alert_count,
    AVG(anomaly_score) as avg_score,
    MAX(timestamp) as last_anomaly_time
FROM anomaly_logs
GROUP BY patient_id
ORDER BY total_anomalies DESC;

-- Create function to update patient stats
CREATE OR REPLACE FUNCTION update_patient_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO patient_stats (
        patient_id,
        total_anomalies,
        high_severity_count,
        medium_severity_count,
        low_severity_count,
        last_vital_timestamp,
        latest_severity,
        latest_anomaly_timestamp,
        first_seen,
        last_updated
    ) VALUES (
        NEW.patient_id,
        1,
        CASE WHEN NEW.severity = 'HIGH' THEN 1 ELSE 0 END,
        CASE WHEN NEW.severity = 'MEDIUM' THEN 1 ELSE 0 END,
        CASE WHEN NEW.severity = 'LOW' THEN 1 ELSE 0 END,
        NEW.timestamp,
        NEW.severity,
        NEW.timestamp,
        NOW(),
        NOW()
    )
    ON CONFLICT (patient_id) DO UPDATE SET
        total_anomalies = patient_stats.total_anomalies + 1,
        high_severity_count = patient_stats.high_severity_count + 
            CASE WHEN NEW.severity = 'HIGH' THEN 1 ELSE 0 END,
        medium_severity_count = patient_stats.medium_severity_count + 
            CASE WHEN NEW.severity = 'MEDIUM' THEN 1 ELSE 0 END,
        low_severity_count = patient_stats.low_severity_count + 
            CASE WHEN NEW.severity = 'LOW' THEN 1 ELSE 0 END,
        last_vital_timestamp = NEW.timestamp,
        latest_severity = NEW.severity,
        latest_anomaly_timestamp = NEW.timestamp,
        last_updated = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update patient stats
CREATE TRIGGER trg_update_patient_stats
AFTER INSERT ON anomaly_logs
FOR EACH ROW
EXECUTE FUNCTION update_patient_stats();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON anomaly_logs TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON patient_baselines TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON patient_stats TO postgres;
GRANT SELECT ON v_recent_anomalies TO postgres;
GRANT SELECT ON v_high_severity_anomalies TO postgres;
GRANT SELECT ON v_anomaly_stats_by_patient TO postgres;

-- Add comment
COMMENT ON TABLE anomaly_logs IS 'Stores all detected anomalies with details for auditing and analysis';
COMMENT ON TABLE patient_baselines IS 'Stores personalized baseline vitals for each patient';
COMMENT ON TABLE patient_stats IS 'Aggregated statistics per patient for dashboard KPIs';
