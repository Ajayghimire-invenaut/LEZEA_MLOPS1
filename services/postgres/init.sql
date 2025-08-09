-- LeZeA MLOps PostgreSQL Database Initialization
-- ===============================================
--
-- This script sets up the complete database schema for LeZeA MLOps:
-- 1. MLflow metadata storage (experiments, runs, metrics, parameters)
-- 2. Custom analytics tables for advanced querying
-- 3. Performance optimization (indexes, partitioning)
-- 4. Security and user management
-- 5. Monitoring and maintenance views

-- Create database and user (run as postgres superuser)
-- Note: Uncomment these lines if running as superuser
-- CREATE DATABASE lezea_mlops;
-- CREATE USER lezea_user WITH PASSWORD 'lezea_secure_password_2024';
-- GRANT ALL PRIVILEGES ON DATABASE lezea_mlops TO lezea_user;

-- Connect to the lezea_mlops database
\c lezea_mlops;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS mlflow;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set default schema search path
ALTER DATABASE lezea_mlops SET search_path TO mlflow, analytics, monitoring, public;

-- ========================================
-- MLflow Core Tables
-- ========================================

-- Experiments table
CREATE TABLE IF NOT EXISTS mlflow.experiments (
    experiment_id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL UNIQUE,
    artifact_location VARCHAR(1000),
    lifecycle_stage VARCHAR(32) DEFAULT 'active',
    creation_time BIGINT,
    last_update_time BIGINT,
    tags TEXT,
    CONSTRAINT valid_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted'))
);

-- Runs table (core experiment runs)
CREATE TABLE IF NOT EXISTS mlflow.runs (
    run_uuid VARCHAR(32) PRIMARY KEY,
    name VARCHAR(250),
    experiment_id INTEGER NOT NULL,
    user_id VARCHAR(256),
    status VARCHAR(20) DEFAULT 'RUNNING',
    start_time BIGINT,
    end_time BIGINT,
    source_type VARCHAR(20),
    source_name VARCHAR(500),
    entry_point_name VARCHAR(50),
    source_version VARCHAR(50),
    lifecycle_stage VARCHAR(20) DEFAULT 'active',
    artifact_uri VARCHAR(200),
    creation_time BIGINT NOT NULL,
    last_update_time BIGINT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES mlflow.experiments(experiment_id),
    CONSTRAINT valid_status CHECK (status IN ('SCHEDULED', 'RUNNING', 'FINISHED', 'FAILED', 'KILLED')),
    CONSTRAINT valid_source_type CHECK (source_type IN ('NOTEBOOK', 'JOB', 'LOCAL', 'UNKNOWN', 'PROJECT')),
    CONSTRAINT valid_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted'))
);

-- Metrics table (time-series metrics)
CREATE TABLE IF NOT EXISTS mlflow.metrics (
    key VARCHAR(250) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp BIGINT NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    step BIGINT DEFAULT 0,
    is_nan BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (key, timestamp, run_uuid, step),
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Parameters table (hyperparameters)
CREATE TABLE IF NOT EXISTS mlflow.params (
    key VARCHAR(250) NOT NULL,
    value TEXT NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    PRIMARY KEY (key, run_uuid),
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Tags table (experiment and run metadata)
CREATE TABLE IF NOT EXISTS mlflow.tags (
    key VARCHAR(250) NOT NULL,
    value TEXT,
    run_uuid VARCHAR(32) NOT NULL,
    PRIMARY KEY (key, run_uuid),
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Model registry tables
CREATE TABLE IF NOT EXISTS mlflow.model_versions (
    name VARCHAR(256) NOT NULL,
    version INTEGER NOT NULL,
    creation_time BIGINT,
    last_updated_time BIGINT,
    description VARCHAR(5000),
    user_id VARCHAR(256),
    current_stage VARCHAR(20) DEFAULT 'None',
    source VARCHAR(500),
    run_id VARCHAR(32),
    run_link VARCHAR(500),
    status VARCHAR(20) DEFAULT 'READY',
    status_message VARCHAR(500),
    storage_location VARCHAR(500),
    PRIMARY KEY (name, version),
    CONSTRAINT valid_stage CHECK (current_stage IN ('None', 'Staging', 'Production', 'Archived')),
    CONSTRAINT valid_status CHECK (status IN ('PENDING_REGISTRATION', 'FAILED_REGISTRATION', 'READY'))
);

CREATE TABLE IF NOT EXISTS mlflow.registered_models (
    name VARCHAR(256) PRIMARY KEY,
    creation_time BIGINT,
    last_updated_time BIGINT,
    description VARCHAR(5000)
);

-- ========================================
-- LeZeA-Specific Analytics Tables
-- ========================================

-- Enhanced experiment tracking with LeZeA-specific fields
CREATE TABLE IF NOT EXISTS analytics.lezea_experiments (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    model_type VARCHAR(100),
    dataset_name VARCHAR(200),
    dataset_version VARCHAR(50),
    architecture_config JSONB,
    training_config JSONB,
    resource_usage JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (experiment_id) REFERENCES mlflow.experiments(experiment_id),
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Model performance tracking over time
CREATE TABLE IF NOT EXISTS analytics.model_performance (
    id SERIAL PRIMARY KEY,
    run_uuid VARCHAR(32) NOT NULL,
    evaluation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_type VARCHAR(50), -- 'train', 'validation', 'test'
    metric_name VARCHAR(100),
    metric_value DOUBLE PRECISION,
    step BIGINT,
    epoch INTEGER,
    additional_metadata JSONB,
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Resource utilization tracking
CREATE TABLE IF NOT EXISTS analytics.resource_metrics (
    id SERIAL PRIMARY KEY,
    run_uuid VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    gpu_utilization DOUBLE PRECISION,
    gpu_memory_used BIGINT,
    gpu_memory_total BIGINT,
    cpu_utilization DOUBLE PRECISION,
    ram_used BIGINT,
    ram_total BIGINT,
    disk_io_read BIGINT,
    disk_io_write BIGINT,
    network_io_send BIGINT,
    network_io_recv BIGINT,
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Dataset lineage and versioning
CREATE TABLE IF NOT EXISTS analytics.dataset_lineage (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(200) NOT NULL,
    version VARCHAR(50) NOT NULL,
    parent_version VARCHAR(50),
    creation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    creator_user VARCHAR(256),
    size_bytes BIGINT,
    sample_count BIGINT,
    checksum VARCHAR(128),
    storage_location VARCHAR(500),
    transformation_config JSONB,
    quality_metrics JSONB,
    UNIQUE(dataset_name, version)
);

-- Hyperparameter optimization tracking
CREATE TABLE IF NOT EXISTS analytics.hyperparameter_optimization (
    id SERIAL PRIMARY KEY,
    optimization_id VARCHAR(100) NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    iteration INTEGER NOT NULL,
    hyperparameters JSONB NOT NULL,
    objective_value DOUBLE PRECISION,
    objective_name VARCHAR(100),
    optimization_algorithm VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (run_uuid) REFERENCES mlflow.runs(run_uuid) ON DELETE CASCADE
);

-- Model deployment tracking
CREATE TABLE IF NOT EXISTS analytics.model_deployments (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(256) NOT NULL,
    model_version INTEGER NOT NULL,
    environment VARCHAR(50), -- 'staging', 'production', 'testing'
    deployment_status VARCHAR(20) DEFAULT 'active',
    endpoint_url VARCHAR(500),
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_health_check TIMESTAMP WITH TIME ZONE,
    health_status VARCHAR(20),
    deployment_config JSONB,
    FOREIGN KEY (model_name, model_version) REFERENCES mlflow.model_versions(name, version),
    CONSTRAINT valid_deployment_status CHECK (deployment_status IN ('active', 'inactive', 'failed', 'retired')),
    CONSTRAINT valid_health_status CHECK (health_status IN ('healthy', 'unhealthy', 'unknown'))
);

-- ========================================
-- Monitoring and Alerting Tables
-- ========================================

-- System health monitoring
CREATE TABLE IF NOT EXISTS monitoring.system_health (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    service_name VARCHAR(100) NOT NULL,
    service_type VARCHAR(50), -- 'mlflow', 'mongodb', 's3', 'prometheus'
    status VARCHAR(20) DEFAULT 'unknown',
    response_time_ms DOUBLE PRECISION,
    error_message TEXT,
    additional_metrics JSONB,
    CONSTRAINT valid_status CHECK (status IN ('healthy', 'unhealthy', 'degraded', 'unknown'))
);

-- Alert rules and notifications
CREATE TABLE IF NOT EXISTS monitoring.alert_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(200) UNIQUE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    condition_type VARCHAR(20), -- 'greater_than', 'less_than', 'equals'
    threshold_value DOUBLE PRECISION,
    duration_minutes INTEGER DEFAULT 5,
    severity VARCHAR(20) DEFAULT 'warning',
    enabled BOOLEAN DEFAULT TRUE,
    notification_channels TEXT[], -- JSON array of notification channels
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_triggered TIMESTAMP WITH TIME ZONE,
    CONSTRAINT valid_condition CHECK (condition_type IN ('greater_than', 'less_than', 'equals', 'not_equals')),
    CONSTRAINT valid_severity CHECK (severity IN ('info', 'warning', 'error', 'critical'))
);

-- Alert history
CREATE TABLE IF NOT EXISTS monitoring.alert_history (
    id SERIAL PRIMARY KEY,
    rule_id INTEGER NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    metric_value DOUBLE PRECISION,
    alert_message TEXT,
    notification_sent BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (rule_id) REFERENCES monitoring.alert_rules(id)
);

-- ========================================
-- Partitioning for Large Tables
-- ========================================

-- Partition metrics table by time (monthly partitions)
-- This is commented out by default - uncomment if you expect high volume
/*
CREATE TABLE mlflow.metrics_y2024m01 PARTITION OF mlflow.metrics
    FOR VALUES FROM ('2024-01-01 00:00:00'::timestamp) TO ('2024-02-01 00:00:00'::timestamp);

CREATE TABLE mlflow.metrics_y2024m02 PARTITION OF mlflow.metrics  
    FOR VALUES FROM ('2024-02-01 00:00:00'::timestamp) TO ('2024-03-01 00:00:00'::timestamp);
*/

-- ========================================
-- Indexes for Performance Optimization
-- ========================================

-- MLflow core indexes
CREATE INDEX IF NOT EXISTS idx_experiments_name ON mlflow.experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_lifecycle ON mlflow.experiments(lifecycle_stage);

CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON mlflow.runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON mlflow.runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_start_time ON mlflow.runs(start_time);
CREATE INDEX IF NOT EXISTS idx_runs_user_id ON mlflow.runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_lifecycle ON mlflow.runs(lifecycle_stage);

CREATE INDEX IF NOT EXISTS idx_metrics_run_uuid ON mlflow.metrics(run_uuid);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON mlflow.metrics(key);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON mlflow.metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON mlflow.metrics(step);

CREATE INDEX IF NOT EXISTS idx_params_run_uuid ON mlflow.params(run_uuid);
CREATE INDEX IF NOT EXISTS idx_params_key ON mlflow.params(key);

CREATE INDEX IF NOT EXISTS idx_tags_run_uuid ON mlflow.tags(run_uuid);
CREATE INDEX IF NOT EXISTS idx_tags_key ON mlflow.tags(key);

-- Model registry indexes
CREATE INDEX IF NOT EXISTS idx_model_versions_name ON mlflow.model_versions(name);
CREATE INDEX IF NOT EXISTS idx_model_versions_stage ON mlflow.model_versions(current_stage);
CREATE INDEX IF NOT EXISTS idx_model_versions_run_id ON mlflow.model_versions(run_id);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_lezea_experiments_model_type ON analytics.lezea_experiments(model_type);
CREATE INDEX IF NOT EXISTS idx_lezea_experiments_dataset ON analytics.lezea_experiments(dataset_name);
CREATE INDEX IF NOT EXISTS idx_lezea_experiments_created_at ON analytics.lezea_experiments(created_at);

CREATE INDEX IF NOT EXISTS idx_model_performance_run_uuid ON analytics.model_performance(run_uuid);
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON analytics.model_performance(evaluation_timestamp);
CREATE INDEX IF NOT EXISTS idx_model_performance_metric ON analytics.model_performance(metric_name);

CREATE INDEX IF NOT EXISTS idx_resource_metrics_run_uuid ON analytics.resource_metrics(run_uuid);
CREATE INDEX IF NOT EXISTS idx_resource_metrics_timestamp ON analytics.resource_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_dataset_lineage_name_version ON analytics.dataset_lineage(dataset_name, version);
CREATE INDEX IF NOT EXISTS idx_dataset_lineage_creation ON analytics.dataset_lineage(creation_timestamp);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON monitoring.system_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_health_service ON monitoring.system_health(service_name);
CREATE INDEX IF NOT EXISTS idx_alert_history_triggered ON monitoring.alert_history(triggered_at);

-- ========================================
-- Useful Views for Analytics
-- ========================================

-- Experiment summary view
CREATE OR REPLACE VIEW analytics.experiment_summary AS
SELECT 
    e.experiment_id,
    e.name as experiment_name,
    r.run_uuid,
    r.status,
    r.start_time,
    r.end_time,
    CASE 
        WHEN r.end_time IS NOT NULL AND r.start_time IS NOT NULL 
        THEN (r.end_time - r.start_time) / 1000.0 
        ELSE NULL 
    END as duration_seconds,
    le.model_type,
    le.dataset_name,
    le.dataset_version,
    (
        SELECT COUNT(*) 
        FROM mlflow.metrics m 
        WHERE m.run_uuid = r.run_uuid
    ) as metric_count,
    (
        SELECT COUNT(*) 
        FROM mlflow.params p 
        WHERE p.run_uuid = r.run_uuid  
    ) as param_count
FROM mlflow.experiments e
JOIN mlflow.runs r ON e.experiment_id = r.experiment_id
LEFT JOIN analytics.lezea_experiments le ON r.run_uuid = le.run_uuid
WHERE e.lifecycle_stage = 'active' AND r.lifecycle_stage = 'active';

-- Model performance trends view
CREATE OR REPLACE VIEW analytics.model_performance_trends AS
SELECT 
    mp.run_uuid,
    r.experiment_id,
    e.name as experiment_name,
    le.model_type,
    mp.metric_name,
    mp.metric_value,
    mp.step,
    mp.epoch,
    mp.evaluation_timestamp,
    ROW_NUMBER() OVER (
        PARTITION BY mp.run_uuid, mp.metric_name 
        ORDER BY mp.evaluation_timestamp DESC
    ) as recency_rank
FROM analytics.model_performance mp
JOIN mlflow.runs r ON mp.run_uuid = r.run_uuid
JOIN mlflow.experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN analytics.lezea_experiments le ON r.run_uuid = le.run_uuid
WHERE r.lifecycle_stage = 'active';

-- Resource utilization summary view
CREATE OR REPLACE VIEW analytics.resource_utilization_summary AS
SELECT 
    rm.run_uuid,
    r.experiment_id,
    e.name as experiment_name,
    AVG(rm.gpu_utilization) as avg_gpu_utilization,
    MAX(rm.gpu_utilization) as max_gpu_utilization,
    AVG(rm.cpu_utilization) as avg_cpu_utilization,
    MAX(rm.gpu_memory_used) as peak_gpu_memory,
    MAX(rm.ram_used) as peak_ram_usage,
    COUNT(*) as measurement_count,
    MIN(rm.timestamp) as monitoring_start,
    MAX(rm.timestamp) as monitoring_end
FROM analytics.resource_metrics rm
JOIN mlflow.runs r ON rm.run_uuid = r.run_uuid
JOIN mlflow.experiments e ON r.experiment_id = e.experiment_id
WHERE r.lifecycle_stage = 'active'
GROUP BY rm.run_uuid, r.experiment_id, e.name;

-- Latest model metrics view
CREATE OR REPLACE VIEW analytics.latest_model_metrics AS
SELECT DISTINCT ON (m.run_uuid, m.key)
    m.run_uuid,
    r.experiment_id,
    e.name as experiment_name,
    m.key as metric_name,
    m.value as metric_value,
    m.step,
    m.timestamp,
    le.model_type,
    le.dataset_name
FROM mlflow.metrics m
JOIN mlflow.runs r ON m.run_uuid = r.run_uuid
JOIN mlflow.experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN analytics.lezea_experiments le ON r.run_uuid = le.run_uuid
WHERE r.lifecycle_stage = 'active'
ORDER BY m.run_uuid, m.key, m.timestamp DESC;

-- ========================================
-- Triggers for Automatic Updates
-- ========================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

-- Trigger for lezea_experiments table
CREATE TRIGGER update_lezea_experiments_updated_at 
    BEFORE UPDATE ON analytics.lezea_experiments 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- Sample Alert Rules
-- ========================================

-- Insert default alert rules
INSERT INTO monitoring.alert_rules (
    rule_name, 
    metric_name, 
    condition_type, 
    threshold_value, 
    duration_minutes, 
    severity,
    notification_channels
) VALUES
('High GPU Memory Usage', 'gpu_memory_utilization', 'greater_than', 0.95, 5, 'warning', ARRAY['email', 'slack']),
('Training Stuck', 'training_progress', 'less_than', 0.01, 30, 'error', ARRAY['email', 'slack']),
('Model Performance Degradation', 'validation_accuracy', 'less_than', 0.8, 10, 'warning', ARRAY['email']),
('Service Down', 'service_availability', 'less_than', 1.0, 2, 'critical', ARRAY['email', 'slack', 'pagerduty']),
('Disk Space Low', 'disk_usage_percent', 'greater_than', 0.9, 15, 'warning', ARRAY['email'])
ON CONFLICT (rule_name) DO NOTHING;

-- ========================================
-- Database Maintenance Functions
-- ========================================

-- Function to clean up old metrics (keep last 90 days)
CREATE OR REPLACE FUNCTION cleanup_old_metrics(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $
DECLARE
    cutoff_timestamp BIGINT;
    deleted_count INTEGER;
BEGIN
    -- Convert days to milliseconds timestamp
    cutoff_timestamp := EXTRACT(EPOCH FROM (NOW() - INTERVAL '1 day' * days_to_keep)) * 1000;
    
    DELETE FROM mlflow.metrics WHERE timestamp < cutoff_timestamp;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$ LANGUAGE plpgsql;

-- Function to archive completed experiments
CREATE OR REPLACE FUNCTION archive_completed_experiments(days_old INTEGER DEFAULT 365)
RETURNS INTEGER AS $
DECLARE
    cutoff_timestamp BIGINT;
    archived_count INTEGER;
BEGIN
    cutoff_timestamp := EXTRACT(EPOCH FROM (NOW() - INTERVAL '1 day' * days_old)) * 1000;
    
    UPDATE mlflow.runs 
    SET lifecycle_stage = 'deleted'
    WHERE status IN ('FINISHED', 'FAILED', 'KILLED') 
      AND end_time < cutoff_timestamp
      AND lifecycle_stage = 'active';
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    
    RETURN archived_count;
END;
$ LANGUAGE plpgsql;

-- ========================================
-- Grant Permissions
-- ========================================

-- Grant permissions to lezea_user
GRANT USAGE ON SCHEMA mlflow TO lezea_user;
GRANT USAGE ON SCHEMA analytics TO lezea_user;
GRANT USAGE ON SCHEMA monitoring TO lezea_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mlflow TO lezea_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO lezea_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO lezea_user;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA mlflow TO lezea_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO lezea_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO lezea_user;

-- Grant permissions on views
GRANT SELECT ON analytics.experiment_summary TO lezea_user;
GRANT SELECT ON analytics.model_performance_trends TO lezea_user;
GRANT SELECT ON analytics.resource_utilization_summary TO lezea_user;
GRANT SELECT ON analytics.latest_model_metrics TO lezea_user;

-- ========================================
-- Performance Tuning Settings
-- ========================================

-- Optimize for analytics workload
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Note: Restart PostgreSQL service for system settings to take effect

-- ========================================
-- Database Setup Complete
-- ========================================

-- Display setup summary
DO $
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    view_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count FROM information_schema.tables 
    WHERE table_schema IN ('mlflow', 'analytics', 'monitoring');
    
    SELECT COUNT(*) INTO index_count FROM pg_indexes 
    WHERE schemaname IN ('mlflow', 'analytics', 'monitoring');
    
    SELECT COUNT(*) INTO view_count FROM information_schema.views 
    WHERE table_schema IN ('analytics');
    
    RAISE NOTICE 'LeZeA MLOps Database Setup Complete!';
    RAISE NOTICE 'Created % tables, % indexes, % views', table_count, index_count, view_count;
    RAISE NOTICE 'Database is ready for MLflow and LeZeA MLOps operations.';
END $;