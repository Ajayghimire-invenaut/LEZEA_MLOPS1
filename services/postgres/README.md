# PostgreSQL Setup for LeZeA MLOps

This directory contains the PostgreSQL database setup for LeZeA MLOps, providing robust metadata storage for MLflow and advanced analytics capabilities.

## Quick Start

### 1. Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS (with Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download and install from [PostgreSQL.org](https://www.postgresql.org/download/windows/)

### 2. Create Database and User

```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE lezea_mlops;
CREATE USER lezea_user WITH PASSWORD 'lezea_secure_password_2024';
GRANT ALL PRIVILEGES ON DATABASE lezea_mlops TO lezea_user;
\q
```

### 3. Initialize Schema

```bash
# Run the initialization script
sudo -u postgres psql -d lezea_mlops -f init.sql
```

### 4. Verify Installation

```bash
# Test connection
psql -h localhost -U lezea_user -d lezea_mlops -c "SELECT COUNT(*) FROM mlflow.experiments;"
```

## Configuration

### Connection Settings

Update your `.env` file with these PostgreSQL settings:

```env
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lezea_mlops
POSTGRES_USER=lezea_user
POSTGRES_PASSWORD=lezea_secure_password_2024

# MLflow Database URL
MLFLOW_DATABASE_URL=postgresql://lezea_user:lezea_secure_password_2024@localhost:5432/lezea_mlops
```

### Performance Tuning

For production environments, edit `/etc/postgresql/[version]/main/postgresql.conf`:

```ini
# Memory settings
shared_buffers = 256MB                 # 25% of RAM for dedicated server
effective_cache_size = 1GB             # 75% of available RAM
work_mem = 4MB                         # Per-operation memory
maintenance_work_mem = 64MB            # Maintenance operations

# Connection settings
max_connections = 200                  # Adjust based on workload

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9

# Query planner
default_statistics_target = 100        # More detailed statistics
```

## Database Schema Overview

The LeZeA MLOps database consists of three main schemas:

### `mlflow` Schema
Core MLflow tables for experiment tracking:
- `experiments` - Experiment definitions
- `runs` - Individual experiment runs
- `metrics` - Time-series metrics data
- `params` - Hyperparameters
- `tags` - Metadata tags
- `model_versions` - Model registry
- `registered_models` - Model registration

### `analytics` Schema
Advanced analytics and LeZeA-specific features:
- `lezea_experiments` - Enhanced experiment metadata
- `model_performance` - Performance tracking over time
- `resource_metrics` - Resource utilization data
- `dataset_lineage` - Dataset versioning and lineage
- `hyperparameter_optimization` - HPO tracking
- `model_deployments` - Deployment tracking

### `monitoring` Schema
System health and alerting:
- `system_health` - Service status monitoring
- `alert_rules` - Alert configuration
- `alert_history` - Alert event log

## Key Features

### 1. Time-Series Optimization
- Partitioned tables for large metrics datasets
- Optimized indexes for time-based queries
- Automatic cleanup of old data

### 2. Analytics Views
Pre-built views for common analytics queries:
- `experiment_summary` - Quick experiment overview
- `model_performance_trends` - Performance over time
- `resource_utilization_summary` - Resource usage stats
- `latest_model_metrics` - Current model state

### 3. Data Integrity
- Foreign key constraints maintain referential integrity
- Check constraints ensure valid data values
- Triggers for automatic timestamp updates

### 4. Performance Monitoring
- Built-in performance monitoring with `pg_stat_statements`
- Query optimization recommendations
- Index usage tracking

## Common Operations

### Query Experiment Data

```sql
-- Get experiment summary
SELECT * FROM analytics.experiment_summary 
WHERE experiment_name = 'my_experiment';

-- Get latest metrics for all experiments
SELECT experiment_name, metric_name, metric_value 
FROM analytics.latest_model_metrics
WHERE metric_name IN ('accuracy', 'loss');

-- Resource utilization for specific run
SELECT * FROM analytics.resource_utilization_summary
WHERE run_uuid = 'your-run-id';
```

### Maintenance Operations

```sql
-- Clean up old metrics (keep last 90 days)
SELECT cleanup_old_metrics(90);

-- Archive completed experiments (older than 1 year)
SELECT archive_completed_experiments(365);

-- Check database size
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname IN ('mlflow', 'analytics', 'monitoring')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Backup and Restore

```bash
# Create backup
pg_dump -h localhost -U lezea_user -d lezea_mlops > lezea_backup.sql

# Restore from backup
psql -h localhost -U lezea_user -d lezea_mlops < lezea_backup.sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U lezea_user -d lezea_mlops | gzip > "$BACKUP_DIR/lezea_mlops_$DATE.sql.gz"
```

## Security Configuration

### SSL/TLS Setup

Edit `postgresql.conf`:
```ini
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

### Access Control

Edit `pg_hba.conf`:
```
# Local connections
local   lezea_mlops     lezea_user                              md5

# Remote connections (adjust IP range as needed)
host    lezea_mlops     lezea_user      192.168.1.0/24        md5

# SSL-only connections for production
hostssl lezea_mlops     lezea_user      0.0.0.0/0             md5
```

### User Permissions

```sql
-- Create read-only user for analytics
CREATE USER lezea_analyst WITH PASSWORD 'analyst_password';
GRANT USAGE ON SCHEMA mlflow, analytics TO lezea_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA mlflow, analytics TO lezea_analyst;

-- Create monitoring user
CREATE USER lezea_monitor WITH PASSWORD 'monitor_password';
GRANT USAGE ON SCHEMA monitoring TO lezea_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO lezea_monitor;
```

## Monitoring and Alerting

### Performance Monitoring

```sql
-- Check slow queries
SELECT query, total_time, calls, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Monitor database size growth
SELECT 
    datname,
    pg_size_pretty(pg_database_size(datname)) as size
FROM pg_database
WHERE datname = 'lezea_mlops';
```

### Health Checks

```bash
#!/bin/bash
# health_check.sh
DB_STATUS=$(psql -h localhost -U lezea_user -d lezea_mlops -c "SELECT 1;" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "PostgreSQL: HEALTHY"
    exit 0
else
    echo "PostgreSQL: UNHEALTHY"
    exit 1
fi
```

## Troubleshooting

### Common Issues

**Connection refused:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check port and listen addresses
sudo netstat -tlnp | grep :5432
```

**Permission denied:**
```bash
# Check pg_hba.conf configuration
sudo tail /var/log/postgresql/postgresql-*.log
```

**Performance issues:**
```sql
-- Check for table bloat
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
       pg_stat_get_live_tuples(c.oid) as live_tuples,
       pg_stat_get_dead_tuples(c.oid) as dead_tuples
FROM pg_tables t
JOIN pg_class c ON c.relname = t.tablename
WHERE schemaname IN ('mlflow', 'analytics')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Run VACUUM if needed
VACUUM ANALYZE;
```

### Log Analysis

```bash
# Find PostgreSQL log location
sudo -u postgres psql -c "SHOW log_directory;"

# Monitor logs in real-time
sudo tail -f /var/log/postgresql/postgresql-*.log
```

## Integration with LeZeA MLOps

The PostgreSQL backend integrates seamlessly with the LeZeA MLOps system:

```python
from lezea_mlops import ExperimentTracker

# MLOps system automatically uses PostgreSQL for:
tracker = ExperimentTracker("my_experiment")

# 1. MLflow metadata storage
tracker.log_params({"learning_rate": 0.001})
tracker.log_metrics({"accuracy": 0.95})

# 2. Advanced analytics queries
tracker.get_experiment_analytics()

# 3. Resource monitoring
tracker.get_resource_usage_summary()
```

## Next Steps

1. **Set up MLflow**: Configure MLflow to use this PostgreSQL database
2. **Configure Monitoring**: Set up Prometheus metrics collection
3. **Create Dashboards**: Build Grafana dashboards for visualization
4. **Set up Alerts**: Configure alert rules for system monitoring

For more information, see the main LeZeA MLOps documentation.