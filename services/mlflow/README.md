# MLflow Setup for LeZeA MLOps

This directory contains the MLflow server configuration and startup scripts for LeZeA MLOps, providing comprehensive experiment tracking and model registry capabilities.

## Quick Start

### 1. Install MLflow

```bash
# Install MLflow with all dependencies
pip install mlflow[extras]

# For production deployment
pip install gunicorn psycopg2-binary boto3
```

### 2. Configure Environment

Create or update your `.env` file:

```env
# MLflow Server Configuration
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
MLFLOW_WORKERS=4

# Database Configuration (PostgreSQL)
MLFLOW_BACKEND_STORE_URI=postgresql://lezea_user:lezea_secure_password_2024@localhost:5432/lezea_mlops

# Artifact Storage (S3)
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://lezea-mlops-artifacts
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Optional: S3-compatible storage (MinIO, etc.)
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Security
MLFLOW_AUTH=false
MLFLOW_AUTH_CONFIG_PATH=./services/mlflow/auth_config.ini
```

### 3. Start MLflow Server

```bash
# Start in foreground (for development)
./services/mlflow/start.sh start

# Start as daemon (for production)
./services/mlflow/start.sh start --daemon

# Start with custom configuration
./services/mlflow/start.sh start --host 0.0.0.0 --port 5001 --workers 8
```

### 4. Verify Installation

```bash
# Check server status
./services/mlflow/start.sh status

# Perform health check
./services/mlflow/start.sh health

# View logs
./services/mlflow/start.sh logs
```

## Server Management

### Starting the Server

```bash
# Development mode (foreground)
./start.sh start

# Production mode (background daemon)
./start.sh start --daemon

# Custom configuration
./start.sh start --host 127.0.0.1 --port 5001 --workers 8 --daemon
```

### Managing the Server

```bash
# Check status
./start.sh status

# Stop server
./start.sh stop

# Restart server
./start.sh restart

# View logs
./start.sh logs

# Health check
./start.sh health
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_HOST` | Server bind address | `0.0.0.0` |
| `MLFLOW_PORT` | Server port | `5000` |
| `MLFLOW_WORKERS` | Number of worker processes | `4` |
| `MLFLOW_BACKEND_STORE_URI` | Database connection string | PostgreSQL URI |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | Artifact storage location | S3 bucket |
| `MLFLOW_AUTH` | Enable authentication | `false` |
| `MLFLOW_LOG_LEVEL` | Logging level | `INFO` |

### Database Configuration

MLflow supports multiple database backends:

**PostgreSQL (Recommended):**
```
MLFLOW_BACKEND_STORE_URI=postgresql://user:password@host:port/database
```

**MySQL:**
```
MLFLOW_BACKEND_STORE_URI=mysql://user:password@host:port/database
```

**SQLite (Development only):**
```
MLFLOW_BACKEND_STORE_URI=sqlite:///path/to/mlflow.db
```

### Artifact Storage

**S3 (Recommended for production):**
```env
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://your-bucket-name/mlflow-artifacts
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

**Local filesystem (Development):**
```env
MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///path/to/artifacts
```

**Azure Blob Storage:**
```env
MLFLOW_DEFAULT_ARTIFACT_ROOT=azure://container@account.blob.core.windows.net/path
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

**Google Cloud Storage:**
```env
MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://your-bucket-name/mlflow-artifacts
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Production Deployment

### Performance Tuning

For production environments, configure these settings:

```env
# Server performance
MLFLOW_WORKERS=8                    # 2x CPU cores
MLFLOW_HOST=0.0.0.0                # Accept external connections

# Database connection pooling
MLFLOW_SQLALCHEMY_ENGINE_OPTIONS={"pool_size": 20, "max_overflow": 0, "pool_pre_ping": true, "pool_recycle": 3600}

# Artifact upload/download timeout
MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT=600

# Gunicorn configuration (when using multiple workers)
GUNICORN_CMD_ARGS="--timeout 120 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100"
```

### Security Configuration

#### Authentication Setup

1. Enable authentication:
```env
MLFLOW_AUTH=true
MLFLOW_AUTH_CONFIG_PATH=./services/mlflow/auth_config.ini
```

2. Create `auth_config.ini`:
```ini
[mlflow]
default_permission = READ
database_uri = postgresql://lezea_user:password@localhost/lezea_mlops_auth
admin_username = admin
admin_password = secure_admin_password

[users]
marcus = password123, EDIT
analyst = analyst_pass, READ
```

#### SSL/TLS Configuration

For HTTPS deployment, use a reverse proxy like Nginx:

```nginx
server {
    listen 443 ssl;
    server_name mlflow.your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring and Logging

### Log Configuration

MLflow logs are written to `logs/mlflow_server.log`. Configure log rotation:

```bash
# Create logrotate configuration
sudo cat > /etc/logrotate.d/mlflow << EOF
/path/to/lezea-mlops/logs/mlflow_server.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### Health Monitoring

The startup script includes built-in health checks:

```bash
# Manual health check
curl -f http://localhost:5000/health

# Automated monitoring script
#!/bin/bash
if ! ./services/mlflow/start.sh health; then
    echo "MLflow server unhealthy, restarting..."
    ./services/mlflow/start.sh restart
fi
```

### Prometheus Metrics

MLflow server exposes metrics at `/metrics` endpoint when configured:

```python
# Enable metrics collection
import mlflow.server.handlers

# Metrics available:
# - mlflow_request_duration_seconds
# - mlflow_request_total
# - mlflow_active_experiments
# - mlflow_active_runs
```

## Backup and Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# Backup MLflow database
BACKUP_DIR="/backup/mlflow"
DATE=$(date +%Y%m%d_%H%M%S)

# Create database backup
pg_dump -h localhost -U lezea_user lezea_mlops > "$BACKUP_DIR/mlflow_db_$DATE.sql"

# Backup artifacts (if using local storage)
rsync -av /path/to/artifacts/ "$BACKUP_DIR/artifacts_$DATE/"
```

### Restore Procedure

```bash
# Restore database
psql -h localhost -U lezea_user -d lezea_mlops < backup_file.sql

# Restore artifacts
rsync -av backup_artifacts/ /path/to/artifacts/
```

## Integration with LeZeA MLOps

The MLflow server integrates seamlessly with the LeZeA MLOps tracker:

```python
from lezea_mlops import ExperimentTracker

# Automatically connects to your MLflow server
tracker = ExperimentTracker("my_experiment")

# All operations use your MLflow server
tracker.log_params({"learning_rate": 0.001})
tracker.log_metrics({"accuracy": 0.95})
tracker.log_artifacts("./model_outputs/")
```

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using the port
lsof -i :5000
# or
netstat -tlnp | grep :5000

# Kill the process
kill -9 <PID>
```

**Database connection errors:**
```bash
# Test database connectivity
psql -h localhost -U lezea_user -d lezea_mlops -c "SELECT 1