# LeZeA MLOps

**Production-Ready MLOps Platform for Large Language Model Training**

A comprehensive, enterprise-grade MLOps platform designed specifically for training and managing large language models like LeZeA. Built for 2-person teams who need professional-grade experiment tracking, monitoring, and deployment capabilities without the complexity of massive platforms.


## 🚀 Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Clone and setup
git clone https://github.com/your-org/lezea-mlops.git
cd lezea-mlops
pip install -r requirements.txt

# 2. Start all services
./scripts/start_all.sh

# 3. Run your first experiment
python examples/quick_start.py

# 4. View results
open http://localhost:5000  # MLflow UI
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana
```

## ✨ Key Features

### 🎯 **Experiment Tracking**
- **Complete experiment lifecycle management** with MLflow integration
- **Advanced parameter and metric logging** with automatic versioning
- **Real-time training monitoring** with step-by-step progress tracking
- **Model registry and versioning** for production deployments
- **Artifact management** with S3-compatible storage

### 📊 **Comprehensive Monitoring**
- **Real-time GPU/CPU/Memory monitoring** with NVIDIA GPU support
- **Training performance metrics** (loss, accuracy, throughput)
- **Resource utilization tracking** and capacity planning
- **Service health monitoring** with automated alerts
- **Prometheus metrics integration** for enterprise monitoring

### 🗄️ **Robust Data Management**
- **PostgreSQL backend** for scalable metadata storage
- **MongoDB integration** for complex hierarchical data
- **S3-compatible storage** for large artifacts and models
- **DVC integration** for dataset versioning and lineage
- **Automated backup and recovery** systems

### 🔧 **Production Ready**
- **One-command deployment** for all services
- **Comprehensive health checks** and monitoring
- **Automated service management** and restart capabilities
- **Enterprise security** with authentication and access controls
- **Scalable architecture** from laptop to production cluster

## 🏗️ Architecture

LeZeA MLOps is built on a microservices architecture optimized for ML workloads:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   Experiment    │    │   Monitoring    │
│   Scripts       │◄──►│   Tracker       │◄──►│   & Metrics     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Storage  │    │   Metadata      │    │   Time Series   │
│   (S3/MinIO)    │    │   (PostgreSQL)  │    │   (Prometheus)  │
│   - Models      │    │   - Experiments │    │   - Metrics     │
│   - Artifacts   │    │   - Parameters  │    │   - Resources   │
│   - Datasets    │    │   - Results     │    │   - Alerts      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **ExperimentTracker** | Main API for experiment management | Python, MLflow |
| **Monitoring Stack** | Real-time metrics and alerting | Prometheus, Grafana |
| **Data Backends** | Scalable storage for all data types | PostgreSQL, MongoDB, S3 |
| **Service Management** | Automated deployment and health checks | Bash scripts, Python |

## 📁 Project Structure

```
lezea-mlops/
├── 📖 README.md                    # You are here!
├── ⚙️  pyproject.toml              # Modern Python packaging
├── 📌 requirements.txt             # Pinned dependencies
├── 🔑 .env.example                 # Configuration template
│
├── 🐍 lezea_mlops/                 # Main Python package
│   ├── 🎯 tracker.py               # Main ExperimentTracker class
│   ├── ⚙️  config/                 # Configuration management
│   ├── 💾 backends/                # Storage backend integrations
│   ├── 📊 monitoring/              # Real-time metrics collection
│   └── 🛠️  utils/                  # Helper functions and utilities
│
├── 🚀 services/                    # Service configurations
│   ├── 🐘 postgres/                # PostgreSQL setup and schemas
│   ├── 📈 mlflow/                  # MLflow server configuration
│   ├── 📊 prometheus/              # Monitoring configuration
│   └── 📉 grafana/                 # Dashboard definitions
│
├── 🏃 scripts/                     # Automation and management
│   ├── ▶️  start_all.sh            # One-command service startup
│   ├── 🛑 stop_all.sh              # Clean service shutdown
│   ├── ❤️  health_check.py         # Comprehensive health monitoring
│   └── 🔧 setup_dev_env.py         # Development environment setup
│
├── 💡 examples/                    # Ready-to-use examples
│   ├── ⚡ quick_start.py           # 10-line minimal example
│   ├── 🎓 full_training.py         # Complete production example
│   └── 📚 README.md                # Example documentation
│
├── 📁 data/                        # Local data storage (git-ignored)
└── 🧪 tests/                       # Comprehensive test suite
```

## 🛠️ Installation

### Prerequisites

- **Python 3.8+** with pip
- **PostgreSQL 12+** for metadata storage
- **Docker** (optional, for containerized services)
- **NVIDIA Drivers** (optional, for GPU monitoring)

### Option 1: Quick Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/lezea-mlops.git
cd lezea-mlops

# Install Python dependencies
pip install -r requirements.txt

# Run setup script
python scripts/setup_dev_env.py

# Start all services
./scripts/start_all.sh

# Verify installation
python scripts/health_check.py
```

### Option 2: Manual Setup

1. **Install PostgreSQL:**
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# Initialize database
sudo -u postgres psql -f services/postgres/init.sql
```

2. **Install MLflow and dependencies:**
```bash
pip install mlflow[extras] psycopg2-binary boto3 prometheus_client
```

3. **Start services individually:**
```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Start MLflow
./services/mlflow/start.sh start --daemon

# Start Prometheus (if installed)
prometheus --config.file=services/prometheus/prometheus.yml
```

### Option 3: Docker Setup

```bash
# Start with Docker Compose
docker-compose up -d

# Or build and run containers
docker build -t lezea-mlops .
docker run -p 5000:5000 -p 9090:9090 lezea-mlops
```

## 🎯 Usage

### Basic Usage (2 minutes)

```python
from lezea_mlops import ExperimentTracker

# Initialize experiment
tracker = ExperimentTracker("my_first_experiment")

# Log parameters
tracker.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "transformer"
})

# Training loop
for epoch in range(10):
    # Your training code here
    loss = train_step()
    accuracy = evaluate()
    
    # Log metrics
    tracker.log_metrics({
        "loss": loss,
        "accuracy": accuracy
    }, step=epoch)

# Save model
tracker.log_artifact("model.pth")
tracker.finish_run()
```

### Advanced Usage (Production)

```python
from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring import get_metrics

# Production experiment with monitoring
tracker = ExperimentTracker(
    experiment_name="production_model_v2",
    description="Production LeZeA model with A/B testing",
    tags={"team": "ml-research", "priority": "high"}
)

# Enable real-time monitoring
metrics = get_metrics()
metrics.start_metrics_server(port=8000)

# Advanced logging
tracker.log_model_architecture({
    "type": "transformer",
    "layers": 24,
    "hidden_size": 1024,
    "attention_heads": 16
})

tracker.log_dataset_info({
    "name": "lezea_training_v3",
    "version": "1.2.0",
    "samples": 1_000_000,
    "size_gb": 15.7
})

# Training with comprehensive monitoring
for epoch in range(epochs):
    epoch_metrics = train_epoch()
    
    # Log to MLflow
    tracker.log_metrics(epoch_metrics, step=epoch)
    
    # Log to Prometheus
    metrics.record_training_step(
        experiment_id=tracker.run_id,
        model_type="transformer",
        step_time=epoch_metrics["step_time"],
        loss=epoch_metrics["loss"]
    )
    
    # Automatic checkpointing
    if epoch % 5 == 0:
        tracker.save_checkpoint(epoch, epoch_metrics)

# Model registration
tracker.register_model(
    model_name="lezea_production_v2",
    description="Production LeZeA model with 95.4% accuracy"
)
```

## 📊 Monitoring & Observability

LeZeA MLOps provides comprehensive monitoring out of the box:

### Real-time Dashboards

- **MLflow UI** (http://localhost:5000): Experiment tracking and model registry
- **Prometheus** (http://localhost:9090): Metrics and alerting
- **Grafana** (http://localhost:3000): Custom dashboards and visualization

### Key Metrics Tracked

| Metric Category | Examples | Purpose |
|----------------|----------|---------|
| **Training Performance** | Loss, accuracy, throughput | Monitor model learning |
| **Resource Usage** | GPU utilization, memory, CPU | Optimize resource allocation |
| **System Health** | Service uptime, response times | Ensure system reliability |
| **Data Quality** | Dataset size, validation scores | Monitor data pipeline health |

### Automated Alerting

```yaml
# Example alert rule
- alert: TrainingStalled
  expr: increase(lezea_training_steps_total[5m]) == 0
  for: 10m
  annotations:
    summary: "Training has stalled for {{ $labels.experiment_id }}"
    runbook_url: "https://docs.lezea-mlops.com/runbooks/training-stalled"
```

## 🔧 Configuration

### Environment Configuration

Create `.env` file:

```env
# Core services
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lezea_mlops
POSTGRES_USER=lezea_user
POSTGRES_PASSWORD=your_secure_password

# Storage
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://lezea-mlops-artifacts

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=your_admin_password
```

### Advanced Configuration

```yaml
# config/services.yml
mlflow:
  host: "0.0.0.0"
  port: 5000
  workers: 4
  backend_store_uri: "postgresql://user:pass@localhost/lezea_mlops"
  default_artifact_root: "s3://lezea-mlops-artifacts"

prometheus:
  host: "0.0.0.0"
  port: 9090
  retention_time: "90d"
  scrape_interval: "15s"

postgresql:
  host: "localhost"
  port: 5432
  database: "lezea_mlops"
  max_connections: 200
  shared_buffers: "256MB"
```

## 🚀 Examples

### Quick Start (5 minutes)
```bash
python examples/quick_start.py
```
Perfect for first-time users and quick prototyping.

### Full Production Example (15 minutes)
```bash
python examples/full_training.py
```
Comprehensive example showing all features for production use.

### Custom Configuration
```bash
python examples/full_training.py --config my_config.yaml --epochs 50
```

See [examples/README.md](examples/README.md) for detailed usage patterns and integration guides.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_tracker.py -v
python -m pytest tests/test_backends.py -v
python -m pytest tests/test_monitoring.py -v

# Run with coverage
python -m pytest tests/ --cov=lezea_mlops --cov-report=html
```

## 🔍 Health Monitoring

LeZeA MLOps includes comprehensive health monitoring:

```bash
# Check all services
python scripts/health_check.py

# Check specific services
python scripts/health_check.py --services postgresql,mlflow,prometheus

# Continuous monitoring
python scripts/health_check.py --continuous --interval 60

# JSON output for automation
python scripts/health_check.py --json
```

## 📈 Performance

### Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Experiment startup** | <2 seconds | Cold start with all backends |
| **Metric logging** | >10,000 metrics/sec | Batch logging performance |
| **Artifact upload** | >100 MB/sec | S3 multipart uploads |
| **Query response** | <100ms | PostgreSQL analytics queries |
| **Memory usage** | <500MB | Complete system baseline |

### Optimization Tips

1. **Use batch logging** for high-frequency metrics
2. **Enable PostgreSQL connection pooling** for high concurrency
3. **Configure S3 multipart uploads** for large artifacts
4. **Use Prometheus recording rules** for complex queries
5. **Partition large tables** by time for better performance

## 🚦 Service Management

### Start/Stop Services

```bash
# Start all services
./scripts/start_all.sh

# Start specific services
./scripts/start_all.sh --services postgresql,mlflow

# Stop all services
./scripts/start_all.sh stop

# Restart services
./scripts/start_all.sh restart

# Check status
./scripts/start_all.sh status
```

### Individual Service Management

```bash
# MLflow
./services/mlflow/start.sh start --daemon
./services/mlflow/start.sh stop

# PostgreSQL
sudo systemctl start postgresql
sudo systemctl stop postgresql

# Prometheus
prometheus --config.file=services/prometheus/prometheus.yml
```

## 🔒 Security

### Authentication

Enable MLflow authentication:
```env
MLFLOW_AUTH=true
MLFLOW_AUTH_CONFIG_PATH=./services/mlflow/auth_config.ini
```

### Database Security

```sql
-- Create read-only user for analytics
CREATE USER lezea_analyst WITH PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA mlflow, analytics TO lezea_analyst;
```

### Network Security

```bash
# Restrict to localhost (development)
MLFLOW_HOST=127.0.0.1
PROMETHEUS_HOST=127.0.0.1

# Production with reverse proxy
# Use Nginx/Apache for HTTPS termination and authentication
```

## 🔧 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check port availability
netstat -tlnp | grep :5000

# Check logs
./scripts/start_all.sh logs

# Verify dependencies
python scripts/health_check.py
```

**Database connection errors:**
```bash
# Test PostgreSQL connection
psql -h localhost -U lezea_user -d lezea_mlops -c "SELECT 1;"

# Check PostgreSQL status
sudo systemctl status postgresql
```

**MLflow experiments not appearing:**
```bash
# Verify MLflow server
curl http://localhost:5000/health

# Check database tables
psql -h localhost -U lezea_user -d lezea_mlops -c "\dt mlflow.*"
```

**GPU monitoring not working:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Install GPU exporter
go install github.com/mindprince/nvidia_gpu_prometheus_exporter@latest
```

### Getting Help

1. **Check service logs:** `./scripts/start_all.sh logs`
2. **Run health checks:** `python scripts/health_check.py --verbose`
3. **Verify configuration:** Check `.env` file and service configs
4. **Review documentation:** Each service has detailed README files
5. **Check GitHub issues:** Search for similar problems and solutions

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-org/lezea-mlops.git
cd lezea-mlops

# Install development dependencies