# Prometheus Setup for LeZeA MLOps

This directory contains the Prometheus monitoring configuration for LeZeA MLOps, providing comprehensive metrics collection, alerting, and observability for the entire ML training pipeline.

## Quick Start

### 1. Install Prometheus

**Linux (Ubuntu/Debian):**
```bash
# Download and install Prometheus
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
sudo mv prometheus-2.45.0.linux-amd64 /opt/prometheus
sudo ln -s /opt/prometheus/prometheus /usr/local/bin/
sudo ln -s /opt/prometheus/promtool /usr/local/bin/

# Create prometheus user
sudo useradd --no-create-home --shell /bin/false prometheus

# Create directories
sudo mkdir -p /etc/prometheus /var/lib/prometheus
sudo chown prometheus:prometheus /etc/prometheus /var/lib/prometheus
```

**macOS (with Homebrew):**
```bash
brew install prometheus
```

**Docker:**
```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/services/prometheus:/etc/prometheus \
  -v prometheus_data:/prometheus \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/etc/prometheus/console_libraries \
  --web.console.templates=/etc/prometheus/consoles \
  --storage.tsdb.retention.time=90d \
  --web.enable-lifecycle
```

### 2. Configure Prometheus

Copy the configuration files:
```bash
# Copy main configuration
sudo cp services/prometheus/prometheus.yml /etc/prometheus/
sudo cp services/prometheus/alerts.yml /etc/prometheus/

# Set permissions
sudo chown prometheus:prometheus /etc/prometheus/*.yml
```

### 3. Install Node Exporter (System Metrics)

```bash
# Download node_exporter
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.0/node_exporter-1.6.0.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.0.linux-amd64.tar.gz
sudo mv node_exporter-1.6.0.linux-amd64/node_exporter /usr/local/bin/

# Create systemd service
sudo cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
```

### 4. Install GPU Exporter (NVIDIA)

```bash
# Install nvidia_gpu_exporter
go install github.com/mindprince/nvidia_gpu_prometheus_exporter@latest

# Or download binary
wget https://github.com/mindprince/nvidia_gpu_prometheus_exporter/releases/download/v1.2.0/nvidia_gpu_prometheus_exporter
chmod +x nvidia_gpu_prometheus_exporter
sudo mv nvidia_gpu_prometheus_exporter /usr/local/bin/

# Create systemd service
sudo cat > /etc/systemd/system/nvidia_gpu_exporter.service << EOF
[Unit]
Description=Nvidia GPU Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/nvidia_gpu_prometheus_exporter

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia_gpu_exporter
sudo systemctl start nvidia_gpu_exporter
```

### 5. Start Prometheus

```bash
# Create systemd service
sudo cat > /etc/systemd/system/prometheus.service << EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.libraries=/etc/prometheus/console_libraries \
    --web.console.templates=/etc/prometheus/consoles \
    --storage.tsdb.retention.time=90d \
    --web.enable-lifecycle

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

### 6. Verify Installation

```bash
# Check Prometheus status
sudo systemctl status prometheus

# Check if Prometheus is accessible
curl http://localhost:9090/api/v1/query?query=up

# Verify targets
curl http://localhost:9090/api/v1/targets
```

## Configuration Overview

### Main Configuration (`prometheus.yml`)

The configuration includes monitoring for:

- **LeZeA MLOps Application** (`localhost:8000`) - Main application metrics
- **System Resources** (`localhost:9100`) - CPU, memory, disk, network via node_exporter
- **GPU Metrics** (`localhost:9835`) - NVIDIA GPU monitoring
- **MLflow Server** (`localhost:5000`) - Experiment tracking metrics
- **PostgreSQL** (`localhost:9187`) - Database performance metrics
- **MongoDB** (`localhost:9216`) - Document store metrics
- **MinIO/S3** (`localhost:9000`) - Object storage metrics

### Alert Rules (`alerts.yml`)

Comprehensive alerting for:

1. **Training Performance**
   - Training stalled or stopped
   - Loss not improving
   - Gradient explosion
   - Low throughput
   - Model accuracy degradation

2. **Resource Utilization**
   - High GPU/CPU utilization
   - Memory exhaustion
   - Disk space running low
   - GPU overheating

3. **Service Health**
   - Service downtime
   - Slow response times
   - High error rates
   - Database connection issues

4. **Data Pipeline**
   - Slow data loading
   - Data quality issues
   - Anomalous dataset changes

5. **Capacity Planning**
   - Predicted resource exhaustion
   - Training completion delays

## Metrics Collection

### LeZeA MLOps Metrics

The LeZeA MLOps system exposes these key metrics:

```prometheus
# Training metrics
lezea_training_steps_total{experiment_id, model_type}
lezea_training_epochs_total{experiment_id, model_type}
lezea_step_duration_seconds{experiment_id, model_type}
lezea_current_loss{experiment_id, model_type, loss_type}
lezea_gradient_norm{experiment_id, model_type}

# Resource metrics  
lezea_gpu_utilization_percent{gpu_id, gpu_name}
lezea_gpu_memory_used_bytes{gpu_id, gpu_name}
lezea_gpu_temperature_celsius{gpu_id, gpu_name}
lezea_cpu_utilization_percent
lezea_memory_used_bytes

# Model metrics
lezea_model_accuracy{experiment_id, model_type, dataset}
lezea_inference_duration_seconds{experiment_id, model_type}

# Service metrics
lezea_service_up{service_name, service_type}
lezea_errors_total{error_type, component}
```

### System Metrics (Node Exporter)

Standard system monitoring:
```prometheus
# CPU
node_cpu_seconds_total
node_load1, node_load5, node_load15

# Memory
node_memory_MemTotal_bytes
node_memory_MemAvailable_bytes
node_memory_Buffers_bytes

# Disk
node_filesystem_size_bytes
node_filesystem_avail_bytes
node_disk_io_time_seconds_total

# Network
node_network_receive_bytes_total
node_network_transmit_bytes_total
```

### GPU Metrics (NVIDIA)

GPU monitoring:
```prometheus
nvidia_gpu_utilization_percent
nvidia_gpu_memory_used_megabytes
nvidia_gpu_memory_total_megabytes
nvidia_gpu_temperature_celsius
nvidia_gpu_power_draw_watts
```

## Query Examples

### Training Performance

```promql
# Current training throughput
rate(lezea_training_steps_total[5m])

# Average step duration over time
rate(lezea_step_duration_seconds_sum[5m]) / rate(lezea_step_duration_seconds_count[5m])

# Loss improvement rate
rate(lezea_current_loss[1h])

# GPU utilization during training
avg_over_time(lezea_gpu_utilization_percent[10m])
```

### Resource Analysis

```promql
# GPU memory usage percentage
(lezea_gpu_memory_used_bytes / lezea_gpu_memory_total_bytes) * 100

# CPU utilization trend
avg_over_time(lezea_cpu_utilization_percent[1h])

# Disk space remaining
(lezea_disk_total_bytes - lezea_disk_used_bytes) / lezea_disk_total_bytes * 100

# Network throughput
rate(lezea_network_bytes_sent_total[5m]) + rate(lezea_network_bytes_received_total[5m])
```

### Service Health

```promql
# Service uptime
avg_over_time(lezea_service_up[24h])

# Error rate by component
rate(lezea_errors_total[5m])

# Response time percentiles
histogram_quantile(0.95, rate(lezea_service_response_seconds_bucket[5m]))
```

## Alerting Setup

### Alertmanager Integration

1. Install Alertmanager:
```bash
wget https://github.com/prometheus/alertmanager/releases/download/v0.25.0/alertmanager-0.25.0.linux-amd64.tar.gz
tar xvfz alertmanager-0.25.0.linux-amd64.tar.gz
sudo mv alertmanager-0.25.0.linux-amd64 /opt/alertmanager
sudo ln -s /opt/alertmanager/alertmanager /usr/local/bin/
```

2. Configure Alertmanager (`/etc/alertmanager/alertmanager.yml`):
```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@lezea-mlops.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'admin@lezea-mlops.com'
        subject: 'LeZeA MLOps Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#lezea-alerts'
        title: 'LeZeA MLOps Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

3. Enable alerting in Prometheus (`prometheus.yml`):
```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - "localhost:9093"
```

### Notification Channels

**Email Notifications:**
```yaml
email_configs:
  - to: 'ml-team@company.com'
    from: 'prometheus@lezea-mlops.com'
    smarthost: 'smtp.gmail.com:587'
    auth_username: 'alerts@company.com'
    auth_password: 'app-password'
```

**Slack Integration:**
```yaml
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ml-alerts'
    username: 'Prometheus'
    color: 'danger'
    title: 'Training Alert: {{ .GroupLabels.alertname }}'
```

**PagerDuty Integration:**
```yaml
pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
    description: 'LeZeA MLOps Alert'
```

## Performance Tuning

### Storage Configuration

```yaml
# prometheus.yml
storage:
  tsdb:
    retention.time: 90d
    retention.size: 50GB
    wal-compression: true
    min-block-duration: 2h
    max-block-duration: 25h
```

### Query Optimization

```yaml
# Limit query resources
query:
  max-samples: 50000000
  timeout: 2m
  max-concurrency: 20
```

### Recording Rules

Create recording rules for expensive queries (`recording_rules.yml`):
```yaml
groups:
  - name: lezea_recording_rules
    interval: 30s
    rules:
      - record: lezea:training_throughput_5m
        expr: rate(lezea_training_steps_total[5m])
      
      - record: lezea:gpu_memory_utilization
        expr: (lezea_gpu_memory_used_bytes / lezea_gpu_memory_total_bytes) * 100
      
      - record: lezea:error_rate_5m
        expr: rate(lezea_errors_total[5m])
```

## Monitoring Best Practices

### 1. Metric Naming

Follow Prometheus naming conventions:
- Use base unit (seconds, bytes, total)
- Include unit suffix (_seconds, _bytes, _total)
- Use snake_case
- Start with application prefix (lezea_)

### 2. Label Usage

- Keep cardinality low (< 10,000 series per metric)
- Use labels for dimensions you want to aggregate by
- Avoid high-cardinality labels (user IDs, timestamps)

### 3. Alert Design

- Make alerts actionable
- Include runbook links
- Set appropriate severity levels
- Use reasonable thresholds and durations

### 4. Data Retention

- Configure appropriate retention (90 days default)
- Use recording rules for long-term queries
- Consider remote storage for historical data

## Grafana Integration

Connect Prometheus to Grafana:

1. Add Prometheus data source in Grafana:
   - URL: `http://localhost:9090`
   - Access: Server (default)

2. Import LeZeA MLOps dashboards:
   - Training Overview Dashboard
   - Resource Monitoring Dashboard
   - Service Health Dashboard

## Backup and Recovery

### Configuration Backup

```bash
#!/bin/bash
# backup_prometheus.sh
BACKUP_DIR="/backup/prometheus"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configuration
tar -czf "$BACKUP_DIR/prometheus_config_$DATE.tar.gz" /etc/prometheus/

# Backup data (if needed)
tar -czf "$BACKUP_DIR/prometheus_data_$DATE.tar.gz" /var/lib/prometheus/
```

### Data Recovery

```bash
# Stop Prometheus
sudo systemctl stop prometheus

# Restore configuration
tar -xzf prometheus_config_backup.tar.gz -C /

# Restore data
tar -xzf prometheus_data_backup.tar.gz -C /

# Start Prometheus
sudo systemctl start prometheus
```

## Troubleshooting

### Common Issues

**High memory usage:**
```bash
# Check memory usage
ps aux | grep prometheus

# Reduce retention or sample frequency
# Edit prometheus.yml:
storage:
  tsdb:
    retention.time: 30d  # Reduce retention
```

**Slow queries:**
```bash
# Check slow queries in Prometheus UI
# Go to Status -> Runtime & Build Information
# Look at query duration metrics

# Use recording rules for expensive queries
```

**Target discovery issues:**
```bash
# Check target status
curl http://localhost:9090/api/v1/targets

# Verify network connectivity
telnet target_host target_port

# Check Prometheus logs
sudo journalctl -u prometheus -f
```

**Disk space issues:**
```bash
# Check Prometheus data size
du -sh /var/lib/prometheus/

# Clean old data
prometheus_data_cleaner --data.dir=/var/lib/prometheus
```

## Integration with LeZeA MLOps

The Prometheus setup integrates seamlessly with LeZeA MLOps:

```python
from lezea_mlops import ExperimentTracker
from lezea_mlops.monitoring import get_metrics

# Initialize metrics collection
metrics = get_metrics()
metrics.start_metrics_server(port=8000)

# Use ExperimentTracker - metrics are automatically collected
tracker = ExperimentTracker("my_experiment")
tracker.log_params({"learning_rate": 0.001})
tracker.log_metrics({"accuracy": 0.95})

# Metrics are now available in Prometheus!
```

## Next Steps

1. **Set up Grafana**: Create visualization dashboards
2. **Configure Alertmanager**: Set up notification channels
3. **Implement recording rules**: Optimize query performance
4. **Set up federation**: Scale monitoring across multiple instances
5. **Configure remote storage**: Set up long-term metrics storage

For more information, see the [Prometheus documentation](https://prometheus.io/docs/) and the main LeZeA MLOps documentation.