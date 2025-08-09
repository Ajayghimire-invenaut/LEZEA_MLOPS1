#!/bin/bash
# LeZeA MLOps - Complete System Startup Script
# =============================================
#
# This script starts all services in the correct order:
# 1. Infrastructure services (PostgreSQL, MongoDB)
# 2. Storage services (MinIO/S3)
# 3. MLflow server
# 4. Monitoring stack (Prometheus, Node Exporter, GPU Exporter)
# 5. Health checks and verification

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Configuration
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$LOG_DIR/pids"
SERVICES_DIR="$PROJECT_ROOT/services"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Service status tracking
declare -A SERVICE_STATUS
declare -A SERVICE_PIDS
declare -A SERVICE_PORTS

# Default ports
SERVICE_PORTS=(
    ["postgresql"]="5432"
    ["mongodb"]="27017" 
    ["minio"]="9000"
    ["mlflow"]="5000"
    ["prometheus"]="9090"
    ["node_exporter"]="9100"
    ["gpu_exporter"]="9835"
    ["grafana"]="3000"
)

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_service() {
    echo -e "${PURPLE}[SERVICE]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_check() {
    echo -e "${CYAN}[CHECK]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [COMMAND]

Complete LeZeA MLOps System Management

Commands:
    start       Start all services (default)
    stop        Stop all services
    restart     Restart all services
    status      Show status of all services
    health      Perform health checks
    logs        Show logs for all services

Options:
    --quick         Skip dependency checks and start faster
    --no-monitoring Skip monitoring services (Prometheus, Grafana)
    --no-storage    Skip storage services (MinIO)
    --services LIST Comma-separated list of services to start
    --wait-time N   Seconds to wait between service starts (default: 5)
    --verbose       Enable verbose output
    --help          Show this help message

Services:
    postgresql      PostgreSQL database
    mongodb         MongoDB document store
    minio           MinIO object storage (optional)
    mlflow          MLflow experiment tracking server
    prometheus      Prometheus monitoring
    node_exporter   System metrics exporter
    gpu_exporter    GPU metrics exporter (if NVIDIA GPUs present)
    grafana         Grafana dashboards (optional)

Examples:
    $0 start                                # Start all services
    $0 start --quick                        # Fast start, skip checks
    $0 start --services postgresql,mlflow   # Start specific services
    $0 restart --no-monitoring              # Restart without monitoring
    $0 health                              # Check all service health

Environment Variables:
    LEZEA_QUICK_START=true     Skip dependency checks
    LEZEA_WAIT_TIME=10         Custom wait time between services
    LEZEA_SERVICES="list"      Comma-separated services to manage

EOF
}

# Function to create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "$LOG_DIR" "$PID_DIR"
    mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/artifacts"
    mkdir -p "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/models"
    
    # Set permissions
    chmod 755 "$LOG_DIR" "$PID_DIR"
    chmod 755 "$PROJECT_ROOT/data" "$PROJECT_ROOT/artifacts"
    
    log_success "Directories created successfully"
}

# Function to check if port is available
check_port() {
    local port=$1
    local service=$2
    
    if command -v netstat &> /dev/null; then
        if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            log_warn "Port $port is already in use (may be $service running)"
            return 1
        fi
    elif command -v ss &> /dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            log_warn "Port $port is already in use (may be $service running)"
            return 1
        fi
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    log_check "Waiting for $service to be ready on port $port..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if command -v nc &> /dev/null; then
            if nc -z localhost "$port" 2>/dev/null; then
                log_success "$service is ready on port $port"
                return 0
            fi
        elif command -v curl &> /dev/null; then
            if curl -s --connect-timeout 2 "http://localhost:$port" >/dev/null 2>&1; then
                log_success "$service is ready on port $port"
                return 0
            fi
        elif command -v telnet &> /dev/null; then
            if timeout 2 telnet localhost "$port" </dev/null >/dev/null 2>&1; then
                log_success "$service is ready on port $port"
                return 0
            fi
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo
    log_error "$service failed to start on port $port after $((max_attempts * 2)) seconds"
    return 1
}

# Function to check service health
check_service_health() {
    local service=$1
    local port=${SERVICE_PORTS[$service]}
    
    case $service in
        "postgresql")
            if command -v psql &> /dev/null; then
                if PGPASSWORD="${POSTGRES_PASSWORD:-lezea_secure_password_2024}" \
                   psql -h localhost -p "$port" -U "${POSTGRES_USER:-lezea_user}" \
                   -d "${POSTGRES_DB:-lezea_mlops}" -c "SELECT 1;" &>/dev/null; then
                    return 0
                fi
            fi
            ;;
        "mongodb")
            if command -v mongosh &> /dev/null || command -v mongo &> /dev/null; then
                local mongo_cmd="mongosh"
                if ! command -v mongosh &> /dev/null; then
                    mongo_cmd="mongo"
                fi
                if $mongo_cmd --host localhost:$port --eval "db.runCommand('ping')" &>/dev/null; then
                    return 0
                fi
            fi
            ;;
        "mlflow"|"prometheus"|"minio"|"grafana"|"node_exporter"|"gpu_exporter")
            if command -v curl &> /dev/null; then
                local health_path=""
                case $service in
                    "mlflow") health_path="/health" ;;
                    "prometheus") health_path="/-/healthy" ;;
                    "minio") health_path="/minio/health/live" ;;
                    "grafana") health_path="/api/health" ;;
                    *) health_path="/metrics" ;;
                esac
                
                if curl -s --connect-timeout 5 "http://localhost:$port$health_path" >/dev/null; then
                    return 0
                fi
            fi
            ;;
    esac
    
    # Fallback: check if port is responding
    if command -v nc &> /dev/null; then
        if nc -z localhost "$port" 2>/dev/null; then
            return 0
        fi
    fi
    
    return 1
}

# Function to start PostgreSQL
start_postgresql() {
    log_service "Starting PostgreSQL..."
    
    # Check if PostgreSQL is already running
    if check_service_health postgresql; then
        log_warn "PostgreSQL is already running"
        SERVICE_STATUS["postgresql"]="running"
        return 0
    fi
    
    # Try to start PostgreSQL using system service
    if command -v systemctl &> /dev/null; then
        if systemctl is-enabled postgresql &>/dev/null || systemctl is-enabled postgresql@* &>/dev/null; then
            log_info "Starting PostgreSQL via systemctl..."
            if sudo systemctl start postgresql; then
                wait_for_service postgresql 5432 15
                if check_service_health postgresql; then
                    SERVICE_STATUS["postgresql"]="running"
                    log_success "PostgreSQL started successfully"
                    return 0
                fi
            fi
        fi
    fi
    
    # Try Homebrew on macOS
    if command -v brew &> /dev/null; then
        if brew services list | grep postgresql | grep -q stopped; then
            log_info "Starting PostgreSQL via Homebrew..."
            if brew services start postgresql; then
                wait_for_service postgresql 5432 15
                if check_service_health postgresql; then
                    SERVICE_STATUS["postgresql"]="running"
                    log_success "PostgreSQL started successfully"
                    return 0
                fi
            fi
        fi
    fi
    
    # Manual start if available
    if command -v pg_ctl &> /dev/null; then
        log_info "Starting PostgreSQL manually..."
        local data_dir="${PGDATA:-/var/lib/postgresql/data}"
        if [[ -d "$data_dir" ]]; then
            pg_ctl -D "$data_dir" -l "$LOG_DIR/postgresql.log" start
            wait_for_service postgresql 5432 15
            if check_service_health postgresql; then
                SERVICE_STATUS["postgresql"]="running"
                log_success "PostgreSQL started successfully"
                return 0
            fi
        fi
    fi
    
    log_error "Failed to start PostgreSQL"
    log_error "Please ensure PostgreSQL is installed and configured"
    SERVICE_STATUS["postgresql"]="failed"
    return 1
}

# Function to start MongoDB
start_mongodb() {
    log_service "Starting MongoDB..."
    
    # Check if MongoDB is already running
    if check_service_health mongodb; then
        log_warn "MongoDB is already running"
        SERVICE_STATUS["mongodb"]="running"
        return 0
    fi
    
    # Try to start MongoDB using system service
    if command -v systemctl &> /dev/null; then
        if systemctl is-enabled mongod &>/dev/null; then
            log_info "Starting MongoDB via systemctl..."
            if sudo systemctl start mongod; then
                wait_for_service mongodb 27017 15
                if check_service_health mongodb; then
                    SERVICE_STATUS["mongodb"]="running"
                    log_success "MongoDB started successfully"
                    return 0
                fi
            fi
        fi
    fi
    
    # Try Homebrew on macOS
    if command -v brew &> /dev/null; then
        if brew services list | grep mongodb | grep -q stopped; then
            log_info "Starting MongoDB via Homebrew..."
            if brew services start mongodb-community; then
                wait_for_service mongodb 27017 15
                if check_service_health mongodb; then
                    SERVICE_STATUS["mongodb"]="running"
                    log_success "MongoDB started successfully"
                    return 0
                fi
            fi
        fi
    fi
    
    # Manual start
    if command -v mongod &> /dev/null; then
        log_info "Starting MongoDB manually..."
        local data_dir="${MONGODB_DATA_DIR:-$PROJECT_ROOT/data/mongodb}"
        mkdir -p "$data_dir"
        
        nohup mongod --dbpath "$data_dir" --logpath "$LOG_DIR/mongodb.log" \
                     --port 27017 --fork &> /dev/null || true
        
        wait_for_service mongodb 27017 15
        if check_service_health mongodb; then
            SERVICE_STATUS["mongodb"]="running"
            log_success "MongoDB started successfully"
            return 0
        fi
    fi
    
    log_error "Failed to start MongoDB"
    log_error "Please ensure MongoDB is installed and configured"
    SERVICE_STATUS["mongodb"]="failed"
    return 1
}

# Function to start MinIO
start_minio() {
    log_service "Starting MinIO..."
    
    # Check if MinIO is already running
    if check_service_health minio; then
        log_warn "MinIO is already running"
        SERVICE_STATUS["minio"]="running"
        return 0
    fi
    
    # Check if MinIO is installed
    if ! command -v minio &> /dev/null; then
        log_warn "MinIO not found, skipping..."
        SERVICE_STATUS["minio"]="skipped"
        return 0
    fi
    
    log_info "Starting MinIO server..."
    local data_dir="${MINIO_DATA_DIR:-$PROJECT_ROOT/data/minio}"
    mkdir -p "$data_dir"
    
    # Set MinIO credentials
    export MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
    export MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin123}"
    
    # Start MinIO
    nohup minio server "$data_dir" \
        --address ":9000" \
        --console-address ":9001" \
        > "$LOG_DIR/minio.log" 2>&1 & echo $! > "$PID_DIR/minio.pid"
    
    wait_for_service minio 9000 20
    if check_service_health minio; then
        SERVICE_STATUS["minio"]="running"
        SERVICE_PIDS["minio"]=$(cat "$PID_DIR/minio.pid")
        log_success "MinIO started successfully"
        log_info "MinIO Console: http://localhost:9001"
        return 0
    else
        log_error "Failed to start MinIO"
        SERVICE_STATUS["minio"]="failed"
        return 1
    fi
}

# Function to start MLflow
start_mlflow() {
    log_service "Starting MLflow..."
    
    # Check if MLflow is already running
    if check_service_health mlflow; then
        log_warn "MLflow is already running"
        SERVICE_STATUS["mlflow"]="running"
        return 0
    fi
    
    # Check if MLflow script exists
    local mlflow_script="$SERVICES_DIR/mlflow/start.sh"
    if [[ ! -x "$mlflow_script" ]]; then
        log_error "MLflow start script not found or not executable: $mlflow_script"
        SERVICE_STATUS["mlflow"]="failed"
        return 1
    fi
    
    log_info "Starting MLflow server..."
    if "$mlflow_script" start --daemon; then
        wait_for_service mlflow 5000 30
        if check_service_health mlflow; then
            SERVICE_STATUS["mlflow"]="running"
            log_success "MLflow started successfully"
            log_info "MLflow UI: http://localhost:5000"
            return 0
        fi
    fi
    
    log_error "Failed to start MLflow"
    SERVICE_STATUS["mlflow"]="failed"
    return 1
}

# Function to start Node Exporter
start_node_exporter() {
    log_service "Starting Node Exporter..."
    
    # Check if Node Exporter is already running
    if check_service_health node_exporter; then
        log_warn "Node Exporter is already running"
        SERVICE_STATUS["node_exporter"]="running"
        return 0
    fi
    
    # Check if Node Exporter is installed
    if ! command -v node_exporter &> /dev/null; then
        log_warn "Node Exporter not found, skipping system metrics..."
        SERVICE_STATUS["node_exporter"]="skipped"
        return 0
    fi
    
    log_info "Starting Node Exporter..."
    nohup node_exporter \
        --web.listen-address=":9100" \
        --log.level=info \
        > "$LOG_DIR/node_exporter.log" 2>&1 & echo $! > "$PID_DIR/node_exporter.pid"
    
    wait_for_service node_exporter 9100 10
    if check_service_health node_exporter; then
        SERVICE_STATUS["node_exporter"]="running"
        SERVICE_PIDS["node_exporter"]=$(cat "$PID_DIR/node_exporter.pid")
        log_success "Node Exporter started successfully"
        return 0
    else
        log_error "Failed to start Node Exporter"
        SERVICE_STATUS["node_exporter"]="failed"
        return 1
    fi
}

# Function to start GPU Exporter
start_gpu_exporter() {
    log_service "Starting GPU Exporter..."
    
    # Check if NVIDIA GPUs are present
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "NVIDIA drivers not found, skipping GPU monitoring..."
        SERVICE_STATUS["gpu_exporter"]="skipped"
        return 0
    fi
    
    # Check if GPU Exporter is already running
    if check_service_health gpu_exporter; then
        log_warn "GPU Exporter is already running"
        SERVICE_STATUS["gpu_exporter"]="running"
        return 0
    fi
    
    # Check if GPU Exporter is installed
    if ! command -v nvidia_gpu_prometheus_exporter &> /dev/null; then
        log_warn "GPU Exporter not found, skipping GPU metrics..."
        SERVICE_STATUS["gpu_exporter"]="skipped"
        return 0
    fi
    
    log_info "Starting GPU Exporter..."
    nohup nvidia_gpu_prometheus_exporter \
        --web.listen-address=":9835" \
        > "$LOG_DIR/gpu_exporter.log" 2>&1 & echo $! > "$PID_DIR/gpu_exporter.pid"
    
    wait_for_service gpu_exporter 9835 10
    if check_service_health gpu_exporter; then
        SERVICE_STATUS["gpu_exporter"]="running"
        SERVICE_PIDS["gpu_exporter"]=$(cat "$PID_DIR/gpu_exporter.pid")
        log_success "GPU Exporter started successfully"
        return 0
    else
        log_error "Failed to start GPU Exporter"
        SERVICE_STATUS["gpu_exporter"]="failed"
        return 1
    fi
}

# Function to start Prometheus
start_prometheus() {
    log_service "Starting Prometheus..."
    
    # Check if Prometheus is already running
    if check_service_health prometheus; then
        log_warn "Prometheus is already running"
        SERVICE_STATUS["prometheus"]="running"
        return 0
    fi
    
    # Check if Prometheus is installed
    if ! command -v prometheus &> /dev/null; then
        log_warn "Prometheus not found, skipping monitoring..."
        SERVICE_STATUS["prometheus"]="skipped"
        return 0
    fi
    
    # Check if config file exists
    local config_file="$SERVICES_DIR/prometheus/prometheus.yml"
    if [[ ! -f "$config_file" ]]; then
        log_error "Prometheus config not found: $config_file"
        SERVICE_STATUS["prometheus"]="failed"
        return 1
    fi
    
    log_info "Starting Prometheus..."
    local data_dir="$PROJECT_ROOT/data/prometheus"
    mkdir -p "$data_dir"
    
    nohup prometheus \
        --config.file="$config_file" \
        --storage.tsdb.path="$data_dir" \
        --web.console.libraries=/etc/prometheus/console_libraries \
        --web.console.templates=/etc/prometheus/consoles \
        --storage.tsdb.retention.time=90d \
        --web.enable-lifecycle \
        --web.listen-address=":9090" \
        > "$LOG_DIR/prometheus.log" 2>&1 & echo $! > "$PID_DIR/prometheus.pid"
    
    wait_for_service prometheus 9090 20
    if check_service_health prometheus; then
        SERVICE_STATUS["prometheus"]="running"
        SERVICE_PIDS["prometheus"]=$(cat "$PID_DIR/prometheus.pid")
        log_success "Prometheus started successfully"
        log_info "Prometheus UI: http://localhost:9090"
        return 0
    else
        log_error "Failed to start Prometheus"
        SERVICE_STATUS["prometheus"]="failed"
        return 1
    fi
}

# Function to start Grafana
start_grafana() {
    log_service "Starting Grafana..."
    
    # Check if Grafana is already running
    if check_service_health grafana; then
        log_warn "Grafana is already running"
        SERVICE_STATUS["grafana"]="running"
        return 0
    fi
    
    # Check if Grafana is installed
    if ! command -v grafana-server &> /dev/null; then
        log_warn "Grafana not found, skipping dashboards..."
        SERVICE_STATUS["grafana"]="skipped"
        return 0
    fi
    
    log_info "Starting Grafana..."
    local data_dir="$PROJECT_ROOT/data/grafana"
    mkdir -p "$data_dir"
    
    # Set Grafana environment
    export GF_PATHS_DATA="$data_dir"
    export GF_PATHS_LOGS="$LOG_DIR"
    export GF_SERVER_HTTP_PORT="3000"
    export GF_SECURITY_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin123}"
    
    nohup grafana-server \
        --homepath=/usr/share/grafana \
        --config=/etc/grafana/grafana.ini \
        > "$LOG_DIR/grafana.log" 2>&1 & echo $! > "$PID_DIR/grafana.pid"
    
    wait_for_service grafana 3000 20
    if check_service_health grafana; then
        SERVICE_STATUS["grafana"]="running"
        SERVICE_PIDS["grafana"]=$(cat "$PID_DIR/grafana.pid")
        log_success "Grafana started successfully"
        log_info "Grafana UI: http://localhost:3000 (admin/admin123)"
        return 0
    else
        log_error "Failed to start Grafana"
        SERVICE_STATUS["grafana"]="failed"
        return 1
    fi
}

# Function to stop a service
stop_service() {
    local service=$1
    
    log_service "Stopping $service..."
    
    case $service in
        "postgresql")
            if command -v systemctl &> /dev/null; then
                sudo systemctl stop postgresql 2>/dev/null || true
            elif command -v brew &> /dev/null; then
                brew services stop postgresql 2>/dev/null || true
            elif command -v pg_ctl &> /dev/null; then
                pg_ctl stop -D "${PGDATA:-/var/lib/postgresql/data}" 2>/dev/null || true
            fi
            ;;
        "mongodb")
            if command -v systemctl &> /dev/null; then
                sudo systemctl stop mongod 2>/dev/null || true
            elif command -v brew &> /dev/null; then
                brew services stop mongodb-community 2>/dev/null || true
            fi
            ;;
        "mlflow")
            local mlflow_script="$SERVICES_DIR/mlflow/start.sh"
            if [[ -x "$mlflow_script" ]]; then
                "$mlflow_script" stop 2>/dev/null || true
            fi
            ;;
        *)
            # Stop services started by this script
            if [[ -f "$PID_DIR/$service.pid" ]]; then
                local pid=$(cat "$PID_DIR/$service.pid")
                if kill -0 "$pid" 2>/dev/null; then
                    kill "$pid" 2>/dev/null || true
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -9 "$pid" 2>/dev/null || true
                    fi
                fi
                rm -f "$PID_DIR/$service.pid"
            fi
            
            # Also try to kill by process name
            pkill -f "$service" 2>/dev/null || true
            ;;
    esac
    
    log_success "$service stopped"
}

# Function to show service status
show_status() {
    local service=$1
    local status="unknown"
    local pid=""
    local color="$NC"
    
    if check_service_health "$service"; then
        status="running"
        color="$GREEN"
    else
        status="stopped"
        color="$RED"
    fi
    
    if [[ -f "$PID_DIR/$service.pid" ]]; then
        pid=" (PID: $(cat "$PID_DIR/$service.pid"))"
    fi
    
    printf "%-15s: ${color}%-8s${NC}%s\n" "$service" "$status" "$pid"
}

# Function to perform comprehensive health check
health_check() {
    log_info "Performing comprehensive health check..."
    
    local all_healthy=true
    local services=("postgresql" "mongodb" "minio" "mlflow" "prometheus" "node_exporter" "gpu_exporter" "grafana")
    
    echo
    echo "Service Health Status:"
    echo "====================="
    
    for service in "${services[@]}"; do
        if check_service_health "$service"; then
            printf "%-15s: ${GREEN}✓ HEALTHY${NC}\n" "$service"
        else
            printf "%-15s: ${RED}✗ UNHEALTHY${NC}\n" "$service"
            all_healthy=false
        fi
    done
    
    echo
    echo "System Resources:"
    echo "================"
    
    # CPU usage
    if command -v top &> /dev/null; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        printf "CPU Usage:       %.1f%%\n" "$cpu_usage"
    fi
    
    # Memory usage
    if command -v free &> /dev/null; then
        local mem_info=$(free -h | awk 'NR==2{printf "Memory Usage:    %s/%s (%.1f%%)", $3,$2,$3*100/$2}')
        echo "$mem_info"
    fi
    
    # Disk usage
    local disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2{printf "Disk Usage:      %s/%s (%s)", $3,$2,$5}')
    echo "$disk_usage"
    
    # GPU status (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo
        echo "GPU Status:"
        echo "==========="
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader,nounits | \
        while IFS=, read -r name util mem_used mem_total temp; do
            printf "%-20s: %2s%% utilization, %s/%s MB memory, %s°C\n" \
                   "$name" "$util" "$mem_used" "$mem_total" "$temp"
        done
    fi
    
    echo
    if $all_healthy; then
        log_success "All services are healthy!"
        return 0
    else
        log_warn "Some services are unhealthy. Check individual service logs."
        return 1
    fi
}

# Function to start all services
start_all_services() {
    local services=("$@")
    local wait_time="${LEZEA_WAIT_TIME:-5}"
    
    if [[ ${#services[@]} -eq 0 ]]; then
        services=("postgresql" "mongodb" "minio" "mlflow" "node_exporter" "gpu_exporter" "prometheus" "grafana")
    fi
    
    log_info "Starting LeZeA MLOps services..."
    log_info "Services to start: ${services[*]}"
    
    # Start infrastructure services first
    for service in "${services[@]}"; do
        case $service in
            "postgresql"|"mongodb"|"minio")
                start_$service
                if [[ $? -eq 0 ]]; then
                    sleep "$wait_time"
                fi
                ;;
        esac
    done
    
    # Start application services
    for service in "${services[@]}"; do
        case $service in
            "mlflow")
                start_$service
                if [[ $? -eq 0 ]]; then
                    sleep "$wait_time"
                fi
                ;;
        esac
    done
    
    # Start monitoring services
    for service in "${services[@]}"; do
        case $service in
            "node_exporter"|"gpu_exporter"|"prometheus"|"grafana")
                start_$service
                if [[ $? -eq 0 ]]; then
                    sleep "$wait_time"
                fi
                ;;
        esac
    done
    
    echo
    log_info "Startup complete! Service status:"
    echo "=================================="
    
    for service in "${services[@]}"; do
        show_status "$service"
    done
    
    echo
    log_info "Access URLs:"
    echo "============"
    echo "MLflow UI:       http://localhost:5000"
    echo "Prometheus UI:   http://localhost:9090"
    echo "Grafana UI:      http://localhost:3000 (admin/admin123)"
    echo "MinIO Console:   http://localhost:9001 (minioadmin/minioadmin123)"
    echo
    echo "Log files location: $LOG_DIR"
    echo "PID files location: $PID_DIR"
}

# Function to stop all services
stop_all_services() {
    local services=("grafana" "prometheus" "gpu_exporter" "node_exporter" "mlflow" "minio" "mongodb" "postgresql")
    
    log_info "Stopping all LeZeA MLOps services..."
    
    for service in "${services[@]}"; do
        stop_service "$service"
        sleep 2
    done
    
    # Clean up PID files
    rm -f "$PID_DIR"/*.pid
    
    log_success "All services stopped"
}

# Function to show logs for all services
show_all_logs() {
    log_info "Recent logs from all services:"
    echo "=============================="
    
    for log_file in "$LOG_DIR"/*.log; do
        if [[ -f "$log_file" ]]; then
            local service=$(basename "$log_file" .log)
            echo
            echo "=== $service logs (last 10 lines) ==="
            tail -n 10 "$log_file"
        fi
    done
}

# Function to validate environment
validate_environment() {
    log_info "Validating environment..."
    
    local issues=0
    
    # Check Python environment
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        ((issues++))
    else
        local python_version=$(python3 --version | cut -d' ' -f2)
        log_info "Python version: $python_version"
    fi
    
    # Check required Python packages
    local required_packages=("mlflow" "psycopg2" "pymongo" "boto3" "prometheus_client")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &>/dev/null; then
            log_warn "Python package '$package' not found"
        fi
    done
    
    # Check environment variables
    if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
        log_warn "POSTGRES_PASSWORD not set, using default"
    fi
    
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
        log_warn "AWS_ACCESS_KEY_ID not set, S3 features may not work"
    fi
    
    # Check disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 1000000 ]]; then  # Less than 1GB
        log_warn "Low disk space available: $(($available_space / 1024))MB"
    fi
    
    if [[ $issues -gt 0 ]]; then
        log_warn "Environment validation found $issues issues"
        return 1
    else
        log_success "Environment validation passed"
        return 0
    fi
}

# Main function
main() {
    local command="start"
    local quick_start=false
    local no_monitoring=false
    local no_storage=false
    local services_list=""
    local wait_time=5
    local verbose=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            start|stop|restart|status|health|logs)
                command="$1"
                shift
                ;;
            --quick)
                quick_start=true
                shift
                ;;
            --no-monitoring)
                no_monitoring=true
                shift
                ;;
            --no-storage)
                no_storage=true
                shift
                ;;
            --services)
                services_list="$2"
                shift 2
                ;;
            --wait-time)
                wait_time="$2"
                shift 2
                ;;
            --verbose)
                verbose=true
                set -x
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Set up environment
    setup_directories
    
    # Override with environment variables
    if [[ "${LEZEA_QUICK_START:-}" == "true" ]]; then
        quick_start=true
    fi
    
    if [[ -n "${LEZEA_WAIT_TIME:-}" ]]; then
        wait_time="$LEZEA_WAIT_TIME"
    fi
    
    if [[ -n "${LEZEA_SERVICES:-}" ]]; then
        services_list="$LEZEA_SERVICES"
    fi
    
    # Determine services to manage
    local services=()
    if [[ -n "$services_list" ]]; then
        IFS=',' read -ra services <<< "$services_list"
    else
        services=("postgresql" "mongodb")
        
        if [[ "$no_storage" != "true" ]]; then
            services+=("minio")
        fi
        
        services+=("mlflow")
        
        if [[ "$no_monitoring" != "true" ]]; then
            services+=("node_exporter" "gpu_exporter" "prometheus" "grafana")
        fi
    fi
    
    # Environment validation (unless quick start)
    if [[ "$quick_start" != "true" ]] && [[ "$command" == "start" || "$command" == "restart" ]]; then
        if ! validate_environment; then
            log_warn "Environment validation failed, but continuing anyway..."
            log_warn "Use --quick to skip validation checks"
        fi
    fi
    
    # Execute command
    case $command in
        start)
            start_all_services "${services[@]}"
            echo
            log_info "LeZeA MLOps is now running!"
            log_info "Use '$0 health' to check system health"
            log_info "Use '$0 stop' to stop all services"
            ;;
        stop)
            stop_all_services
            ;;
        restart)
            log_info "Restarting LeZeA MLOps services..."
            stop_all_services
            sleep 5
            start_all_services "${services[@]}"
            ;;
        status)
            echo "LeZeA MLOps Service Status:"
            echo "=========================="
            for service in "${services[@]}"; do
                show_status "$service"
            done
            ;;
        health)
            health_check
            ;;
        logs)
            show_all_logs
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'log_info "Received interrupt signal, cleaning up..."; stop_all_services; exit 130' INT TERM

# Run main function with all arguments
main "$@"