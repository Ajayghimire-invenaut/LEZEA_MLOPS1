#!/bin/bash
# LeZeA MLOps - MLflow Server Startup Script
# ==========================================
#
# This script starts MLflow server with production-ready configuration:
# - PostgreSQL backend for metadata storage
# - S3-compatible artifact storage
# - Authentication and security
# - Performance optimization
# - Health checks and monitoring

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="5000"
DEFAULT_WORKERS="4"
DEFAULT_BACKEND_STORE_URI="postgresql://lezea_user:lezea_secure_password_2024@localhost:5432/lezea_mlops"
DEFAULT_ARTIFACT_ROOT="s3://lezea-mlops-artifacts"
DEFAULT_LOG_LEVEL="INFO"

# Load environment variables from .env file if it exists
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment from $ENV_FILE"
    set -a  # Automatically export variables
    source "$ENV_FILE"
    set +a
fi

# Configuration from environment variables or defaults
MLFLOW_HOST="${MLFLOW_HOST:-$DEFAULT_HOST}"
MLFLOW_PORT="${MLFLOW_PORT:-$DEFAULT_PORT}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-$DEFAULT_WORKERS}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-$DEFAULT_BACKEND_STORE_URI}"
MLFLOW_DEFAULT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-$DEFAULT_ARTIFACT_ROOT}"
MLFLOW_LOG_LEVEL="${MLFLOW_LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"

# S3 configuration
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-}"

# Authentication configuration
MLFLOW_AUTH="${MLFLOW_AUTH:-false}"
MLFLOW_AUTH_CONFIG_PATH="${MLFLOW_AUTH_CONFIG_PATH:-$SCRIPT_DIR/auth_config.ini}"

# Logging configuration
LOG_DIR="$PROJECT_ROOT/logs"
MLFLOW_LOG_FILE="$LOG_DIR/mlflow_server.log"
PID_FILE="$LOG_DIR/mlflow_server.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [COMMAND]

MLflow Server Management Script for LeZeA MLOps

Commands:
    start       Start MLflow server (default)
    stop        Stop MLflow server
    restart     Restart MLflow server
    status      Check MLflow server status
    logs        Show MLflow server logs
    health      Check MLflow server health

Options:
    -h, --host HOST         Server host (default: $DEFAULT_HOST)
    -p, --port PORT         Server port (default: $DEFAULT_PORT)
    -w, --workers WORKERS   Number of workers (default: $DEFAULT_WORKERS)
    -d, --daemon            Run as daemon (background)
    -v, --verbose           Verbose output
    --help                  Show this help message

Environment Variables:
    MLFLOW_HOST                    Server host
    MLFLOW_PORT                    Server port
    MLFLOW_WORKERS                 Number of workers
    MLFLOW_BACKEND_STORE_URI       Database connection string
    MLFLOW_DEFAULT_ARTIFACT_ROOT   Artifact storage location
    MLFLOW_AUTH                    Enable authentication (true/false)
    AWS_ACCESS_KEY_ID              S3 access key
    AWS_SECRET_ACCESS_KEY          S3 secret key
    AWS_DEFAULT_REGION             S3 region

Examples:
    $0 start                       # Start with default settings
    $0 start -p 5001 -w 8         # Start on port 5001 with 8 workers
    $0 start -d                   # Start as daemon
    $0 stop                       # Stop the server
    $0 restart                    # Restart the server
    $0 status                     # Check server status

EOF
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if MLflow is installed
    if ! command -v mlflow &> /dev/null; then
        log_error "MLflow is not installed. Please install it with: pip install mlflow"
        exit 1
    fi
    
    # Check MLflow version
    MLFLOW_VERSION=$(mlflow --version 2>/dev/null | cut -d' ' -f2 || echo "unknown")
    log_info "MLflow version: $MLFLOW_VERSION"
    
    # Check if gunicorn is available for production deployment
    if command -v gunicorn &> /dev/null; then
        GUNICORN_AVAILABLE=true
        GUNICORN_VERSION=$(gunicorn --version 2>/dev/null | cut -d' ' -f2 || echo "unknown")
        log_info "Gunicorn version: $GUNICORN_VERSION"
    else
        GUNICORN_AVAILABLE=false
        log_warn "Gunicorn not available. Using development server."
    fi
}

# Function to validate configuration
validate_config() {
    log_info "Validating configuration..."
    
    # Check database connectivity
    if [[ "$MLFLOW_BACKEND_STORE_URI" == postgresql* ]]; then
        log_info "Testing PostgreSQL connection..."
        
        # Extract connection details from URI
        DB_HOST=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        DB_NAME=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*\/\([^?]*\).*/\1/p')
        DB_USER=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
        
        # Test connection (requires psql)
        if command -v psql &> /dev/null; then
            if PGPASSWORD=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*\/\/[^:]*:\([^@]*\)@.*/\1/p') \
               psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &>/dev/null; then
                log_success "PostgreSQL connection successful"
            else
                log_error "Cannot connect to PostgreSQL database"
                log_error "Please check your database configuration and ensure PostgreSQL is running"
                exit 1
            fi
        else
            log_warn "psql not available, skipping database connection test"
        fi
    fi
    
    # Check S3 configuration
    if [[ "$MLFLOW_DEFAULT_ARTIFACT_ROOT" == s3://* ]]; then
        log_info "Validating S3 configuration..."
        
        if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
            log_error "S3 artifact storage requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            exit 1
        fi
        
        # Test S3 access (requires aws cli)
        if command -v aws &> /dev/null; then
            BUCKET_NAME=$(echo "$MLFLOW_DEFAULT_ARTIFACT_ROOT" | sed 's|s3://||' | cut -d'/' -f1)
            if aws s3 ls "s3://$BUCKET_NAME" &>/dev/null; then
                log_success "S3 bucket access successful"
            else
                log_warn "Cannot access S3 bucket, but continuing anyway"
            fi
        else
            log_warn "AWS CLI not available, skipping S3 access test"
        fi
    fi
    
    # Check port availability
    if command -v netstat &> /dev/null; then
        if netstat -tlnp 2>/dev/null | grep -q ":$MLFLOW_PORT "; then
            log_error "Port $MLFLOW_PORT is already in use"
            log_error "Please choose a different port or stop the service using that port"
            exit 1
        fi
    elif command -v ss &> /dev/null; then
        if ss -tlnp 2>/dev/null | grep -q ":$MLFLOW_PORT "; then
            log_error "Port $MLFLOW_PORT is already in use"
            exit 1
        fi
    fi
}

# Function to create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Create MLflow configuration directory
    MLFLOW_CONFIG_DIR="$PROJECT_ROOT/.mlflow"
    mkdir -p "$MLFLOW_CONFIG_DIR"
    
    # Set permissions
    chmod 755 "$LOG_DIR"
    chmod 755 "$MLFLOW_CONFIG_DIR"
}

# Function to export environment variables
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Core MLflow settings
    export MLFLOW_BACKEND_STORE_URI
    export MLFLOW_DEFAULT_ARTIFACT_ROOT
    
    # S3 settings
    if [[ -n "$AWS_ACCESS_KEY_ID" ]]; then
        export AWS_ACCESS_KEY_ID
    fi
    if [[ -n "$AWS_SECRET_ACCESS_KEY" ]]; then
        export AWS_SECRET_ACCESS_KEY
    fi
    export AWS_DEFAULT_REGION
    
    if [[ -n "$MLFLOW_S3_ENDPOINT_URL" ]]; then
        export MLFLOW_S3_ENDPOINT_URL
    fi
    
    # Performance tuning
    export MLFLOW_SQLALCHEMY_ENGINE_OPTIONS='{"pool_pre_ping": true, "pool_recycle": 3600}'
    export MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT="600"
    
    # Security settings
    if [[ "$MLFLOW_AUTH" == "true" ]]; then
        export MLFLOW_AUTH_CONFIG_PATH
    fi
}

# Function to start MLflow server
start_mlflow() {
    local daemon_mode="$1"
    
    log_info "Starting MLflow server..."
    log_info "Host: $MLFLOW_HOST"
    log_info "Port: $MLFLOW_PORT"
    log_info "Workers: $MLFLOW_WORKERS"
    log_info "Backend Store: $MLFLOW_BACKEND_STORE_URI"
    log_info "Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
    
    # Build MLflow command
    MLFLOW_CMD="mlflow server"
    MLFLOW_CMD+=" --host $MLFLOW_HOST"
    MLFLOW_CMD+=" --port $MLFLOW_PORT"
    MLFLOW_CMD+=" --backend-store-uri $MLFLOW_BACKEND_STORE_URI"
    MLFLOW_CMD+=" --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT"
    MLFLOW_CMD+=" --workers $MLFLOW_WORKERS"
    
    # Add authentication if enabled
    if [[ "$MLFLOW_AUTH" == "true" ]]; then
        MLFLOW_CMD+=" --app-name basic-auth"
        log_info "Authentication enabled"
    fi
    
    # Add serve-artifacts flag for artifact proxying
    MLFLOW_CMD+=" --serve-artifacts"
    
    # Production deployment with Gunicorn
    if [[ "$GUNICORN_AVAILABLE" == "true" ]] && [[ "$MLFLOW_WORKERS" -gt 1 ]]; then
        MLFLOW_CMD+=" --gunicorn-opts '--timeout 120 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100'"
        log_info "Using Gunicorn for production deployment"
    fi
    
    if [[ "$daemon_mode" == "true" ]]; then
        # Start as daemon
        log_info "Starting MLflow server as daemon..."
        nohup $MLFLOW_CMD > "$MLFLOW_LOG_FILE" 2>&1 & echo $! > "$PID_FILE"
        
        # Wait a moment and check if process is still running
        sleep 3
        if check_mlflow_process; then
            log_success "MLflow server started successfully (PID: $(cat "$PID_FILE"))"
            log_info "Server URL: http://$MLFLOW_HOST:$MLFLOW_PORT"
            log_info "Logs: $MLFLOW_LOG_FILE"
        else
            log_error "Failed to start MLflow server"
            if [[ -f "$MLFLOW_LOG_FILE" ]]; then
                log_error "Check logs: $MLFLOW_LOG_FILE"
                tail -n 20 "$MLFLOW_LOG_FILE"
            fi
            exit 1
        fi
    else
        # Start in foreground
        log_info "Starting MLflow server in foreground..."
        log_info "Press Ctrl+C to stop"
        
        # Set up signal handlers
        trap cleanup EXIT INT TERM
        
        exec $MLFLOW_CMD
    fi
}

# Function to stop MLflow server
stop_mlflow() {
    log_info "Stopping MLflow server..."
    
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            log_info "Stopping process $PID..."
            kill "$PID"
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                log_warn "Forcefully killing process $PID..."
                kill -9 "$PID"
            fi
            
            rm -f "$PID_FILE"
            log_success "MLflow server stopped"
        else
            log_warn "Process $PID not found, removing stale PID file"
            rm -f "$PID_FILE"
        fi
    else
        # Try to find and kill MLflow processes
        MLFLOW_PIDS=$(pgrep -f "mlflow server" || true)
        if [[ -n "$MLFLOW_PIDS" ]]; then
            log_info "Found running MLflow processes: $MLFLOW_PIDS"
            echo "$MLFLOW_PIDS" | xargs kill
            log_success "Stopped MLflow processes"
        else
            log_info "No running MLflow server found"
        fi
    fi
}

# Function to check if MLflow process is running
check_mlflow_process() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Function to show MLflow server status
show_status() {
    log_info "Checking MLflow server status..."
    
    if check_mlflow_process; then
        PID=$(cat "$PID_FILE")
        log_success "MLflow server is running (PID: $PID)"
        log_info "Server URL: http://$MLFLOW_HOST:$MLFLOW_PORT"
        
        # Show resource usage if available
        if command -v ps &> /dev/null; then
            CPU_MEM=$(ps -p "$PID" -o pid,pcpu,pmem,etime --no-headers 2>/dev/null || echo "N/A")
            log_info "Resource usage: $CPU_MEM"
        fi
        
        # Test server connectivity
        health_check
    else
        log_warn "MLflow server is not running"
        return 1
    fi
}

# Function to perform health check
health_check() {
    local max_attempts=3
    local attempt=1
    
    log_info "Performing health check..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if command -v curl &> /dev/null; then
            if curl -s --connect-timeout 5 "http://$MLFLOW_HOST:$MLFLOW_PORT/health" > /dev/null; then
                log_success "MLflow server is healthy"
                return 0
            fi
        elif command -v wget &> /dev/null; then
            if wget -q --timeout=5 --tries=1 "http://$MLFLOW_HOST:$MLFLOW_PORT/health" -O /dev/null; then
                log_success "MLflow server is healthy"
                return 0
            fi
        else
            log_warn "Neither curl nor wget available for health check"
            return 0
        fi
        
        log_warn "Health check attempt $attempt failed"
        ((attempt++))
        if [[ $attempt -le $max_attempts ]]; then
            sleep 2
        fi
    done
    
    log_error "MLflow server health check failed after $max_attempts attempts"
    return 1
}

# Function to show logs
show_logs() {
    if [[ -f "$MLFLOW_LOG_FILE" ]]; then
        log_info "Showing MLflow server logs (last 50 lines):"
        echo "=================================================================================="
        tail -n 50 "$MLFLOW_LOG_FILE"
        echo "=================================================================================="
        log_info "Full logs: $MLFLOW_LOG_FILE"
    else
        log_warn "Log file not found: $MLFLOW_LOG_FILE"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Add any cleanup tasks here
}

# Main function
main() {
    local command="start"
    local daemon_mode="false"
    local verbose="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            start|stop|restart|status|logs|health)
                command="$1"
                shift
                ;;
            -h|--host)
                MLFLOW_HOST="$2"
                shift 2
                ;;
            -p|--port)
                MLFLOW_PORT="$2"
                shift 2
                ;;
            -w|--workers)
                MLFLOW_WORKERS="$2"
                shift 2
                ;;
            -d|--daemon)
                daemon_mode="true"
                shift
                ;;
            -v|--verbose)
                verbose="true"
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
    
    # Set up logging
    if [[ "$verbose" == "true" ]]; then
        log_info "Verbose mode enabled"
    fi
    
    # Execute command
    case $command in
        start)
            check_dependencies
            validate_config
            setup_directories
            setup_environment
            
            # Check if already running
            if check_mlflow_process; then
                log_warn "MLflow server is already running"
                show_status
                exit 1
            fi
            
            start_mlflow "$daemon_mode"
            ;;
        stop)
            stop_mlflow
            ;;
        restart)
            log_info "Restarting MLflow server..."
            stop_mlflow
            sleep 2
            
            check_dependencies
            validate_config
            setup_directories
            setup_environment
            start_mlflow "true"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        health)
            health_check
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"