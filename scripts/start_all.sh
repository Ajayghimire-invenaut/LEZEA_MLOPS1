#!/usr/bin/env bash
# LeZeA MLOps - Complete System Startup Script
# =============================================
#
# Starts services in order:
# 1. Infrastructure (PostgreSQL, MongoDB)
# 2. Storage (MinIO/S3)
# 3. MLflow server
# 4. Monitoring (Prometheus, Node Exporter, GPU Exporter, Grafana)
# 5. Health checks

set -euo pipefail

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.env"
  set +a
fi

LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$LOG_DIR/pids"
SERVICES_DIR="$PROJECT_ROOT/services"

# --- Colors ------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; PURPLE='\033[0;35m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}    $(date '+%F %T') - $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%F %T') - $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}   $(date '+%F %T') - $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC}  $(date '+%F %T') - $*"; }
log_service() { echo -e "${PURPLE}[SERVICE]${NC} $(date '+%F %T') - $*"; }
log_check()   { echo -e "${CYAN}[CHECK]${NC}   $(date '+%F %T') - $*"; }

# --- Service metadata --------------------------------------------------------
declare -A SERVICE_STATUS
declare -A SERVICE_PIDS
declare -A SERVICE_PORTS=(
  [postgresql]=5432
  [mongodb]=27017
  [minio]=9000
  [mlflow]=5000
  [prometheus]=9090
  [node_exporter]=9100
  [gpu_exporter]=9835     # will switch to 9400 if DCGM exporter is used
  [grafana]=3000
)

# --- Usage -------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 [OPTIONS] {start|stop|restart|status|health|logs}

Manage LeZeA MLOps services

Options:
  --quick             Skip validation checks
  --no-monitoring     Skip Prometheus/Grafana/exporters
  --no-storage        Skip MinIO
  --services LIST     Comma-separated subset to manage
  --wait-time N       Seconds to wait between starts (default: 5)
  --verbose           Verbose shell tracing
  --help              Show this help

Env overrides:
  LEZEA_QUICK_START=true
  LEZEA_WAIT_TIME=10
  LEZEA_SERVICES="postgresql,mongodb,mlflow"
EOF
}

# --- Setup -------------------------------------------------------------------
setup_directories() {
  log_info "Setting up directories..."
  mkdir -p "$LOG_DIR" "$PID_DIR" \
           "$PROJECT_ROOT/data" "$PROJECT_ROOT/artifacts" \
           "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/models"
  chmod 755 "$LOG_DIR" "$PID_DIR" \
            "$PROJECT_ROOT/data" "$PROJECT_ROOT/artifacts"
  log_success "Directories created"
}

# --- Port checks -------------------------------------------------------------
check_port() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null | grep -qE "[:.]$port\s"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -i :"$port" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v netstat >/dev/null 2>&1; then
    netstat -tlnp 2>/dev/null | grep -q ":$port "
  else
    return 1
  fi
}

# --- Wait for service --------------------------------------------------------
wait_for_service() {
  local service=$1
  local port=$2
  local max_attempts=${3:-30}
  local attempt=1
  log_check "Waiting for $service to be ready on port $port..."
  while [[ $attempt -le $max_attempts ]]; do
    if command -v nc >/dev/null 2>&1; then
      if nc -z localhost "$port" 2>/dev/null; then
        log_success "$service is ready on port $port"; return 0; fi
    fi
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS --connect-timeout 2 "http://localhost:$port" >/dev/null 2>&1; then
        log_success "$service is ready on port $port"; return 0; fi
    fi
    if check_port "$port"; then
      log_success "$service port $port is listening"; return 0
    fi
    sleep 2; ((attempt++))
  done
  log_error "$service failed to start on port $port after $((max_attempts*2))s"
  return 1
}

# --- Health checks -----------------------------------------------------------
check_service_health() {
  local service=$1
  local port=${SERVICE_PORTS[$service]}
  case "$service" in
    postgresql)
      if command -v psql >/dev/null 2>&1; then
        PGPASSWORD="${POSTGRES_PASSWORD:-lezea_secure_password_2024}" \
          psql -h localhost -p "${port}" -U "${POSTGRES_USER:-lezea_user}" \
               -d "${POSTGRES_DB:-lezea_mlops}" -c "SELECT 1;" >/dev/null 2>&1 && return 0
      fi
      ;;
    mongodb)
      local mongo_cmd="mongosh"; command -v mongosh >/dev/null 2>&1 || mongo_cmd="mongo"
      if command -v "$mongo_cmd" >/dev/null 2>&1; then
        "$mongo_cmd" --host "localhost:${port}" --eval "db.runCommand('ping')" >/dev/null 2>&1 && return 0
      fi
      ;;
    mlflow)
      # MLflow has no /health; root responds with HTML
      command -v curl >/dev/null 2>&1 && \
        curl -fsS --connect-timeout 2 "http://localhost:${port}/" >/dev/null 2>&1 && return 0
      ;;
    prometheus)
      command -v curl >/dev/null 2>&1 && \
        curl -fsS "http://localhost:${port}/-/healthy" >/dev/null 2>&1 && return 0
      ;;
    minio)
      command -v curl >/dev/null 2>&1 && \
        curl -fsS "http://localhost:${port}/minio/health/live" >/dev/null 2>&1 && return 0
      ;;
    grafana)
      command -v curl >/dev/null 2>&1 && \
        curl -fsS "http://localhost:${port}/api/health" >/dev/null 2>&1 && return 0
      ;;
    node_exporter|gpu_exporter)
      command -v curl >/dev/null 2>&1 && \
        curl -fsS "http://localhost:${port}/metrics" >/dev/null 2>&1 && return 0
      ;;
  esac
  # fallback: port listening
  check_port "$port"
}

# --- Start PostgreSQL --------------------------------------------------------
start_postgresql() {
  log_service "Starting PostgreSQL..."
  if check_service_health postgresql; then log_warn "PostgreSQL already running"; SERVICE_STATUS[postgresql]=running; return 0; fi

  if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-unit-files | grep -q '^postgresql'; then
      sudo systemctl start postgresql || true
      wait_for_service postgresql "${SERVICE_PORTS[postgresql]}" 15 && SERVICE_STATUS[postgresql]=running && log_success "PostgreSQL started" && return 0
    fi
  fi

  if command -v brew >/dev/null 2>&1; then
    if brew services info postgresql >/dev/null 2>&1; then
      brew services start postgresql || true
      wait_for_service postgresql "${SERVICE_PORTS[postgresql]}" 15 && SERVICE_STATUS[postgresql]=running && log_success "PostgreSQL started" && return 0
    fi
  fi

  if command -v pg_ctl >/dev/null 2>&1; then
    local data_dir="${PGDATA:-$PROJECT_ROOT/data/postgresql}"
    mkdir -p "$data_dir"
    [[ -f "$data_dir/PG_VERSION" ]] || initdb -D "$data_dir" >/dev/null 2>&1 || true
    pg_ctl -D "$data_dir" -l "$LOG_DIR/postgresql.log" start || true
    wait_for_service postgresql "${SERVICE_PORTS[postgresql]}" 15 && SERVICE_STATUS[postgresql]=running && log_success "PostgreSQL started" && return 0
  fi

  log_error "PostgreSQL not started; ensure it's installed & configured"
  SERVICE_STATUS[postgresql]=failed; return 1
}

# --- Start MongoDB -----------------------------------------------------------
start_mongodb() {
  log_service "Starting MongoDB..."
  if check_service_health mongodb; then log_warn "MongoDB already running"; SERVICE_STATUS[mongodb]=running; return 0; fi

  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^mongod'; then
    sudo systemctl start mongod || true
    wait_for_service mongodb "${SERVICE_PORTS[mongodb]}" 15 && SERVICE_STATUS[mongodb]=running && log_success "MongoDB started" && return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    if brew services info mongodb-community >/dev/null 2>&1; then
      brew services start mongodb-community || true
      wait_for_service mongodb "${SERVICE_PORTS[mongodb]}" 15 && SERVICE_STATUS[mongodb]=running && log_success "MongoDB started" && return 0
    fi
  fi

  if command -v mongod >/dev/null 2>&1; then
    local data_dir="${MONGODB_DATA_DIR:-$PROJECT_ROOT/data/mongodb}"
    mkdir -p "$data_dir"
    nohup mongod --dbpath "$data_dir" --logpath "$LOG_DIR/mongodb.log" --port "${SERVICE_PORTS[mongodb]}" --fork \
      >/dev/null 2>&1 || true
    wait_for_service mongodb "${SERVICE_PORTS[mongodb]}" 15 && SERVICE_STATUS[mongodb]=running && log_success "MongoDB started" && return 0
  fi

  log_error "MongoDB not started; ensure it's installed & configured"
  SERVICE_STATUS[mongodb]=failed; return 1
}

# --- Start MinIO -------------------------------------------------------------
start_minio() {
  log_service "Starting MinIO..."
  if check_service_health minio; then log_warn "MinIO already running"; SERVICE_STATUS[minio]=running; return 0; fi
  if ! command -v minio >/dev/null 2>&1; then log_warn "MinIO not found, skipping"; SERVICE_STATUS[minio]=skipped; return 0; fi

  local data_dir="${MINIO_DATA_DIR:-$PROJECT_ROOT/data/minio}"
  mkdir -p "$data_dir"
  export MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
  export MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin123}"

  nohup minio server "$data_dir" --address ":${SERVICE_PORTS[minio]}" --console-address ":9001" \
    >"$LOG_DIR/minio.log" 2>&1 & echo $! >"$PID_DIR/minio.pid"

  wait_for_service minio "${SERVICE_PORTS[minio]}" 20
  if check_service_health minio; then
    SERVICE_STATUS[minio]=running; SERVICE_PIDS[minio]=$(cat "$PID_DIR/minio.pid")
    log_success "MinIO started (console http://localhost:9001)"; return 0
  fi
  log_error "MinIO failed to start"; SERVICE_STATUS[minio]=failed; return 1
}

# --- Start MLflow ------------------------------------------------------------
start_mlflow() {
  log_service "Starting MLflow..."
  if check_service_health mlflow; then log_warn "MLflow already running"; SERVICE_STATUS[mlflow]=running; return 0; fi

  local mlflow_script="$SERVICES_DIR/mlflow/start.sh"
  if [[ -x "$mlflow_script" ]]; then
    "$mlflow_script" start --daemon || true
  else
    if ! command -v mlflow >/dev/null 2>&1; then
      log_error "mlflow CLI not found (pip install mlflow)"; SERVICE_STATUS[mlflow]=failed; return 1
    fi
    local store_dir="${MLFLOW_STORE_DIR:-$PROJECT_ROOT/data/mlflow}"
    local artifacts_dir="${MLFLOW_ARTIFACT_DIR:-$PROJECT_ROOT/artifacts/mlflow}"
    mkdir -p "$store_dir" "$artifacts_dir"
    local backend_uri="${MLFLOW_BACKEND_URI:-sqlite:///$store_dir/mlflow.db}"
    local artifact_uri="${MLFLOW_ARTIFACT_URI:-$artifacts_dir}"
    nohup mlflow server \
      --backend-store-uri "$backend_uri" \
      --default-artifact-root "$artifact_uri" \
      --host 0.0.0.0 --port "${SERVICE_PORTS[mlflow]}" \
      >"$LOG_DIR/mlflow.log" 2>&1 & echo $! >"$PID_DIR/mlflow.pid"
  fi

  wait_for_service mlflow "${SERVICE_PORTS[mlflow]}" 30
  if check_service_health mlflow; then
    SERVICE_STATUS[mlflow]=running
    [[ -f "$PID_DIR/mlflow.pid" ]] && SERVICE_PIDS[mlflow]=$(cat "$PID_DIR/mlflow.pid")
    log_success "MLflow started (http://localhost:${SERVICE_PORTS[mlflow]})"; return 0
  fi
  log_error "MLflow failed to start"; SERVICE_STATUS[mlflow]=failed; return 1
}

# --- Start Node Exporter -----------------------------------------------------
start_node_exporter() {
  log_service "Starting Node Exporter..."
  if check_service_health node_exporter; then log_warn "Node Exporter already running"; SERVICE_STATUS[node_exporter]=running; return 0; fi
  if ! command -v node_exporter >/dev/null 2>&1; then log_warn "node_exporter not found, skipping"; SERVICE_STATUS[node_exporter]=skipped; return 0; fi

  nohup node_exporter --web.listen-address=":${SERVICE_PORTS[node_exporter]}" \
    >"$LOG_DIR/node_exporter.log" 2>&1 & echo $! >"$PID_DIR/node_exporter.pid"

  wait_for_service node_exporter "${SERVICE_PORTS[node_exporter]}" 10
  if check_service_health node_exporter; then
    SERVICE_STATUS[node_exporter]=running; SERVICE_PIDS[node_exporter]=$(cat "$PID_DIR/node_exporter.pid")
    log_success "Node Exporter started"; return 0
  fi
  log_error "Node Exporter failed to start"; SERVICE_STATUS[node_exporter]=failed; return 1
}

# --- Start GPU Exporter (autodetect binary) ---------------------------------
start_gpu_exporter() {
  log_service "Starting GPU Exporter..."
  if ! command -v nvidia-smi >/dev/null 2>&1; then log_warn "NVIDIA drivers not found, skipping GPU exporter"; SERVICE_STATUS[gpu_exporter]=skipped; return 0; fi
  if check_service_health gpu_exporter; then log_warn "GPU Exporter already running"; SERVICE_STATUS[gpu_exporter]=running; return 0; fi

  local bin=""
  local port="${SERVICE_PORTS[gpu_exporter]}"

  if command -v nvidia-dcgm-exporter >/dev/null 2>&1; then
    bin="nvidia-dcgm-exporter"; port=9400
  elif command -v dcgm-exporter >/dev/null 2>&1; then
    bin="dcgm-exporter"; port=9400
  elif command -v nvidia_gpu_prometheus_exporter >/dev/null 2>&1; then
    bin="nvidia_gpu_prometheus_exporter"; port="${SERVICE_PORTS[gpu_exporter]}"
  else
    log_warn "No GPU exporter binary found, skipping"; SERVICE_STATUS[gpu_exporter]=skipped; return 0
  fi
  SERVICE_PORTS[gpu_exporter]=$port

  if [[ "$bin" == "nvidia_gpu_prometheus_exporter" ]]; then
    nohup nvidia_gpu_prometheus_exporter --web.listen-address=":${port}" \
      >"$LOG_DIR/gpu_exporter.log" 2>&1 & echo $! >"$PID_DIR/gpu_exporter.pid"
  else
    nohup "$bin" --web-listen-port "$port" \
      >"$LOG_DIR/gpu_exporter.log" 2>&1 & echo $! >"$PID_DIR/gpu_exporter.pid"
  fi

  wait_for_service gpu_exporter "$port" 15
  if check_service_health gpu_exporter; then
    SERVICE_STATUS[gpu_exporter]=running; SERVICE_PIDS[gpu_exporter]=$(cat "$PID_DIR/gpu_exporter.pid")
    log_success "GPU Exporter started on :$port"; return 0
  fi
  log_error "GPU Exporter failed to start"; SERVICE_STATUS[gpu_exporter]=failed; return 1
}

# --- Start Prometheus --------------------------------------------------------
start_prometheus() {
  log_service "Starting Prometheus..."
  if check_service_health prometheus; then log_warn "Prometheus already running"; SERVICE_STATUS[prometheus]=running; return 0; fi
  if ! command -v prometheus >/dev/null 2>&1; then log_warn "Prometheus not found, skipping"; SERVICE_STATUS[prometheus]=skipped; return 0; fi

  local cfg_project="$SERVICES_DIR/prometheus/prometheus.yml"
  local cfg_system="/etc/prometheus/prometheus.yml"
  local config_file=""
  if [[ -f "$cfg_project" ]]; then config_file="$cfg_project"
  elif [[ -f "$cfg_system" ]]; then config_file="$cfg_system"
  else log_error "Prometheus config not found"; SERVICE_STATUS[prometheus]=failed; return 1
  fi

  local data_dir="$PROJECT_ROOT/data/prometheus"; mkdir -p "$data_dir"
  nohup prometheus \
    --config.file="$config_file" \
    --storage.tsdb.path="$data_dir" \
    --storage.tsdb.retention.time="${PROM_RETENTION:-90d}" \
    --web.enable-lifecycle \
    --web.listen-address=":${SERVICE_PORTS[prometheus]}" \
    >"$LOG_DIR/prometheus.log" 2>&1 & echo $! >"$PID_DIR/prometheus.pid"

  wait_for_service prometheus "${SERVICE_PORTS[prometheus]}" 20
  if check_service_health prometheus; then
    SERVICE_STATUS[prometheus]=running; SERVICE_PIDS[prometheus]=$(cat "$PID_DIR/prometheus.pid")
    log_success "Prometheus started (http://localhost:${SERVICE_PORTS[prometheus]})"; return 0
  fi
  log_error "Prometheus failed to start"; SERVICE_STATUS[prometheus]=failed; return 1
}

# --- Start Grafana -----------------------------------------------------------
start_grafana() {
  log_service "Starting Grafana..."
  if check_service_health grafana; then log_warn "Grafana already running"; SERVICE_STATUS[grafana]=running; return 0; fi
  if ! command -v grafana-server >/dev/null 2>&1; then log_warn "grafana-server not found, skipping"; SERVICE_STATUS[grafana]=skipped; return 0; fi

  local data_dir="$PROJECT_ROOT/data/grafana"; mkdir -p "$data_dir"
  export GF_PATHS_DATA="$data_dir"
  export GF_PATHS_LOGS="$LOG_DIR"
  export GF_SERVER_HTTP_PORT="${SERVICE_PORTS[grafana]}"
  export GF_SECURITY_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin123}"

  nohup grafana-server \
    >"$LOG_DIR/grafana.log" 2>&1 & echo $! >"$PID_DIR/grafana.pid"

  wait_for_service grafana "${SERVICE_PORTS[grafana]}" 20
  if check_service_health grafana; then
    SERVICE_STATUS[grafana]=running; SERVICE_PIDS[grafana]=$(cat "$PID_DIR/grafana.pid")
    log_success "Grafana started (http://localhost:${SERVICE_PORTS[grafana]})"; return 0
  fi
  log_error "Grafana failed to start"; SERVICE_STATUS[grafana]=failed; return 1
}

# --- Stop service ------------------------------------------------------------
stop_service() {
  local service=$1
  log_service "Stopping $service..."
  case "$service" in
    postgresql)
      if command -v systemctl >/dev/null 2>&1; then sudo systemctl stop postgresql 2>/dev/null || true
      elif command -v brew >/dev/null 2>&1; then brew services stop postgresql 2>/dev/null || true
      elif command -v pg_ctl >/dev/null 2>&1; then pg_ctl stop -D "${PGDATA:-$PROJECT_ROOT/data/postgresql}" 2>/dev/null || true
      fi
      ;;
    mongodb)
      if command -v systemctl >/dev/null 2>&1; then sudo systemctl stop mongod 2>/dev/null || true
      elif command -v brew >/dev/null 2>&1; then brew services stop mongodb-community 2>/dev/null || true
      fi
      ;;
    mlflow)
      local mlflow_script="$SERVICES_DIR/mlflow/start.sh"
      if [[ -x "$mlflow_script" ]]; then "$mlflow_script" stop 2>/dev/null || true; fi
      ;;
  esac

  if [[ -f "$PID_DIR/$service.pid" ]]; then
    local pid; pid="$(cat "$PID_DIR/$service.pid" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true; sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_DIR/$service.pid"
  fi

  pkill -f "$service" 2>/dev/null || true
  log_success "$service stopped"
}

# --- Status ------------------------------------------------------------------
show_status() {
  local service=$1
  local color="$RED"; local status="stopped"
  if check_service_health "$service"; then color="$GREEN"; status="running"; fi
  local pid_str=""
  [[ -f "$PID_DIR/$service.pid" ]] && pid_str=" (PID: $(cat "$PID_DIR/$service.pid"))"
  printf "%-15s: ${color}%-8s${NC}%s\n" "$service" "$status" "$pid_str"
}

# --- Health check ------------------------------------------------------------
health_check() {
  log_info "Comprehensive health check..."
  local all_healthy=true
  local services=(postgresql mongodb minio mlflow prometheus node_exporter gpu_exporter grafana)

  echo; echo "Service Health Status:"; echo "====================="
  for s in "${services[@]}"; do
    if check_service_health "$s"; then
      printf "%-15s: ${GREEN}✓ HEALTHY${NC}\n" "$s"
    else
      printf "%-15s: ${RED}✗ UNHEALTHY${NC}\n" "$s"
      all_healthy=false
    fi
  done

  echo; echo "System Resources:"; echo "================"
  if command -v top >/dev/null 2>&1; then
    local cpu_usage
    cpu_usage=$(top -bn1 | awk -F'[, ]+' '/Cpu\(s\)/ {print 100 - $8}')
    printf "CPU Usage:       %.1f%%\n" "${cpu_usage:-0}"
  fi
  if command -v free >/dev/null 2>&1; then
    free -h | awk 'NR==2{printf "Memory Usage:    %s/%s (%.1f%%)\n",$3,$2,($3/$2)*100}'
  fi
  df -h "$PROJECT_ROOT" | awk 'NR==2{printf "Disk Usage:      %s/%s (%s)\n",$3,$2,$5}'

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo; echo "GPU Status:"; echo "==========="
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits | \
      while IFS=, read -r name util mem_used mem_total temp; do
        printf "%-22s: %3s%% util, %s/%s MB, %s°C\n" "$name" "$util" "$mem_used" "$mem_total" "$temp"
      done
  fi

  echo
  if $all_healthy; then log_success "All services healthy"; return 0
  else log_warn "Some services unhealthy — check logs"; return 1
  fi
}

# --- Start all ---------------------------------------------------------------
start_all_services() {
  local wait_time="${LEZEA_WAIT_TIME:-5}"
  local services=("$@")
  if [[ ${#services[@]} -eq 0 ]]; then
    services=(postgresql mongodb minio mlflow node_exporter gpu_exporter prometheus grafana)
  fi

  log_info "Starting services: ${services[*]}"

  for s in "${services[@]}"; do
    case "$s" in postgresql|mongodb|minio) "start_$s"; [[ $? -eq 0 ]] && sleep "$wait_time";; esac
  done
  for s in "${services[@]}"; do
    case "$s" in mlflow) "start_$s"; [[ $? -eq 0 ]] && sleep "$wait_time";; esac
  done
  for s in "${services[@]}"; do
    case "$s" in node_exporter|gpu_exporter|prometheus|grafana) "start_$s"; [[ $? -eq 0 ]] && sleep "$wait_time";; esac
  done

  echo; log_info "Startup complete! Status:"; echo "========================="
  for s in "${services[@]}"; do show_status "$s"; done

  echo; log_info "Access URLs:"; echo "============"
  echo "MLflow UI:       http://localhost:${SERVICE_PORTS[mlflow]}"
  echo "Prometheus UI:   http://localhost:${SERVICE_PORTS[prometheus]}"
  echo "Grafana UI:      http://localhost:${SERVICE_PORTS[grafana]} (admin/${GRAFANA_ADMIN_PASSWORD:-admin123})"
  echo "MinIO Console:   http://localhost:9001 (user: ${MINIO_ROOT_USER:-minioadmin})"
  echo; echo "Logs: $LOG_DIR"; echo "PIDs: $PID_DIR"
}

# --- Stop all ---------------------------------------------------------------
stop_all_services() {
  local order=(grafana prometheus gpu_exporter node_exporter mlflow minio mongodb postgresql)
  log_info "Stopping all services..."
  for s in "${order[@]}"; do stop_service "$s"; sleep 1; done
  rm -f "$PID_DIR"/*.pid 2>/dev/null || true
  log_success "All services stopped"
}

# --- Show logs ---------------------------------------------------------------
show_all_logs() {
  log_info "Recent logs:"
  for f in "$LOG_DIR"/*.log; do
    [[ -f "$f" ]] || continue
    echo; echo "=== $(basename "$f") (last 20 lines) ==="; tail -n 20 "$f"
  done
}

# --- Validation --------------------------------------------------------------
validate_environment() {
  log_info "Validating environment..."
  local issues=0

  if ! command -v python3 >/dev/null 2>&1; then log_error "python3 not found"; ((issues++)); else
    log_info "Python: $(python3 --version 2>/dev/null | awk '{print $2}')"
  fi

  local pkgs=(mlflow psycopg2 pymongo boto3 prometheus_client)
  for p in "${pkgs[@]}"; do
    python3 - <<PY >/dev/null 2>&1 || log_warn "Python package missing: $p"
import importlib, sys; sys.exit(0 if importlib.util.find_spec("$p") else 1)
PY
  done

  [[ -z "${POSTGRES_PASSWORD:-}" ]] && log_warn "POSTGRES_PASSWORD not set (using default)"
  [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && log_warn "AWS_ACCESS_KEY_ID not set (S3 may be disabled)"

  local avail_kb; avail_kb=$(df "$PROJECT_ROOT" | awk 'NR==2{print $4}')
  if [[ ${avail_kb:-0} -lt 1000000 ]]; then log_warn "Low disk space: $((avail_kb/1024)) MB"; fi

  if (( issues > 0 )); then log_warn "Validation found $issues issue(s)"; return 1
  else log_success "Environment validation passed"; return 0
  fi
}

# --- Main --------------------------------------------------------------------
main() {
  local command="start"
  local quick_start=false
  local no_monitoring=false
  local no_storage=false
  local services_list=""
  local wait_time=5
  local verbose=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      start|stop|restart|status|health|logs) command="$1"; shift;;
      --quick) quick_start=true; shift;;
      --no-monitoring) no_monitoring=true; shift;;
      --no-storage) no_storage=true; shift;;
      --services) services_list="${2:-}"; shift 2;;
      --wait-time) wait_time="${2:-5}"; shift 2;;
      --verbose) verbose=true; set -x; shift;;
      --help|-h) usage; exit 0;;
      *) log_error "Unknown option: $1"; usage; exit 1;;
    esac
  done

  setup_directories

  [[ "${LEZEA_QUICK_START:-}" == "true" ]] && quick_start=true
  [[ -n "${LEZEA_WAIT_TIME:-}" ]] && wait_time="$LEZEA_WAIT_TIME"
  [[ -n "${LEZEA_SERVICES:-}" ]] && services_list="$LEZEA_SERVICES"

  local services=()
  if [[ -n "$services_list" ]]; then
    IFS=',' read -r -a services <<<"$services_list"
  else
    services=(postgresql mongodb)
    [[ "$no_storage" != "true" ]] && services+=(minio)
    services+=(mlflow)
    [[ "$no_monitoring" != "true" ]] && services+=(node_exporter gpu_exporter prometheus grafana)
  fi
  export LEZEA_WAIT_TIME="$wait_time"

  if [[ "$quick_start" != "true" ]] && [[ "$command" == "start" || "$command" == "restart" ]]; then
    validate_environment || { log_warn "Validation failed; continuing (use --quick to skip)"; }
  fi

  case "$command" in
    start)   start_all_services "${services[@]}"; echo; log_info "LeZeA MLOps is up. Try '$0 health'.";;
    stop)    stop_all_services;;
    restart) log_info "Restarting..."; stop_all_services; sleep 3; start_all_services "${services[@]}";;
    status)  echo "LeZeA MLOps Service Status:"; echo "=========================="; for s in "${services[@]}"; do show_status "$s"; done;;
    health)  health_check;;
    logs)    show_all_logs;;
    *)       log_error "Unknown command: $command"; usage; exit 1;;
  esac
}

trap 'log_info "Interrupt received, stopping..."; stop_all_services; exit 130' INT TERM
main "$@"
