#!/bin/bash

# Model Monitoring Startup Script
# Starts the complete monitoring stack for the MLOps pipeline

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODE="full"
CONFIG_FILE="$PROJECT_ROOT/config/monitoring.json"
ENVIRONMENT="dev"
DETACHED=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Start the MLOps model monitoring system with various components.

OPTIONS:
    -m, --mode MODE             Monitoring mode: full, basic, drift-only, performance-only (default: full)
    -c, --config CONFIG_FILE    Path to monitoring configuration file
    -e, --environment ENV       Environment: dev, staging, prod (default: dev)
    -d, --detached              Run in detached mode (background)
    -v, --verbose               Enable verbose output
    -h, --help                  Show this help message

MODES:
    full                        Start all monitoring components (default)
    basic                       Start basic performance monitoring only
    drift-only                  Start data drift detection only
    performance-only            Start performance monitoring only
    ab-testing                  Start A/B testing analysis only

EXAMPLES:
    # Start full monitoring stack
    $0

    # Start only drift detection
    $0 --mode drift-only

    # Start in production environment
    $0 --environment prod --detached

    # Start with custom config
    $0 --config /path/to/custom/monitoring.json

COMPONENTS:
    - Model Performance Monitor
    - Data Drift Detector
    - A/B Testing Analyzer
    - Alert Manager
    - Prometheus Metrics
    - MLflow Integration

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--detached)
                DETACHED=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    if [[ ! "$MODE" =~ ^(full|basic|drift-only|performance-only|ab-testing)$ ]]; then
        log_error "Invalid mode: $MODE"
        exit 1
    fi

    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi

    # Check required Python packages
    if ! python3 -c "import mlflow, pandas, numpy, prometheus_client, redis, requests" 2>/dev/null; then
        log_error "Required Python packages are missing. Please run: pip install -r requirements.txt"
        exit 1
    fi

    # Check if monitoring dependencies are running
    check_dependencies

    log_success "Prerequisites check passed"
}

# Check external dependencies
check_dependencies() {
    local config_content=$(cat "$CONFIG_FILE")
    
    # Check MLflow
    local mlflow_uri=$(echo "$config_content" | python3 -c "
import json, sys
config = json.load(sys.stdin)
print(config['mlflow']['tracking_uri'])
")
    
    if ! curl -s "$mlflow_uri/health" > /dev/null 2>&1; then
        log_warning "MLflow server is not accessible at $mlflow_uri"
        log_info "Please start MLflow server: mlflow server --host 0.0.0.0 --port 5000"
    else
        log_success "MLflow server is accessible"
    fi
    
    # Check Redis (if enabled)
    local redis_enabled=$(echo "$config_content" | python3 -c "
import json, sys
config = json.load(sys.stdin)
print(config.get('redis', {}).get('enabled', False))
")
    
    if [[ "$redis_enabled" == "True" ]]; then
        if ! command -v redis-cli &> /dev/null || ! redis-cli ping > /dev/null 2>&1; then
            log_warning "Redis is not accessible"
            log_info "Please start Redis server: redis-server"
        else
            log_success "Redis server is accessible"
        fi
    fi
    
    # Check Prometheus Pushgateway (if enabled)
    local prometheus_enabled=$(echo "$config_content" | python3 -c "
import json, sys
config = json.load(sys.stdin)
print(config.get('prometheus', {}).get('enabled', False))
")
    
    if [[ "$prometheus_enabled" == "True" ]]; then
        local pushgateway_url=$(echo "$config_content" | python3 -c "
import json, sys
config = json.load(sys.stdin)
print(config.get('prometheus', {}).get('pushgateway', 'localhost:9091'))
")
        
        if ! curl -s "http://$pushgateway_url/metrics" > /dev/null 2>&1; then
            log_warning "Prometheus Pushgateway is not accessible at $pushgateway_url"
            log_info "Metrics will still work, but won't be pushed to Prometheus"
        else
            log_success "Prometheus Pushgateway is accessible"
        fi
    fi
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    # Create log directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Create prediction logs directory (for monitoring)
    mkdir -p "$PROJECT_ROOT/prediction_logs"
    
    # Create monitoring reports directory
    mkdir -p "$PROJECT_ROOT/monitoring_reports"
    
    log_success "Directories created"
}

# Start monitoring components based on mode
start_monitoring() {
    log_info "Starting monitoring in $MODE mode for $ENVIRONMENT environment"
    
    local python_args="--config $CONFIG_FILE"
    local log_file="$PROJECT_ROOT/logs/monitoring_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"
    
    if [[ $VERBOSE == true ]]; then
        python_args="$python_args --verbose"
    fi
    
    case "$MODE" in
        full)
            start_full_monitoring "$python_args" "$log_file"
            ;;
        basic)
            start_basic_monitoring "$python_args" "$log_file"
            ;;
        drift-only)
            start_drift_monitoring "$python_args" "$log_file"
            ;;
        performance-only)
            start_performance_monitoring "$python_args" "$log_file"
            ;;
        ab-testing)
            start_ab_testing "$python_args" "$log_file"
            ;;
    esac
}

# Start full monitoring stack
start_full_monitoring() {
    local args="$1"
    local log_file="$2"
    
    log_info "Starting full monitoring stack..."
    
    if [[ $DETACHED == true ]]; then
        nohup python3 "$PROJECT_ROOT/src/monitoring/model_monitor.py" $args --mode continuous > "$log_file" 2>&1 &
        local pid=$!
        echo $pid > "$PROJECT_ROOT/logs/monitoring.pid"
        log_success "Full monitoring started in background (PID: $pid)"
        log_info "Log file: $log_file"
        log_info "To stop: kill $pid or use scripts/stop-monitoring.sh"
    else
        log_info "Starting full monitoring (Press Ctrl+C to stop)..."
        python3 "$PROJECT_ROOT/src/monitoring/model_monitor.py" $args --mode continuous
    fi
}

# Start basic monitoring
start_basic_monitoring() {
    local args="$1"
    local log_file="$2"
    
    log_info "Starting basic performance monitoring..."
    
    # Create a basic config that only enables performance monitoring
    local basic_config=$(mktemp)
    cat "$CONFIG_FILE" | python3 -c "
import json, sys
config = json.load(sys.stdin)
config['drift_detection']['enabled'] = False
config['ab_testing']['enabled'] = False
print(json.dumps(config, indent=2))
" > "$basic_config"
    
    if [[ $DETACHED == true ]]; then
        nohup python3 "$PROJECT_ROOT/src/monitoring/model_monitor.py" --config "$basic_config" --mode continuous > "$log_file" 2>&1 &
        local pid=$!
        echo $pid > "$PROJECT_ROOT/logs/monitoring.pid"
        log_success "Basic monitoring started in background (PID: $pid)"
    else
        python3 "$PROJECT_ROOT/src/monitoring/model_monitor.py" --config "$basic_config" --mode continuous
    fi
    
    rm "$basic_config"
}

# Start drift monitoring only
start_drift_monitoring() {
    local args="$1"
    local log_file="$2"
    
    log_info "Starting data drift monitoring..."
    
    # Run single cycle focused on drift detection
    python3 - << EOF
import asyncio
import sys
sys.path.append('$PROJECT_ROOT')
from src.monitoring.model_monitor import ModelMonitor

async def main():
    monitor = ModelMonitor('$CONFIG_FILE')
    while True:
        await monitor._monitor_drift()
        await asyncio.sleep(monitor.config['drift_detection']['check_interval'])

if __name__ == "__main__":
    asyncio.run(main())
EOF
}

# Start performance monitoring only
start_performance_monitoring() {
    local args="$1"
    local log_file="$2"
    
    log_info "Starting performance monitoring..."
    
    python3 - << EOF
import asyncio
import sys
sys.path.append('$PROJECT_ROOT')
from src.monitoring.model_monitor import ModelMonitor

async def main():
    monitor = ModelMonitor('$CONFIG_FILE')
    while True:
        await monitor._monitor_performance()
        await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    asyncio.run(main())
EOF
}

# Start A/B testing analysis
start_ab_testing() {
    local args="$1"
    local log_file="$2"
    
    log_info "Starting A/B testing analysis..."
    
    python3 - << EOF
import asyncio
import sys
sys.path.append('$PROJECT_ROOT')
from src.monitoring.model_monitor import ModelMonitor

async def main():
    monitor = ModelMonitor('$CONFIG_FILE')
    while True:
        await monitor._analyze_ab_tests()
        await asyncio.sleep(3600)  # 1 hour

if __name__ == "__main__":
    asyncio.run(main())
EOF
}

# Show monitoring status
show_status() {
    log_info "=== Monitoring Status ==="
    
    # Check if monitoring is running
    if [[ -f "$PROJECT_ROOT/logs/monitoring.pid" ]]; then
        local pid=$(cat "$PROJECT_ROOT/logs/monitoring.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_success "Monitoring is running (PID: $pid)"
        else
            log_warning "Monitoring PID file exists but process is not running"
            rm -f "$PROJECT_ROOT/logs/monitoring.pid"
        fi
    else
        log_info "No monitoring process detected"
    fi
    
    # Show recent logs
    local latest_log=$(ls -t "$PROJECT_ROOT/logs/monitoring_"*.log 2>/dev/null | head -1)
    if [[ -n "$latest_log" ]]; then
        log_info "Latest log file: $latest_log"
        if [[ $VERBOSE == true ]]; then
            log_info "Recent log entries:"
            tail -10 "$latest_log" 2>/dev/null || true
        fi
    fi
    
    # Check component health
    check_component_health
}

# Check component health
check_component_health() {
    log_info "=== Component Health ==="
    
    # Check if model serving API is responding
    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        log_success "Model serving API is healthy"
    else
        log_warning "Model serving API is not responding"
    fi
    
    # Check MLflow
    local mlflow_uri=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config['mlflow']['tracking_uri'])
")
    
    if curl -s "$mlflow_uri/health" > /dev/null 2>&1; then
        log_success "MLflow is healthy"
    else
        log_warning "MLflow is not responding"
    fi
}

# Main function
main() {
    log_info "=== MLOps Model Monitoring System ==="
    log_info "Mode: $MODE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Config: $CONFIG_FILE"
    
    if [[ $DETACHED == true ]]; then
        log_info "Running in detached mode"
    fi

    # Setup
    check_prerequisites
    setup_directories
    
    # Show current status first
    show_status
    
    # Start monitoring
    start_monitoring
    
    log_success "Monitoring system ready!"
}

# Handle cleanup on exit
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Monitoring startup completed successfully"
    else
        log_error "Monitoring startup failed with exit code $exit_code"
    fi
    
    exit $exit_code
}

# Set up error handling
trap cleanup EXIT

# Parse arguments and run main function
parse_args "$@"
validate_args
main