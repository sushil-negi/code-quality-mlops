#!/bin/bash

# Model Deployment Script
# Automates the deployment of trained models to various environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_NAME=""
MODEL_VERSION=""
ENVIRONMENT="dev"
TARGET="kubernetes"
STRATEGY="rolling"
CONFIG_FILE="$PROJECT_ROOT/config/deployment.json"
DRY_RUN=false
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

Deploy ML models to various environments with different strategies.

OPTIONS:
    -m, --model-name MODEL_NAME     Name of the model to deploy (required)
    -v, --model-version VERSION     Version of the model to deploy (required)
    -e, --environment ENV           Deployment environment: dev, staging, prod (default: dev)
    -t, --target TARGET             Deployment target: kubernetes, ecs, lambda (default: kubernetes)
    -s, --strategy STRATEGY         Deployment strategy: rolling, blue_green, canary (default: rolling)
    -c, --config CONFIG_FILE        Path to deployment configuration file
    -d, --dry-run                   Show what would be deployed without actually deploying
    -V, --verbose                   Enable verbose output
    -h, --help                      Show this help message

EXAMPLES:
    # Deploy simple model to dev environment
    $0 --model-name simple_bug_predictor --model-version 1 --environment dev

    # Deploy hybrid model to production with blue-green strategy
    $0 -m hybrid_bug_predictor -v 2 -e prod -s blue_green

    # Dry run deployment to staging
    $0 -m simple_bug_predictor -v 1 -e staging --dry-run

    # Deploy to AWS Lambda
    $0 -m simple_bug_predictor -v 1 -t lambda -e prod

DEPLOYMENT TARGETS:
    kubernetes      Deploy to Kubernetes cluster with auto-scaling
    ecs             Deploy to AWS ECS/Fargate
    lambda          Deploy to AWS Lambda (serverless)

DEPLOYMENT STRATEGIES:
    rolling         Rolling update deployment (zero downtime)
    blue_green      Blue-green deployment (instant switch)
    canary          Canary deployment (gradual rollout)

ENVIRONMENTS:
    dev             Development environment (minimal resources)
    staging         Staging environment (production-like)
    prod            Production environment (full resources)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            -v|--model-version)
                MODEL_VERSION="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -V|--verbose)
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

# Validate required arguments
validate_args() {
    if [[ -z "$MODEL_NAME" ]]; then
        log_error "Model name is required. Use --model-name or -m."
        exit 1
    fi

    if [[ -z "$MODEL_VERSION" ]]; then
        log_error "Model version is required. Use --model-version or -v."
        exit 1
    fi

    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
        exit 1
    fi

    if [[ ! "$TARGET" =~ ^(kubernetes|ecs|lambda)$ ]]; then
        log_error "Invalid target: $TARGET. Must be kubernetes, ecs, or lambda."
        exit 1
    fi

    if [[ ! "$STRATEGY" =~ ^(rolling|blue_green|canary)$ ]]; then
        log_error "Invalid strategy: $STRATEGY. Must be rolling, blue_green, or canary."
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
    if ! python3 -c "import mlflow, docker, kubernetes, boto3" 2>/dev/null; then
        log_error "Required Python packages are missing. Please run: pip install -r requirements.txt"
        exit 1
    fi

    # Check Docker (if using Kubernetes or ECS)
    if [[ "$TARGET" == "kubernetes" || "$TARGET" == "ecs" ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is required for $TARGET deployment but not installed."
            exit 1
        fi

        # Check if Docker daemon is running
        if ! docker info &> /dev/null; then
            log_error "Docker daemon is not running."
            exit 1
        fi
    fi

    # Check kubectl (if using Kubernetes)
    if [[ "$TARGET" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is required for Kubernetes deployment but not installed."
            exit 1
        fi

        # Check cluster connectivity
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
            exit 1
        fi
    fi

    # Check AWS CLI (if using ECS or Lambda)
    if [[ "$TARGET" == "ecs" || "$TARGET" == "lambda" ]]; then
        if ! command -v aws &> /dev/null; then
            log_error "AWS CLI is required for $TARGET deployment but not installed."
            exit 1
        fi

        # Check AWS credentials
        if ! aws sts get-caller-identity &> /dev/null; then
            log_error "AWS credentials not configured. Run 'aws configure'."
            exit 1
        fi
    fi

    log_success "Prerequisites check passed"
}

# Validate model exists in MLflow
validate_model() {
    log_info "Validating model in MLflow..."

    local mlflow_uri=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config['mlflow']['tracking_uri'])
")

    # Check if MLflow server is accessible
    if ! curl -s "$mlflow_uri/health" > /dev/null; then
        log_error "MLflow server is not accessible at $mlflow_uri"
        exit 1
    fi

    # Validate model exists
    local validation_result=$(python3 - << EOF
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('$mlflow_uri')
client = MlflowClient()

try:
    model_version = client.get_model_version('$MODEL_NAME', '$MODEL_VERSION')
    print(f"Model validated: {model_version.name} v{model_version.version}")
    print(f"Stage: {model_version.current_stage}")
    print(f"Status: {model_version.status}")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
EOF
)

    if [[ $? -ne 0 ]]; then
        log_error "Model validation failed"
        exit 1
    fi

    if [[ $VERBOSE == true ]]; then
        log_info "Model validation result:"
        echo "$validation_result"
    fi

    log_success "Model validation passed"
}

# Build deployment configuration
build_deployment_config() {
    log_info "Building deployment configuration..."

    # Create temporary config file
    local temp_config=$(mktemp)

    # Extract environment-specific configuration
    python3 - << EOF > "$temp_config"
import json

# Load base configuration
with open('$CONFIG_FILE') as f:
    config = json.load(f)

# Extract environment-specific settings
env_config = config.get('environments', {}).get('$ENVIRONMENT', {})

# Build deployment configuration
deployment_config = {
    'model_name': '$MODEL_NAME',
    'model_version': '$MODEL_VERSION',
    'deployment_target': '$TARGET',
    'environment': '$ENVIRONMENT',
    'strategy': '$STRATEGY',
    'cpu_request': env_config.get('resources', {}).get('cpu_request', '100m'),
    'cpu_limit': env_config.get('resources', {}).get('cpu_limit', '500m'),
    'memory_request': env_config.get('resources', {}).get('memory_request', '256Mi'),
    'memory_limit': env_config.get('resources', {}).get('memory_limit', '512Mi'),
    'min_replicas': env_config.get('replicas', {}).get('min', 1),
    'max_replicas': env_config.get('replicas', {}).get('max', 3),
    'target_cpu_utilization': 70
}

print(json.dumps(deployment_config, indent=2))
EOF

    if [[ $VERBOSE == true ]]; then
        log_info "Deployment configuration:"
        cat "$temp_config"
    fi

    echo "$temp_config"
}

# Execute deployment
execute_deployment() {
    local config_file="$1"

    log_info "Starting deployment: $MODEL_NAME v$MODEL_VERSION to $ENVIRONMENT ($TARGET)"

    if [[ $DRY_RUN == true ]]; then
        log_warning "DRY RUN MODE - No actual deployment will be performed"
        log_info "Would deploy with configuration:"
        cat "$config_file"
        return 0
    fi

    # Run deployment manager
    local deployment_result=$(python3 "$PROJECT_ROOT/src/pipeline/deployment_manager.py" \
        --model-name "$MODEL_NAME" \
        --model-version "$MODEL_VERSION" \
        --target "$TARGET" \
        --environment "$ENVIRONMENT" \
        --strategy "$STRATEGY" \
        --config "$CONFIG_FILE" 2>&1)

    if [[ $? -eq 0 ]]; then
        log_success "Deployment completed successfully"
        
        if [[ $VERBOSE == true ]]; then
            log_info "Deployment result:"
            echo "$deployment_result"
        fi

        # Extract deployment information
        local deployment_info=$(echo "$deployment_result" | tail -n 1)
        log_info "Deployment details: $deployment_info"

    else
        log_error "Deployment failed"
        echo "$deployment_result"
        return 1
    fi
}

# Health check after deployment
health_check() {
    log_info "Performing post-deployment health check..."

    # Wait a bit for services to start
    sleep 30

    case "$TARGET" in
        kubernetes)
            health_check_kubernetes
            ;;
        ecs)
            health_check_ecs
            ;;
        lambda)
            health_check_lambda
            ;;
    esac
}

health_check_kubernetes() {
    local app_name="${MODEL_NAME}-serving"
    local namespace="mlops-${ENVIRONMENT}"

    # Check deployment status
    local deployment_status=$(kubectl get deployment "$app_name" -n "$namespace" -o jsonpath='{.status.readyReplicas}/{.status.replicas}' 2>/dev/null || echo "0/0")
    
    if [[ "$deployment_status" == "0/0" ]]; then
        log_warning "Deployment not found or not ready: $deployment_status"
        return 1
    fi

    # Check service endpoints
    local service_ip=$(kubectl get service "$app_name" -n "$namespace" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$service_ip" ]]; then
        # Test health endpoint
        if curl -s "http://$service_ip/health" > /dev/null; then
            log_success "Health check passed - Service is responding"
        else
            log_warning "Health check failed - Service not responding"
            return 1
        fi
    else
        log_info "LoadBalancer IP not yet assigned, checking pod status"
        kubectl get pods -n "$namespace" -l app="$app_name"
    fi
}

health_check_ecs() {
    log_info "ECS health check - checking service status"
    # Implementation would check ECS service status
    log_warning "ECS health check not implemented yet"
}

health_check_lambda() {
    log_info "Lambda health check - invoking function"
    # Implementation would invoke Lambda function
    log_warning "Lambda health check not implemented yet"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    # Remove temporary files
    find /tmp -name "deployment_config_*" -delete 2>/dev/null || true
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Deployment script completed successfully"
    else
        log_error "Deployment script failed with exit code $exit_code"
    fi
    
    exit $exit_code
}

# Main function
main() {
    # Set up error handling
    trap cleanup EXIT

    log_info "=== MLOps Model Deployment Script ==="
    log_info "Model: $MODEL_NAME v$MODEL_VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Target: $TARGET"
    log_info "Strategy: $STRATEGY"
    
    if [[ $DRY_RUN == true ]]; then
        log_warning "DRY RUN MODE ENABLED"
    fi

    # Check prerequisites
    check_prerequisites

    # Validate model
    validate_model

    # Build configuration
    local config_file=$(build_deployment_config)

    # Execute deployment
    execute_deployment "$config_file"

    # Health check
    if [[ $DRY_RUN == false ]]; then
        health_check
    fi

    log_success "Deployment pipeline completed!"
}

# Parse arguments and run main function
parse_args "$@"
validate_args
main