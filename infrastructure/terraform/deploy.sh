#!/bin/bash

# MLOps Infrastructure Deployment Script
# This script provides an easy way to deploy the modular infrastructure

set -e

# Default values
ENVIRONMENT="dev"
ACTION="plan"
AUTO_APPROVE=false
DESTROY=false
REGION="us-west-2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

MLOps Infrastructure Deployment Script

OPTIONS:
    -e, --environment ENVIRONMENT    Environment to deploy (dev, prod, experimental)
    -a, --action ACTION             Action to perform (plan, apply, destroy)
    -y, --auto-approve              Auto approve terraform apply/destroy
    -r, --region REGION             AWS region (default: us-west-2)
    -h, --help                      Show this help message

EXAMPLES:
    # Plan dev environment
    $0 --environment dev --action plan
    
    # Deploy prod environment
    $0 --environment prod --action apply --auto-approve
    
    # Destroy experimental environment
    $0 --environment experimental --action destroy --auto-approve

SUPPORTED MODULE COMBINATIONS:
    Data Pipeline: kafka, kinesis, pulsar
    ML Platform: mlflow, kubeflow, sagemaker
    Monitoring: prometheus, datadog, cloudwatch
    Serving: kubernetes, ecs, cloudrun

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -y|--auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|prod|experimental)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT"
    print_error "Supported environments: dev, prod, experimental"
    exit 1
fi

# Validate action
if [[ ! "$ACTION" =~ ^(plan|apply|destroy|validate|fmt|init)$ ]]; then
    print_error "Invalid action: $ACTION"
    print_error "Supported actions: plan, apply, destroy, validate, fmt, init"
    exit 1
fi

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    print_error "Terraform is not installed or not in PATH"
    exit 1
fi

# Check if AWS CLI is installed and configured
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed or not in PATH"
    exit 1
fi

# Verify AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured or invalid"
    exit 1
fi

# Set variables file
TFVARS_FILE="environments/${ENVIRONMENT}.tfvars"

# Check if tfvars file exists
if [[ ! -f "$TFVARS_FILE" ]]; then
    print_error "Environment file not found: $TFVARS_FILE"
    exit 1
fi

print_status "Starting deployment for environment: $ENVIRONMENT"
print_status "Action: $ACTION"
print_status "Region: $REGION"
print_status "Variables file: $TFVARS_FILE"

# Export AWS region
export AWS_DEFAULT_REGION=$REGION

# Function to validate module configuration
validate_modules() {
    print_status "Validating module configuration..."
    
    # Extract module configuration from tfvars file
    local data_pipeline_type=$(grep -A5 "data_pipeline" $TFVARS_FILE | grep "type" | cut -d'"' -f4)
    local ml_platform_type=$(grep -A5 "ml_platform" $TFVARS_FILE | grep "type" | cut -d'"' -f4)
    local monitoring_type=$(grep -A5 "monitoring" $TFVARS_FILE | grep "type" | cut -d'"' -f4)
    local serving_type=$(grep -A5 "serving" $TFVARS_FILE | grep "type" | cut -d'"' -f4)
    
    print_status "Selected modules:"
    print_status "  Data Pipeline: $data_pipeline_type"
    print_status "  ML Platform: $ml_platform_type"
    print_status "  Monitoring: $monitoring_type"
    print_status "  Serving: $serving_type"
    
    # Validate combinations
    if [[ "$serving_type" == "kubernetes" && "$ml_platform_type" == "kubeflow" ]]; then
        print_success "✓ Kubeflow on Kubernetes - compatible combination"
    fi
    
    if [[ "$data_pipeline_type" == "kafka" && "$ENVIRONMENT" == "prod" ]]; then
        print_success "✓ Kafka for production - recommended for high throughput"
    fi
    
    if [[ "$monitoring_type" == "prometheus" && "$serving_type" == "kubernetes" ]]; then
        print_success "✓ Prometheus on Kubernetes - native integration available"
    fi
}

# Function to estimate costs
estimate_costs() {
    print_status "Estimating infrastructure costs for $ENVIRONMENT environment..."
    
    case $ENVIRONMENT in
        "dev")
            print_status "Estimated monthly cost: $300-600"
            print_status "  - Optimized for development with spot instances"
            print_status "  - Minimal redundancy and smaller instance sizes"
            ;;
        "prod")
            print_status "Estimated monthly cost: $1200-2500"
            print_status "  - Production-grade with high availability"
            print_status "  - Larger instances and full redundancy"
            ;;
        "experimental")
            print_status "Estimated monthly cost: $400-800"
            print_status "  - Mixed configuration for testing"
            print_status "  - Spot instances where possible"
            ;;
    esac
}

# Function to create backend configuration
setup_backend() {
    print_status "Setting up Terraform backend..."
    
    # Get the S3 bucket name from tfvars
    local bucket=$(grep "terraform_state_bucket" $TFVARS_FILE | cut -d'"' -f4)
    
    if [[ -z "$bucket" ]]; then
        print_error "terraform_state_bucket not found in $TFVARS_FILE"
        exit 1
    fi
    
    # Check if bucket exists, create if not
    if ! aws s3 ls "s3://$bucket" &> /dev/null; then
        print_warning "S3 bucket $bucket does not exist. Creating..."
        aws s3 mb "s3://$bucket" --region $REGION
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket $bucket \
            --versioning-configuration Status=Enabled
        
        # Enable encryption
        aws s3api put-bucket-encryption \
            --bucket $bucket \
            --server-side-encryption-configuration '{
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            }'
        
        print_success "Created S3 bucket: $bucket"
    fi
}

# Function to run terraform commands
run_terraform() {
    local tf_action=$1
    
    case $tf_action in
        "init")
            print_status "Initializing Terraform..."
            terraform init
            ;;
        "validate")
            print_status "Validating Terraform configuration..."
            terraform validate
            ;;
        "fmt")
            print_status "Formatting Terraform files..."
            terraform fmt -recursive
            ;;
        "plan")
            print_status "Planning Terraform deployment..."
            terraform plan -var-file="$TFVARS_FILE" -out="$ENVIRONMENT.tfplan"
            ;;
        "apply")
            if [[ "$AUTO_APPROVE" == true ]]; then
                print_status "Applying Terraform configuration (auto-approved)..."
                terraform apply -var-file="$TFVARS_FILE" -auto-approve
            else
                print_status "Applying Terraform configuration..."
                if [[ -f "$ENVIRONMENT.tfplan" ]]; then
                    terraform apply "$ENVIRONMENT.tfplan"
                else
                    terraform apply -var-file="$TFVARS_FILE"
                fi
            fi
            ;;
        "destroy")
            if [[ "$AUTO_APPROVE" == true ]]; then
                print_warning "Destroying infrastructure (auto-approved)..."
                terraform destroy -var-file="$TFVARS_FILE" -auto-approve
            else
                print_warning "Destroying infrastructure..."
                terraform destroy -var-file="$TFVARS_FILE"
            fi
            ;;
    esac
}

# Function to show deployment summary
show_summary() {
    print_success "Deployment completed successfully!"
    print_status "Summary:"
    print_status "  Environment: $ENVIRONMENT"
    print_status "  Action: $ACTION"
    print_status "  Region: $REGION"
    
    if [[ "$ACTION" == "apply" ]]; then
        print_status "\nTo get connection information:"
        print_status "  terraform output -var-file=$TFVARS_FILE"
        
        print_status "\nTo connect to your resources:"
        print_status "  kubectl get nodes  # For EKS clusters"
        print_status "  aws eks update-kubeconfig --region $REGION --name <cluster-name>"
        
        print_status "\nTo view costs:"
        print_status "  Use AWS Cost Explorer or run:"
        print_status "  aws ce get-cost-and-usage --time-period Start=\$(date -d '1 month ago' '+%Y-%m-%d'),End=\$(date '+%Y-%m-%d') --granularity MONTHLY --metrics BlendedCost"
    fi
}

# Main execution
main() {
    print_status "MLOps Infrastructure Deployment"
    print_status "================================"
    
    # Validate configuration
    validate_modules
    
    # Estimate costs
    estimate_costs
    
    # Confirm with user for destructive actions
    if [[ "$ACTION" == "destroy" && "$AUTO_APPROVE" == false ]]; then
        echo
        read -p "Are you sure you want to destroy the $ENVIRONMENT environment? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            print_status "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Setup backend
    setup_backend
    
    # Run terraform init first
    run_terraform "init"
    
    # Run the requested action
    run_terraform "$ACTION"
    
    # Show summary
    if [[ $? -eq 0 ]]; then
        show_summary
    else
        print_error "Deployment failed!"
        exit 1
    fi
}

# Run main function
main