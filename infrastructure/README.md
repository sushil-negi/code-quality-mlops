# 🚀 Modular MLOps Infrastructure

This directory contains a **fully modular Terraform infrastructure** for the Code Quality MLOps pipeline. The architecture is designed for **maximum flexibility** - you can easily swap in and out different tools based on your project needs, budget, and technical requirements.

## 🎯 Key Features

- **🔄 Modular Design**: Swap between Kafka/Kinesis, MLflow/Kubeflow, Prometheus/DataDog, etc.
- **💰 Cost Optimization**: 3 pre-configured deployment tiers from $300-$3000/month
- **🛡️ Production Ready**: Security, monitoring, and compliance built-in
- **📊 Smart Selection**: Interactive tool to recommend optimal module combinations
- **⚡ One-Click Deploy**: Automated deployment scripts with validation

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │    │   ML Platform   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Kafka (MSK)   │    │ • MLflow        │    │ • Prometheus    │
│ • Kinesis       │────│ • Kubeflow      │────│ • DataDog       │
│ • Pulsar        │    │ • SageMaker     │    │ • CloudWatch    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Serving Layer   │
                    │                 │
                    │ • Kubernetes    │
                    │ • ECS/Fargate   │
                    │ • Lambda        │
                    └─────────────────┘
```

## 🚦 Quick Start

### 1. Interactive Module Selection (Recommended)

```bash
cd infrastructure/terraform
./module-selector.py --interactive
```

This wizard will:
- Ask about your use case (startup/enterprise/research/production)
- Understand your budget and complexity tolerance
- Recommend optimal module combinations
- Generate a custom configuration file

### 2. Use Pre-configured Environments

```bash
# Development environment (Cost-optimized: ~$300-600/month)
./deploy.sh --environment dev --action plan

# Production environment (Performance-optimized: ~$1200-2500/month)
./deploy.sh --environment prod --action apply --auto-approve

# Experimental environment (Mixed tools for testing)
./deploy.sh --environment experimental --action plan
```

### 3. Custom Configuration

Copy and modify an existing environment file:

```bash
cp environments/dev.tfvars environments/my-custom.tfvars
# Edit the modules section to your needs
./deploy.sh --environment my-custom --action plan
```

## 🔧 Available Modules

### Data Pipeline Options

| Module | Cost | Complexity | Best For |
|--------|------|------------|----------|
| **Kafka (MSK)** | 💰💰💰 | Medium | High throughput, Production |
| **Kinesis** | 💰💰 | Low | AWS-native, Serverless |
| **Pulsar** | 💰💰💰 | High | Multi-tenant, Cloud-native |

### ML Platform Options

| Module | Cost | Complexity | Best For |
|--------|------|------------|----------|
| **MLflow** | 💰 | Low | Cost-conscious, Flexibility |
| **Kubeflow** | 💰💰 | High | Kubernetes, Workflows |
| **SageMaker** | 💰💰💰💰 | Low | Managed service, Enterprise |

### Monitoring Options

| Module | Cost | Complexity | Best For |
|--------|------|------------|----------|
| **Prometheus** | 💰 | Medium | Cost-effective, Customizable |
| **DataDog** | 💰💰💰💰💰 | Low | Enterprise, Easy setup |
| **CloudWatch** | 💰💰 | Low | AWS-native, Basic monitoring |

### Serving Options

| Module | Cost | Complexity | Best For |
|--------|------|------------|----------|
| **Kubernetes** | 💰💰💰 | High | Scalability, Flexibility |
| **ECS/Fargate** | 💰💰 | Medium | AWS-native, Containers |
| **Lambda** | 💰 | Low | Serverless, Event-driven |

## 📋 Configuration Examples

### Startup Configuration (Cost-optimized)
```hcl
modules = {
  data_pipeline = {
    enabled = true
    type    = "kinesis"  # Serverless, pay-as-you-go
  }
  ml_platform = {
    enabled = true
    type    = "mlflow"   # Open source, low cost
  }
  monitoring = {
    enabled = true
    type    = "prometheus"  # Open source monitoring
  }
  serving = {
    enabled = true
    type    = "lambda"   # Serverless serving
  }
}
```

### Enterprise Configuration (Feature-rich)
```hcl
modules = {
  data_pipeline = {
    enabled = true
    type    = "kafka"    # High throughput, reliable
  }
  ml_platform = {
    enabled = true
    type    = "sagemaker"  # Fully managed, enterprise features
  }
  monitoring = {
    enabled = true
    type    = "datadog"  # Enterprise monitoring, APM
  }
  serving = {
    enabled = true
    type    = "kubernetes"  # Scalable, flexible
  }
}
```

### Research Configuration (Experimental)
```hcl
modules = {
  data_pipeline = {
    enabled = true
    type    = "pulsar"   # Cutting-edge messaging
  }
  ml_platform = {
    enabled = true
    type    = "kubeflow"  # Workflow orchestration
  }
  monitoring = {
    enabled = true
    type    = "prometheus"  # Customizable monitoring
  }
  serving = {
    enabled = true
    type    = "kubernetes"  # Flexible deployment
  }
}
```

## 💰 Cost Breakdown

### Development Environment (~$300-600/month)
- **Compute**: t3.small instances, spot pricing
- **Data Pipeline**: Kinesis with minimal shards
- **ML Platform**: Self-hosted MLflow
- **Monitoring**: Prometheus + Grafana
- **Storage**: Minimal retention periods

### Production Environment (~$1200-2500/month)
- **Compute**: m5.large+ instances, on-demand + spot mix
- **Data Pipeline**: MSK with 3 brokers
- **ML Platform**: MLflow with HA setup
- **Monitoring**: Full Prometheus stack with alerting
- **Storage**: Production retention policies

### Enterprise Environment (~$2500-5000/month)
- **Compute**: Large instances, GPU support
- **Data Pipeline**: Multi-AZ MSK or managed Kinesis
- **ML Platform**: SageMaker with all features
- **Monitoring**: DataDog with full APM
- **Storage**: Long-term retention and compliance

## 🔨 Deployment Commands

```bash
# Initialize and validate
./deploy.sh --environment dev --action init
./deploy.sh --environment dev --action validate

# Plan deployment
./deploy.sh --environment dev --action plan

# Deploy infrastructure
./deploy.sh --environment dev --action apply

# Auto-approve for CI/CD
./deploy.sh --environment prod --action apply --auto-approve

# Destroy environment
./deploy.sh --environment dev --action destroy --auto-approve
```

## 🎛️ Module Selector Tool

The interactive module selector helps you choose the right combination:

```bash
# Run interactive wizard
./module-selector.py --interactive

# List all available modules
./module-selector.py --list-modules

# Compare specific modules
./module-selector.py --compare data_pipeline kafka,kinesis,pulsar
```

### Sample Wizard Flow
```
🚀 MLOps Module Selector
==================================================

📋 What's your primary use case?
  1. Startup - Budget: low, Complexity: low
  2. Enterprise - Budget: high, Complexity: medium
  3. Research - Budget: medium, Complexity: high
  4. Production - Budget: medium-high, Complexity: medium

💰 What's your monthly budget range?
  1. $300-800 - Optimized for cost
  2. $800-1500 - Balanced cost/performance
  3. $1500-3000 - Performance focused
  4. $3000+ - Enterprise grade

🎯 Recommended Configuration
==================================================

📦 Data Pipeline: AWS Kinesis
   Managed streaming service with serverless scaling
   Cost factor: 1.0x
   ✅ Pros: Fully managed, Auto-scaling, AWS integration

📦 ML Platform: MLflow
   Open source ML lifecycle management
   Cost factor: 0.5x
   ✅ Pros: Open source, Language agnostic, Simple setup

💰 Estimated Monthly Cost: $450
```

## 🔒 Security Features

- **Encryption**: KMS encryption for all data at rest and in transit
- **Network Security**: VPC with private subnets, security groups
- **Access Control**: IAM roles with least privilege
- **Monitoring**: CloudTrail, VPC Flow Logs
- **Compliance**: SOC2, GDPR-ready configurations

## 📊 Monitoring & Observability

All configurations include comprehensive monitoring:

- **Infrastructure Metrics**: CPU, memory, disk, network
- **Application Metrics**: Request rates, latencies, errors
- **ML Metrics**: Model performance, drift detection
- **Business Metrics**: Cost tracking, usage analytics
- **Alerting**: Configurable thresholds and notifications

## 🔄 Migration Between Modules

Easily migrate between different modules:

```bash
# Current: Kinesis + MLflow
# Target: Kafka + Kubeflow

# 1. Update configuration
sed -i 's/kinesis/kafka/g' environments/prod.tfvars
sed -i 's/mlflow/kubeflow/g' environments/prod.tfvars

# 2. Plan migration
./deploy.sh --environment prod --action plan

# 3. Execute migration (with data backup)
./deploy.sh --environment prod --action apply
```

## 🚀 Getting Started Checklist

- [ ] Install Prerequisites (Terraform, AWS CLI, Python)
- [ ] Configure AWS credentials (`aws configure`)
- [ ] Create S3 bucket for Terraform state
- [ ] Run module selector wizard
- [ ] Review generated configuration
- [ ] Deploy with `./deploy.sh`
- [ ] Verify deployment with `terraform output`
- [ ] Test application connectivity
- [ ] Set up monitoring dashboards

## 🆘 Troubleshooting

### Common Issues

**1. Terraform State Bucket Not Found**
```bash
aws s3 mb s3://your-terraform-state-bucket --region us-west-2
```

**2. EKS Cluster Access Denied**
```bash
aws eks update-kubeconfig --region us-west-2 --name your-cluster-name
```

**3. Cost Alerts**
```bash
# Check current costs
aws ce get-cost-and-usage --time-period Start=$(date -d '1 month ago' '+%Y-%m-%d'),End=$(date '+%Y-%m-%d') --granularity MONTHLY --metrics BlendedCost
```

### Support Channels

- **Documentation**: See `/docs` folder for detailed guides
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions

## 🔮 Roadmap

- [ ] **GCP Support**: Add Google Cloud module options
- [ ] **Azure Support**: Add Microsoft Azure module options
- [ ] **Cost Optimization**: Automated right-sizing recommendations
- [ ] **Multi-Region**: Cross-region deployment support
- [ ] **Backup/Restore**: Automated backup and disaster recovery
- [ ] **Compliance**: Additional compliance frameworks (HIPAA, PCI)

---

## 📄 License

This infrastructure code is part of the Code Quality MLOps project. See the main repository for license details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with multiple module combinations
4. Submit a pull request with detailed description

---

> **💡 Pro Tip**: Start with the development environment to test the pipeline, then use the interactive selector to optimize for your production needs!