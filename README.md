# 🚀 Code Quality MLOps Pipeline

A comprehensive MLOps pipeline for automated code quality monitoring using machine learning to detect bugs, ensure code consistency, and manage technical debt.

## 🎯 Overview

This project implements a complete MLOps pipeline that demonstrates all 6 stages of machine learning operations:

1. **Data Ingestion** - Collect code data from GitHub repositories
2. **Data Processing** - Extract meaningful features from code commits
3. **Model Training** - Train bug prediction models with MLflow tracking
4. **Model Validation** - Comprehensive testing and validation framework
5. **Model Deployment** - Automated deployment to multiple environments
6. **Monitoring & Feedback** - Real-time monitoring and continuous improvement

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Pipeline  │    │   ML Platform   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • GitHub API    │    │ • MLflow        │    │ • Prometheus    │
│ • Kafka/Kinesis │────│ • Model Training│────│ • Drift Detection│
│ • Feature Eng.  │    │ • Validation    │    │ • A/B Testing   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Model Serving   │
                    │                 │
                    │ • FastAPI       │
                    │ • Kubernetes    │
                    │ • Auto-scaling  │
                    └─────────────────┘
```

## 🚦 Quick Start

### Prerequisites

- Python 3.9+
- Docker
- Kubernetes (optional)
- AWS CLI (for cloud deployment)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd code-quality-mlops
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example configurations
cp config/serving.json.example config/serving.json
cp config/deployment.json.example config/deployment.json

# Set environment variables
export GITHUB_TOKEN="your-github-token"
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

### 3. Start MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### 4. Run the Complete Pipeline

```bash
# Stage 1: Data Ingestion
python src/data_collection/github_ingestion.py \
  --repositories "facebook/react,microsoft/vscode" \
  --days 30

# Stage 2: Feature Engineering
python src/preprocessing/feature_engineering.py

# Stage 3: Model Training
python src/models/bug_predictor.py \
  --model-type hybrid \
  --tune-hyperparams

# Stage 4: Model Validation
python src/models/model_validation.py \
  --model-name hybrid_bug_predictor

# Stage 5: Model Deployment
./scripts/deploy-model.sh \
  --model-name hybrid_bug_predictor \
  --model-version 1 \
  --environment dev

# Stage 6: Start Monitoring
./scripts/start-monitoring.sh --mode full
```

## 📊 Pipeline Stages

### Stage 1: Data Ingestion

**Location**: `src/data_collection/github_ingestion.py`

Collects data from GitHub repositories including:
- Commit history and metadata
- Code changes and file content
- Issue reports and bug labels
- Author information and experience

**Key Features**:
- Rate-limited API calls
- Supports both Kafka and Kinesis pipelines
- Automatic bug label detection
- Configurable collection periods

```bash
# Example usage
python src/data_collection/github_ingestion.py \
  --repositories "facebook/react" \
  --days 30 \
  --pipeline-type kinesis
```

### Stage 2: Data Processing

**Location**: `src/preprocessing/feature_engineering.py`

Extracts 20+ features from code including:
- Code complexity metrics (cyclomatic, cognitive, Halstead)
- Code quality indicators (comments, documentation)
- Pattern-based features (test coverage, imports)
- Historical patterns (author experience, file changes)

**Key Features**:
- Real-time stream processing
- Comprehensive code analysis
- Feature validation and normalization
- Support for multiple programming languages

### Stage 3: Model Training

**Location**: `src/models/bug_predictor.py`

Implements two model architectures:
- **Simple Model**: RandomForest for baseline performance
- **Hybrid Model**: Transformer + traditional features for enhanced accuracy

**Key Features**:
- MLflow experiment tracking
- Hyperparameter optimization with Optuna
- Cross-validation and early stopping
- Model versioning and registry

```bash
# Train both models for comparison
python src/models/bug_predictor.py --model-type both --tune-hyperparams
```

### Stage 4: Model Validation

**Location**: `src/models/model_validation.py`

Comprehensive validation framework including:
- Performance metrics calculation
- Cross-validation analysis
- Data drift detection
- Model stability testing
- Bias and fairness analysis

**Key Features**:
- Automated validation pipelines
- Statistical significance testing
- Performance regression detection
- SHAP-based explainability

### Stage 5: Model Deployment

**Location**: `src/pipeline/deployment_manager.py`

Automated deployment with multiple strategies:
- **Rolling Deployment**: Zero-downtime updates
- **Blue-Green Deployment**: Instant rollback capability
- **Canary Deployment**: Gradual rollout with monitoring

**Supported Platforms**:
- Kubernetes with auto-scaling
- AWS ECS/Fargate
- AWS Lambda (serverless)

```bash
# Deploy with blue-green strategy
./scripts/deploy-model.sh \
  --model-name hybrid_bug_predictor \
  --model-version 2 \
  --environment prod \
  --strategy blue_green
```

### Stage 6: Monitoring & Feedback

**Location**: `src/monitoring/model_monitor.py`

Real-time monitoring system with:
- Performance metrics tracking
- Data drift detection using statistical tests
- A/B testing analysis
- Automated alerting (Slack, email)
- Business impact measurement

**Key Features**:
- Prometheus metrics integration
- Statistical drift detection (KS test, PSI, Wasserstein)
- A/B test significance testing
- Cost impact analysis

## 🔧 API Endpoints

The model serving API provides the following endpoints:

### Prediction Endpoint

```bash
POST /predict
```

```json
{
  "repository": "facebook/react",
  "lines_of_code": 150,
  "files_changed": 3,
  "additions": 45,
  "deletions": 12,
  "cyclomatic_complexity": 4.2,
  "commit_message": "Fix memory leak in component lifecycle"
}
```

**Response**:
```json
{
  "prediction": "bug_fix",
  "confidence": 0.87,
  "model_name": "hybrid_bug_predictor",
  "model_version": "2",
  "prediction_id": "react_1703123456",
  "timestamp": "2023-12-20T10:30:45Z"
}
```

### Health Check

```bash
GET /health
```

### Metrics

```bash
GET /metrics  # Prometheus metrics
```

### Model Management

```bash
GET /models                    # List active models
POST /models/{name}/reload     # Reload specific model
```

## 🚀 Infrastructure

### Modular Terraform Infrastructure

**Location**: `infrastructure/terraform/`

Supports multiple deployment configurations:

```bash
# Development environment (~$300-600/month)
./infrastructure/terraform/deploy.sh --environment dev

# Production environment (~$1200-2500/month)  
./infrastructure/terraform/deploy.sh --environment prod

# Use interactive module selector
./infrastructure/terraform/module-selector.py --interactive
```

**Available Modules**:
- **Data Pipeline**: Kafka (MSK), Kinesis, Pulsar
- **ML Platform**: MLflow, Kubeflow, SageMaker
- **Monitoring**: Prometheus, DataDog, CloudWatch
- **Serving**: Kubernetes, ECS/Fargate, Lambda

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/model-serving.yaml

# Check deployment status
kubectl get pods -n mlops-prod
kubectl get services -n mlops-prod
```

## 📈 Monitoring Dashboard

The monitoring system provides comprehensive insights:

### Performance Metrics
- Model accuracy, precision, recall, F1-score
- Prediction latency and throughput
- Error rates and availability

### Data Quality Metrics
- Data drift detection across features
- Feature distribution changes
- Missing value patterns

### Business Metrics
- Cost per prediction
- False positive/negative costs
- ROI from improved accuracy

### A/B Testing
- Statistical significance testing
- Confidence intervals
- Business impact analysis

## 🔍 Key Features

### 🤖 Advanced ML Models

- **Hybrid Architecture**: Combines transformer-based text analysis with traditional numerical features
- **Feature Engineering**: 20+ sophisticated code quality metrics
- **AutoML**: Automated hyperparameter tuning with Optuna
- **Model Validation**: Comprehensive testing including bias detection

### 🚀 Production-Ready Deployment

- **Multiple Strategies**: Rolling, blue-green, and canary deployments
- **Auto-scaling**: Kubernetes HPA based on CPU/memory metrics
- **Health Checks**: Readiness and liveness probes
- **A/B Testing**: Traffic splitting for model comparison

### 📊 Comprehensive Monitoring

- **Real-time Metrics**: Prometheus integration with custom metrics
- **Drift Detection**: Statistical tests (KS, PSI, Wasserstein distance)
- **Alerting**: Multi-channel notifications (Slack, email)
- **Business Intelligence**: Cost tracking and ROI analysis

### 🔧 Developer Experience

- **Easy Setup**: One-command pipeline execution
- **Interactive Tools**: Module selector for infrastructure
- **Comprehensive Logging**: Structured logging throughout
- **Documentation**: Detailed guides and examples

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### End-to-End Tests
```bash
python -m pytest tests/e2e/
```

### Model Validation
```bash
# Validate specific model
python src/models/model_validation.py \
  --model-name hybrid_bug_predictor \
  --generate-report
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Architecture Guide](docs/architecture.md)** - System design and components
- **[ML Models Guide](docs/ml-models.md)** - Model architectures and training
- **[Pipeline Implementation](docs/pipeline-implementation.md)** - End-to-end pipeline
- **[Cost Optimization](docs/cost-optimization.md)** - Cost management strategies
- **[Monitoring Guide](docs/monitoring-feedback.md)** - Monitoring and alerting

## 💰 Cost Optimization

The pipeline supports three cost tiers:

### Development ($300-600/month)
- Kinesis with minimal shards
- Self-hosted MLflow
- Prometheus monitoring
- Lambda serving

### Production ($1200-2500/month)
- MSK with 3 brokers
- MLflow with HA setup
- Full Prometheus stack
- Kubernetes with auto-scaling

### Enterprise ($2500-5000/month)
- Multi-AZ MSK
- SageMaker integration
- DataDog monitoring
- GPU support

## 🔐 Security

- **Network Security**: VPC with private subnets, security groups
- **Encryption**: KMS encryption for data at rest and in transit
- **Access Control**: IAM roles with least privilege
- **Compliance**: SOC2, GDPR-ready configurations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See the `docs/` folder
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions

## 🗺️ Roadmap

- [ ] **Multi-cloud Support**: Azure and GCP deployment options
- [ ] **Enhanced ML Models**: Support for more programming languages
- [ ] **Real-time Training**: Online learning capabilities
- [ ] **Advanced Analytics**: Deeper business intelligence features
- [ ] **Enterprise Features**: SSO, RBAC, audit logging

---

## 🎉 Getting Started Checklist

- [ ] Install prerequisites (Python, Docker, etc.)
- [ ] Configure environment variables
- [ ] Start MLflow server
- [ ] Run data ingestion pipeline
- [ ] Train your first model
- [ ] Deploy model to development
- [ ] Set up monitoring
- [ ] Review monitoring dashboard
- [ ] Deploy to production

**Ready to revolutionize your code quality with ML?** Start with the Quick Start guide above! 🚀