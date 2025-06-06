# Code Quality MLOps Architecture

## High-Level Architecture Flow

```mermaid
graph TB
    subgraph "Data Sources"
        A[Git Repositories] --> B[Code Collection Agent]
        C[Issue Trackers] --> B
        D[CI/CD Logs] --> B
        E[Code Review Data] --> B
    end
    
    subgraph "Data Pipeline"
        B --> F[Apache Kafka]
        F --> G[Data Validation]
        G --> H[Feature Engineering]
        H --> I[Feature Store]
    end
    
    subgraph "ML Pipeline"
        I --> J[Model Training]
        J --> K[Model Validation]
        K --> L[Model Registry MLflow]
        L --> M[Model Serving]
    end
    
    subgraph "Inference & Monitoring"
        M --> N[Real-time API]
        N --> O[Code Quality Predictions]
        O --> P[Alert System]
        O --> Q[Dashboard]
        O --> R[IDE Plugins]
    end
    
    subgraph "Feedback Loop"
        P --> S[Human Review]
        S --> T[Label Correction]
        T --> I
    end
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "Data Collection Layer"
        GH[GitHub/GitLab Webhooks] --> KF1[Kafka Topic: raw-code]
        ST[Static Analysis Tools] --> KF1
        PR[Pull Request Events] --> KF1
    end
    
    subgraph "Processing Layer"
        KF1 --> AP[Airflow Pipeline]
        AP --> FE[Feature Extractors]
        FE --> FS[Feature Store]
    end
    
    subgraph "ML Training"
        FS --> DL[DataLoader]
        DL --> MT[Model Trainer]
        MT --> MV[Model Validator]
        MV --> MR[MLflow Registry]
    end
    
    subgraph "Serving Layer"
        MR --> TS[TorchServe]
        TS --> API[FastAPI Gateway]
        API --> CACHE[Redis Cache]
    end
    
    subgraph "Monitoring"
        API --> PROM[Prometheus]
        PROM --> GRAF[Grafana]
        API --> ELK[ELK Stack]
    end
```

## Data Flow Pipeline

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Git as Git Repository
    participant Hook as Webhook
    participant Kafka as Kafka Stream
    participant Airflow as Airflow DAG
    participant Feature as Feature Engineering
    participant Train as ML Training
    participant Serve as Model Serving
    participant API as Inference API
    participant IDE as IDE Plugin
    
    Dev->>Git: Push Code
    Git->>Hook: Trigger Webhook
    Hook->>Kafka: Send Code Event
    Kafka->>Airflow: Consume Event
    Airflow->>Feature: Extract Features
    Feature->>Train: Training Data
    Train->>Serve: Deploy Model
    Dev->>API: Request Analysis
    API->>Serve: Get Prediction
    Serve->>API: Quality Metrics
    API->>IDE: Display Results
```

## Feature Engineering Pipeline

```mermaid
graph TD
    A[Raw Code] --> B[AST Parser]
    B --> C[Complexity Metrics]
    B --> D[Code Patterns]
    B --> E[Dependency Graph]
    
    C --> F[Cyclomatic Complexity]
    C --> G[Cognitive Complexity]
    C --> H[Halstead Metrics]
    
    D --> I[Design Patterns]
    D --> J[Anti-patterns]
    D --> K[Code Smells]
    
    E --> L[Module Coupling]
    E --> M[Cohesion Metrics]
    
    F --> N[Feature Vector]
    G --> N
    H --> N
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
```

## Detailed Tool-Based Pipeline Architecture

```mermaid
graph TB
    subgraph "Data Collection Tools"
        GH[GitHub API/Webhooks] --> KProd[Kafka Producer]
        GL[GitLab API] --> KProd
        BB[Bitbucket API] --> KProd
        JIRA[JIRA REST API] --> KProd
        SQ[SonarQube Scanner] --> KProd
    end
    
    subgraph "Stream Processing"
        KProd --> KC[Kafka Cluster<br/>3 Brokers, RF=3]
        KC --> KS[Kafka Streams<br/>Data Validation]
        KS --> KC2[Validated Topics]
    end
    
    subgraph "Orchestration Layer"
        KC2 --> AF[Apache Airflow<br/>DAG Scheduler]
        AF --> SP[Spark Processing<br/>Feature Engineering]
        SP --> FS[Feature Store<br/>Feast/Tecton]
    end
    
    subgraph "ML Training Infrastructure"
        FS --> KUB[Kubernetes Cluster]
        KUB --> GPU[GPU Nodes<br/>NVIDIA A100]
        GPU --> MLF[MLflow<br/>Experiment Tracking]
        MLF --> MR[Model Registry<br/>Version Control]
    end
    
    subgraph "Model Serving"
        MR --> TS[TorchServe<br/>Model Server]
        MR --> TF[TensorFlow Serving]
        TS --> LB[Load Balancer<br/>NGINX]
        TF --> LB
    end
    
    subgraph "API Gateway"
        LB --> FAPI[FastAPI<br/>REST Endpoints]
        FAPI --> RED[Redis Cache<br/>Response Cache]
        FAPI --> AUTH[Auth Service<br/>OAuth2/JWT]
    end
    
    subgraph "Client Integration"
        FAPI --> VSC[VS Code Extension]
        FAPI --> IJ[IntelliJ Plugin]
        FAPI --> CLI[CLI Tool]
        FAPI --> WEB[Web Dashboard]
    end
    
    subgraph "Monitoring Stack"
        FAPI --> PROM[Prometheus<br/>Metrics Collection]
        TS --> PROM
        PROM --> GRAF[Grafana<br/>Dashboards]
        FAPI --> ELK[ELK Stack<br/>Log Analysis]
        GRAF --> PG[PagerDuty<br/>Alerting]
    end
```

## Cost-Optimized Architecture Views

### Premium Architecture ($2,895/month)
```mermaid
graph LR
    subgraph "AWS Infrastructure"
        subgraph "Compute"
            EKS[EKS Cluster<br/>3x m5.2xlarge masters<br/>5x m5.4xlarge workers<br/>2x g4dn.2xlarge GPU]
        end
        
        subgraph "Storage"
            S3[S3 Buckets<br/>10TB Data Lake<br/>Lifecycle Policies]
            EBS[EBS Volumes<br/>500GB SSD per node]
        end
        
        subgraph "Networking"
            ALB[Application Load Balancer<br/>Multi-AZ]
            CF[CloudFront CDN<br/>Global Distribution]
        end
        
        subgraph "Data Services"
            MSK[MSK Kafka<br/>3 brokers, m5.large]
            RDS[RDS PostgreSQL<br/>Multi-AZ, db.r5.large]
            EC[ElastiCache Redis<br/>3 nodes, cache.r6g.large]
        end
    end
```

### Cost-Optimized Architecture ($1,250/month)
```mermaid
graph LR
    subgraph "Mixed Cloud Strategy"
        subgraph "AWS Core Services"
            EKS2[EKS Cluster<br/>1x t3.large master<br/>3x t3.xlarge workers<br/>Spot Instances 80%]
            S3_2[S3 + Glacier<br/>Intelligent Tiering]
        end
        
        subgraph "Self-Managed Services"
            K8S[Self-managed K8s<br/>on EC2 Spot]
            KAFKA[Kafka on K8s<br/>Persistent Volumes]
            PG[PostgreSQL on K8s<br/>with Backups]
        end
        
        subgraph "Serverless Components"
            LAMBDA[Lambda Functions<br/>Data Processing]
            FARGATE[Fargate Tasks<br/>Batch Training]
        end
    end
```

### Startup/MVP Architecture ($300/month)
```mermaid
graph LR
    subgraph "Minimal Infrastructure"
        subgraph "Single Node Setup"
            EC2[EC2 t3.xlarge<br/>Docker Compose<br/>All Services]
        end
        
        subgraph "Managed Services"
            S3_MIN[S3 Bucket<br/>100GB Storage]
            CF_FREE[CloudFlare Free<br/>CDN & DDoS]
        end
        
        subgraph "Open Source Stack"
            OSS[PostgreSQL<br/>Redis<br/>Kafka Single Broker<br/>MLflow<br/>FastAPI]
        end
    end
```

## ML Model Training Pipeline

```mermaid
graph TD
    subgraph "Data Preparation"
        RC[Raw Code Files] --> DP[Data Preprocessor]
        DP --> TV[Train/Val/Test Split<br/>70/15/15]
        TV --> AUG[Data Augmentation<br/>Code Mutations]
    end
    
    subgraph "Feature Engineering"
        AUG --> FE1[Static Features<br/>AST, Complexity]
        AUG --> FE2[Semantic Features<br/>Code Embeddings]
        AUG --> FE3[Graph Features<br/>Call Graphs]
        FE1 --> FM[Feature Merger]
        FE2 --> FM
        FE3 --> FM
    end
    
    subgraph "Model Training"
        FM --> HYP[Hyperparameter<br/>Optimization<br/>Optuna]
        HYP --> TR[Distributed Training<br/>PyTorch DDP]
        TR --> VAL[Validation<br/>Cross-validation]
        VAL --> TEST[Test Evaluation<br/>Hold-out Set]
    end
    
    subgraph "Model Selection"
        TEST --> COMP[Model Comparison<br/>A/B Testing]
        COMP --> SEL[Best Model<br/>Selection]
        SEL --> VER[Model Versioning<br/>Git LFS]
        VER --> REG[Registry Upload<br/>MLflow]
    end
```

## Real-time Inference Pipeline

```mermaid
sequenceDiagram
    participant IDE as IDE Plugin
    participant LB as Load Balancer
    participant API as FastAPI
    participant Cache as Redis Cache
    participant Auth as Auth Service
    participant Queue as Request Queue
    participant Model as Model Server
    participant Monitor as Monitoring
    
    IDE->>LB: POST /analyze/code
    LB->>API: Forward Request
    API->>Auth: Validate Token
    Auth->>API: Token Valid
    API->>Cache: Check Cache
    
    alt Cache Hit
        Cache->>API: Return Cached Result
        API->>IDE: Return Predictions
    else Cache Miss
        API->>Queue: Queue Request
        Queue->>Model: Process Request
        Model->>Model: Load Code Features
        Model->>Model: Run Inference
        Model->>API: Return Predictions
        API->>Cache: Store in Cache
        API->>Monitor: Log Metrics
        API->>IDE: Return Predictions
    end
    
    Monitor->>Monitor: Track Latency
    Monitor->>Monitor: Track Accuracy
```

## Continuous Learning Pipeline

```mermaid
graph TD
    subgraph "Feedback Collection"
        USER[User Feedback] --> FB[Feedback API]
        PROD[Production Metrics] --> FB
        GIT[Git Commit Results] --> FB
    end
    
    subgraph "Data Processing"
        FB --> VAL[Validation Service]
        VAL --> LABEL[Label Aggregation<br/>Majority Vote]
        LABEL --> STORE[Feedback Store<br/>PostgreSQL]
    end
    
    subgraph "Model Retraining"
        STORE --> DRIFT[Drift Detection<br/>Statistical Tests]
        DRIFT --> TRIGGER{Retrain<br/>Needed?}
        TRIGGER -->|Yes| RETRAIN[Automated<br/>Retraining]
        TRIGGER -->|No| MONITOR[Continue<br/>Monitoring]
        RETRAIN --> NEWDATA[Merge with<br/>New Data]
        NEWDATA --> TRAIN[Training<br/>Pipeline]
    end
    
    subgraph "Deployment"
        TRAIN --> AB[A/B Testing<br/>Canary Deploy]
        AB --> EVAL[Performance<br/>Evaluation]
        EVAL --> PROMOTE{Better<br/>Performance?}
        PROMOTE -->|Yes| DEPLOY[Full<br/>Deployment]
        PROMOTE -->|No| ROLLBACK[Keep<br/>Current]
    end
```

## Security Architecture

```mermaid
graph TD
    subgraph "Security Layers"
        subgraph "Network Security"
            WAF[Web Application Firewall]
            VPC[VPC with Private Subnets]
            SG[Security Groups]
            NACL[Network ACLs]
        end
        
        subgraph "Application Security"
            OAUTH[OAuth2 Authentication]
            JWT[JWT Token Validation]
            RBAC[Role-Based Access Control]
            SECRETS[Secrets Manager<br/>AWS/HashiCorp Vault]
        end
        
        subgraph "Data Security"
            ENCRYPT[Encryption at Rest<br/>AES-256]
            TLS[TLS 1.3 in Transit]
            ANONYMIZE[PII Anonymization]
            AUDIT[Audit Logging]
        end
        
        subgraph "Compliance"
            GDPR[GDPR Compliance]
            SOC2[SOC2 Controls]
            ISO[ISO 27001]
        end
    end
```