# ML Models for Code Quality Prediction

## 1. Bug Prediction Model

### Architecture: Code-BERT + Classification Head

```python
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class BugPredictionModel(nn.Module):
    def __init__(self, pretrained_model='microsoft/codebert-base'):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: bug/no-bug
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
```

### Features Used:
- Code tokens and AST structure
- Historical bug patterns
- Code complexity metrics
- Developer experience signals
- Code review comments

### Training Data:
```yaml
dataset:
  positive_samples:
    - Buggy code from issue trackers
    - Code before bug-fix commits
    - Security vulnerability examples
  negative_samples:
    - Stable production code
    - Well-tested modules
    - Code after bug fixes
  
  size: 1M+ code samples
  languages: Python, Java, JavaScript, Go
  balance: 30% positive, 70% negative
```

## 2. Code Complexity Analyzer

### Architecture: Graph Neural Network (GNN)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ComplexityGNN(nn.Module):
    def __init__(self, node_features=128, hidden_dim=256):
        super().__init__()
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 128)
        
        self.complexity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Complexity score
        )
        
    def forward(self, x, edge_index, batch):
        # Node embedding
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Complexity prediction
        complexity = self.complexity_head(x)
        return complexity
```

### Graph Construction:
- Nodes: Functions, classes, modules
- Edges: Function calls, imports, inheritance
- Node features: Cyclomatic complexity, LOC, parameters

## 3. Security Vulnerability Scanner

### Architecture: CNN-LSTM Hybrid

```python
class SecurityScanner(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # CNN for local pattern detection
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        self.conv3 = nn.Conv1d(embedding_dim, 128, kernel_size=7)
        
        # LSTM for sequential patterns
        self.lstm = nn.LSTM(384, 256, num_layers=2, 
                           bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, num_heads=8)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # 10 vulnerability categories
        )
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x_t = x.transpose(1, 2)
        
        # CNN features
        conv1_out = F.relu(self.conv1(x_t))
        conv2_out = F.relu(self.conv2(x_t))
        conv3_out = F.relu(self.conv3(x_t))
        
        # Combine CNN outputs
        cnn_out = torch.cat([
            F.max_pool1d(conv1_out, conv1_out.size(2)),
            F.max_pool1d(conv2_out, conv2_out.size(2)),
            F.max_pool1d(conv3_out, conv3_out.size(2))
        ], dim=1).transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits
```

### Vulnerability Categories:
1. SQL Injection
2. XSS (Cross-Site Scripting)
3. Buffer Overflow
4. Path Traversal
5. Insecure Deserialization
6. Hardcoded Credentials
7. Race Conditions
8. Memory Leaks
9. Cryptographic Weaknesses
10. Access Control Issues

## 4. Technical Debt Predictor

### Architecture: Ensemble Model

```python
class TechnicalDebtEnsemble:
    def __init__(self):
        self.models = {
            'complexity': XGBRegressor(n_estimators=100),
            'maintainability': RandomForestRegressor(n_estimators=200),
            'test_coverage': GradientBoostingRegressor(),
            'code_smells': LGBMRegressor()
        }
        self.meta_model = LinearRegression()
        
    def train(self, X, y):
        # Train base models
        predictions = []
        for name, model in self.models.items():
            model.fit(X[name], y)
            pred = model.predict(X[name])
            predictions.append(pred.reshape(-1, 1))
        
        # Train meta-model
        meta_features = np.hstack(predictions)
        self.meta_model.fit(meta_features, y)
        
    def predict(self, X):
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X[name])
            predictions.append(pred.reshape(-1, 1))
        
        meta_features = np.hstack(predictions)
        return self.meta_model.predict(meta_features)
```

### Feature Categories:
```yaml
complexity_features:
  - cyclomatic_complexity
  - cognitive_complexity
  - nesting_depth
  - parameter_count
  - line_count

maintainability_features:
  - code_duplication_ratio
  - coupling_between_objects
  - depth_of_inheritance
  - lack_of_cohesion
  - response_for_class

test_coverage_features:
  - line_coverage
  - branch_coverage
  - mutation_score
  - test_to_code_ratio
  - test_execution_time

code_smell_features:
  - long_method_count
  - large_class_count
  - feature_envy_score
  - data_clumps
  - primitive_obsession
```

## Model Training Pipeline

```python
# Training configuration
training_config = {
    'bug_prediction': {
        'batch_size': 32,
        'learning_rate': 2e-5,
        'epochs': 10,
        'warmup_steps': 1000,
        'gradient_accumulation': 4,
        'fp16': True
    },
    'complexity_analyzer': {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 50,
        'early_stopping_patience': 5
    },
    'security_scanner': {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 30,
        'class_weights': 'balanced'
    },
    'tech_debt_predictor': {
        'cv_folds': 5,
        'hyperparameter_tuning': True,
        'optimization_metric': 'rmse'
    }
}
```

## Model Performance Metrics

```yaml
bug_prediction:
  precision: 0.89
  recall: 0.84
  f1_score: 0.86
  auc_roc: 0.92

complexity_analyzer:
  mae: 2.3
  rmse: 3.1
  r2_score: 0.87
  
security_scanner:
  accuracy: 0.94
  macro_f1: 0.91
  weighted_f1: 0.93
  
technical_debt:
  correlation: 0.85
  mae: 0.12
  explained_variance: 0.82
```

## Detailed Model Architecture Diagrams

### Bug Prediction Model Architecture
```mermaid
graph TD
    subgraph "Input Processing"
        CODE[Source Code] --> TOK[Tokenizer<br/>CodeBERT]
        TOK --> EMB[Token Embeddings<br/>768-dim]
        TOK --> MASK[Attention Mask]
    end
    
    subgraph "Transformer Layers"
        EMB --> L1[Layer 1<br/>Self-Attention + FFN]
        MASK --> L1
        L1 --> L2[Layer 2<br/>Self-Attention + FFN]
        L2 --> L12[...<br/>Layer 12]
        L12 --> POOL[Pooler Output<br/>768-dim]
    end
    
    subgraph "Classification Head"
        POOL --> DROP1[Dropout<br/>p=0.3]
        DROP1 --> FC1[Linear<br/>768→256]
        FC1 --> RELU1[ReLU]
        RELU1 --> DROP2[Dropout<br/>p=0.2]
        DROP2 --> FC2[Linear<br/>256→64]
        FC2 --> RELU2[ReLU]
        RELU2 --> FC3[Linear<br/>64→2]
        FC3 --> SOFT[Softmax]
        SOFT --> PRED[Bug/No-Bug]
    end
```

### Code Complexity GNN Architecture
```mermaid
graph TD
    subgraph "Graph Construction"
        AST[Abstract Syntax Tree] --> NODES[Node Features<br/>Functions, Classes]
        AST --> EDGES[Edge Relations<br/>Calls, Imports]
        NODES --> GRAPH[Code Graph]
        EDGES --> GRAPH
    end
    
    subgraph "GNN Layers"
        GRAPH --> ENC[Node Encoder<br/>128→256]
        ENC --> GCN1[GCN Layer 1<br/>256→256]
        GCN1 --> ACT1[ReLU + Dropout]
        ACT1 --> GCN2[GCN Layer 2<br/>256→256]
        GCN2 --> ACT2[ReLU]
        ACT2 --> GCN3[GCN Layer 3<br/>256→128]
    end
    
    subgraph "Aggregation"
        GCN3 --> POOL[Global Mean Pool]
        POOL --> HEAD[Complexity Head<br/>128→64→1]
        HEAD --> SCORE[Complexity Score]
    end
```

### Security Scanner CNN-LSTM Architecture
```mermaid
graph TD
    subgraph "Input Encoding"
        CODE[Code Tokens] --> EMB[Embedding Layer<br/>50K vocab → 256-dim]
    end
    
    subgraph "CNN Feature Extraction"
        EMB --> CNN1[Conv1D<br/>k=3, 128 filters]
        EMB --> CNN2[Conv1D<br/>k=5, 128 filters]
        EMB --> CNN3[Conv1D<br/>k=7, 128 filters]
        CNN1 --> POOL1[Max Pool]
        CNN2 --> POOL2[Max Pool]
        CNN3 --> POOL3[Max Pool]
        POOL1 --> CONCAT[Concatenate<br/>384-dim]
        POOL2 --> CONCAT
        POOL3 --> CONCAT
    end
    
    subgraph "Sequential Processing"
        CONCAT --> LSTM1[Bi-LSTM Layer 1<br/>256 units]
        LSTM1 --> LSTM2[Bi-LSTM Layer 2<br/>256 units]
        LSTM2 --> ATTN[Multi-Head Attention<br/>8 heads, 512-dim]
    end
    
    subgraph "Classification"
        ATTN --> MEAN[Mean Pooling]
        MEAN --> FC1[Linear<br/>512→256]
        FC1 --> DROP[Dropout 0.3]
        DROP --> FC2[Linear<br/>256→10]
        FC2 --> VULN[Vulnerability Classes]
    end
```

### Technical Debt Ensemble Architecture
```mermaid
graph TD
    subgraph "Feature Extraction"
        CODE[Source Code] --> FE1[Complexity<br/>Features]
        CODE --> FE2[Maintainability<br/>Features]
        CODE --> FE3[Test Coverage<br/>Features]
        CODE --> FE4[Code Smell<br/>Features]
    end
    
    subgraph "Base Models"
        FE1 --> XGB[XGBoost<br/>100 trees]
        FE2 --> RF[Random Forest<br/>200 trees]
        FE3 --> GB[Gradient Boost]
        FE4 --> LGBM[LightGBM]
    end
    
    subgraph "Meta Learning"
        XGB --> PRED1[Prediction 1]
        RF --> PRED2[Prediction 2]
        GB --> PRED3[Prediction 3]
        LGBM --> PRED4[Prediction 4]
        PRED1 --> META[Meta Features]
        PRED2 --> META
        PRED3 --> META
        PRED4 --> META
        META --> LR[Linear Regression<br/>Meta Model]
        LR --> FINAL[Tech Debt Score]
    end
```

## Training Pipeline Flow

```mermaid
graph LR
    subgraph "Data Collection"
        GIT[Git Repositories] --> CRAWLER[Code Crawler]
        ISSUES[Issue Trackers] --> CRAWLER
        CRAWLER --> RAW[Raw Dataset]
    end
    
    subgraph "Data Preprocessing"
        RAW --> CLEAN[Data Cleaning]
        CLEAN --> PARSE[AST Parsing]
        PARSE --> FEAT[Feature Extraction]
        FEAT --> SPLIT[Train/Val/Test Split<br/>70/15/15]
    end
    
    subgraph "Model Training"
        SPLIT --> AUG[Data Augmentation]
        AUG --> TRAIN[Distributed Training<br/>Multi-GPU]
        TRAIN --> VAL[Validation]
        VAL --> TUNE[Hyperparameter<br/>Tuning]
        TUNE --> BEST[Best Model<br/>Selection]
    end
    
    subgraph "Evaluation & Deployment"
        BEST --> TEST[Test Set<br/>Evaluation]
        TEST --> METRICS[Performance<br/>Metrics]
        METRICS --> DEPLOY{Deploy?}
        DEPLOY -->|Yes| REG[Model Registry]
        DEPLOY -->|No| TRAIN
    end
```

## Model Inference Pipeline

```mermaid
sequenceDiagram
    participant Client as Client App
    participant API as FastAPI
    participant Cache as Redis Cache
    participant Preprocess as Preprocessor
    participant Models as Model Ensemble
    participant Monitor as Monitoring
    
    Client->>API: POST /analyze {code}
    API->>Cache: Check Cache
    
    alt Cache Hit
        Cache->>API: Cached Results
    else Cache Miss
        API->>Preprocess: Extract Features
        Preprocess->>Preprocess: Parse AST
        Preprocess->>Preprocess: Compute Metrics
        Preprocess->>Models: Feature Vectors
        
        par Parallel Inference
            Models->>Models: Bug Prediction
        and
            Models->>Models: Complexity Analysis
        and
            Models->>Models: Security Scan
        and
            Models->>Models: Tech Debt
        end
        
        Models->>API: Predictions
        API->>Cache: Store Results
        API->>Monitor: Log Metrics
    end
    
    API->>Client: Analysis Results
```

## Model Update Strategy

```mermaid
graph TD
    subgraph "Continuous Learning"
        PROD[Production<br/>Predictions] --> FEEDBACK[User Feedback]
        FEEDBACK --> VALID{Valid<br/>Feedback?}
        VALID -->|Yes| STORE[Feedback Store]
        VALID -->|No| DISCARD[Discard]
        
        STORE --> BATCH[Batch<br/>Accumulation]
        BATCH --> SIZE{Batch<br/>Size > 1000?}
        SIZE -->|Yes| RETRAIN[Trigger<br/>Retraining]
        SIZE -->|No| WAIT[Wait for<br/>More Data]
        
        RETRAIN --> NEWMODEL[Train New<br/>Model Version]
        NEWMODEL --> AB[A/B Testing]
        AB --> COMPARE{Better<br/>Performance?}
        COMPARE -->|Yes| DEPLOY[Deploy New<br/>Version]
        COMPARE -->|No| KEEP[Keep Current<br/>Version]
    end
```

## Model Interpretability

```mermaid
graph TD
    subgraph "Explainability Pipeline"
        PRED[Model Prediction] --> SHAP[SHAP Values]
        PRED --> LIME[LIME Explanations]
        PRED --> ATTN[Attention Weights]
        
        SHAP --> VIZ1[Feature Importance]
        LIME --> VIZ2[Local Explanations]
        ATTN --> VIZ3[Code Attention Map]
        
        VIZ1 --> REPORT[Explanation Report]
        VIZ2 --> REPORT
        VIZ3 --> REPORT
        
        REPORT --> USER[User Interface]
    end
```