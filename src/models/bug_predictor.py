#!/usr/bin/env python3
"""
Model Training Pipeline - Bug Prediction Model

This module demonstrates Stage 3: Model Training
- Implements a transformer-based bug prediction model
- Uses MLflow for experiment tracking and model registry
- Includes hyperparameter optimization
- Supports distributed training
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import optuna
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeFeaturesDataset(Dataset):
    """PyTorch dataset for code features"""
    
    def __init__(self, features_df: pd.DataFrame, tokenizer=None, max_length: int = 512):
        self.features_df = features_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Separate numerical and text features
        self.numerical_features = self._extract_numerical_features()
        self.text_features = self._extract_text_features()
        self.labels = features_df['is_bug_fix'].astype(int).values
    
    def _extract_numerical_features(self) -> np.ndarray:
        """Extract numerical features"""
        numerical_cols = [
            'lines_of_code', 'files_changed', 'additions', 'deletions',
            'cyclomatic_complexity', 'cognitive_complexity', 'halstead_volume',
            'halstead_difficulty', 'function_count', 'class_count',
            'comment_ratio', 'docstring_ratio', 'test_file_ratio',
            'import_complexity', 'nested_depth', 'commit_message_length',
            'code_readability', 'error_handling_ratio', 'todo_comment_count',
            'magic_number_count', 'author_experience', 'file_change_frequency'
        ]
        
        # Fill missing values and normalize
        features = self.features_df[numerical_cols].fillna(0).values
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    
    def _extract_text_features(self) -> List[str]:
        """Extract text features (commit messages)"""
        return self.features_df['commit_message'].fillna('').tolist()
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        # Numerical features
        numerical = torch.FloatTensor(self.numerical_features[idx])
        
        # Text features (if tokenizer provided)
        if self.tokenizer:
            text = self.text_features[idx]
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'numerical_features': numerical,
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'label': torch.LongTensor([self.labels[idx]])
            }
        else:
            return {
                'numerical_features': numerical,
                'label': torch.LongTensor([self.labels[idx]])
            }

class HybridBugPredictor(nn.Module):
    """Hybrid model combining transformer and traditional features"""
    
    def __init__(self, 
                 numerical_features_dim: int,
                 transformer_model: str = 'microsoft/codebert-base',
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        # Text encoder (CodeBERT)
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.transformer_dim = self.transformer.config.hidden_size
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer
        fusion_dim = self.transformer_dim + hidden_dim // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # Binary classification
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.numerical_encoder, self.fusion_layer, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids=None, attention_mask=None, numerical_features=None):
        # Text encoding
        if input_ids is not None and attention_mask is not None:
            transformer_output = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = transformer_output.pooler_output
        else:
            # If no text input, use zero vector
            batch_size = numerical_features.size(0)
            text_features = torch.zeros(batch_size, self.transformer_dim, 
                                      device=numerical_features.device)
        
        # Numerical encoding
        numerical_encoded = self.numerical_encoder(numerical_features)
        
        # Feature fusion
        fused_features = torch.cat([text_features, numerical_encoded], dim=1)
        fused_output = self.fusion_layer(fused_features)
        
        # Classification
        logits = self.classifier(fused_output)
        
        return logits

class SimpleBugPredictor:
    """Simple scikit-learn based bug predictor for baseline"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        # Select numerical features
        numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        X_numerical = X[numerical_cols].fillna(0)
        
        # Store feature names
        self.feature_names = X_numerical.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numerical)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_numerical = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_numerical)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions"""
        X_numerical = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_numerical)
        return self.model.predict_proba(X_scaled)

class MLflowTrainingPipeline:
    """MLflow-integrated training pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.mlflow_client = None
        self._setup_mlflow()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'mlflow': {
                'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
                'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'bug_prediction'),
                'artifact_location': os.getenv('MLFLOW_ARTIFACT_LOCATION', 's3://mlops-artifacts/bug-prediction')
            },
            'data': {
                'features_path': os.getenv('FEATURES_PATH', 'processed_features/'),
                'train_test_split': 0.2,
                'validation_split': 0.1
            },
            'training': {
                'model_type': os.getenv('MODEL_TYPE', 'hybrid'),  # 'simple' or 'hybrid'
                'batch_size': int(os.getenv('BATCH_SIZE', '32')),
                'learning_rate': float(os.getenv('LEARNING_RATE', '2e-5')),
                'epochs': int(os.getenv('EPOCHS', '10')),
                'early_stopping_patience': int(os.getenv('EARLY_STOPPING_PATIENCE', '3'))
            },
            'hyperparameter_tuning': {
                'enabled': os.getenv('HYPERPARAMETER_TUNING', 'false').lower() == 'true',
                'n_trials': int(os.getenv('OPTUNA_TRIALS', '50'))
            }
        }
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.config['mlflow']['experiment_name'])
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.config['mlflow']['experiment_name'],
                    artifact_location=self.config['mlflow']['artifact_location']
                )
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            self.mlflow_client = MlflowClient()
            
            logger.info(f"MLflow setup complete. Experiment: {self.config['mlflow']['experiment_name']}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and combine training data from processed features"""
        features_path = Path(self.config['data']['features_path'])
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features directory not found: {features_path}")
        
        # Load all feature files
        all_features = []
        for feature_file in features_path.glob("*.jsonl"):
            logger.info(f"Loading features from: {feature_file}")
            
            with open(feature_file, 'r') as f:
                for line in f:
                    try:
                        feature_data = json.loads(line.strip())
                        all_features.append(feature_data)
                    except json.JSONDecodeError:
                        continue
        
        if not all_features:
            raise ValueError("No training data found")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Add derived features
        df['commit_message'] = df['commit_hash'].apply(lambda x: f"Commit {x}")  # Placeholder
        
        logger.info(f"Loaded {len(df)} training samples")
        logger.info(f"Bug fix ratio: {df['is_bug_fix'].mean():.3f}")
        
        return df
    
    def prepare_datasets(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test datasets"""
        # Split data
        train_df, test_df = train_test_split(
            df, 
            test_size=self.config['data']['train_test_split'],
            stratify=df['is_bug_fix'],
            random_state=42
        )
        
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.config['data']['validation_split'],
            stratify=train_df['is_bug_fix'],
            random_state=42
        )
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        if self.config['training']['model_type'] == 'hybrid':
            tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
            
            train_dataset = CodeFeaturesDataset(train_df, tokenizer)
            val_dataset = CodeFeaturesDataset(val_df, tokenizer)
            test_dataset = CodeFeaturesDataset(test_df, tokenizer)
        else:
            train_dataset = CodeFeaturesDataset(train_df)
            val_dataset = CodeFeaturesDataset(val_df)
            test_dataset = CodeFeaturesDataset(test_df)
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_simple_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train simple scikit-learn model"""
        with mlflow.start_run(run_name="simple_bug_predictor"):
            # Log parameters
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Prepare data
            X = df.drop(['is_bug_fix', 'commit_hash', 'repository', 'timestamp'], axis=1)
            y = df['is_bug_fix']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Train model
            model = SimpleBugPredictor()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="simple_bug_predictor"
            )
            
            logger.info(f"Simple model metrics: {metrics}")
            return metrics
    
    def train_hybrid_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train hybrid transformer model"""
        with mlflow.start_run(run_name="hybrid_bug_predictor"):
            # Model setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get feature dimensions from first batch
            sample_batch = next(iter(train_loader))
            numerical_dim = sample_batch['numerical_features'].shape[1]
            
            model = HybridBugPredictor(numerical_features_dim=numerical_dim)
            model.to(device)
            
            # Log parameters
            mlflow.log_param("model_type", "hybrid_transformer")
            mlflow.log_param("numerical_features_dim", numerical_dim)
            mlflow.log_param("learning_rate", self.config['training']['learning_rate'])
            mlflow.log_param("batch_size", self.config['training']['batch_size'])
            mlflow.log_param("epochs", self.config['training']['epochs'])
            
            # Training setup
            optimizer = optim.AdamW(model.parameters(), lr=self.config['training']['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            best_val_f1 = 0
            patience_counter = 0
            
            # Training loop
            for epoch in range(self.config['training']['epochs']):
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    # Move to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    if 'input_ids' in batch:
                        logits = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            numerical_features=batch['numerical_features']
                        )
                    else:
                        logits = model(numerical_features=batch['numerical_features'])
                    
                    loss = criterion(logits, batch['label'].squeeze())
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    train_total += batch['label'].size(0)
                    train_correct += (predicted == batch['label'].squeeze()).sum().item()
                
                train_accuracy = train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                val_metrics = self._evaluate_model(model, val_loader, device, criterion)
                
                # Log metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
                mlflow.log_metric("val_f1", val_metrics['f1_score'], step=epoch)
                
                logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}: "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Train Acc: {train_accuracy:.4f}, "
                           f"Val F1: {val_metrics['f1_score']:.4f}")
                
                # Early stopping
                if val_metrics['f1_score'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_score']
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            model.load_state_dict(torch.load('best_model.pth'))
            
            # Log model
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="hybrid_bug_predictor"
            )
            
            return val_metrics
    
    def _evaluate_model(self, model, data_loader, device, criterion):
        """Evaluate model on validation/test set"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass
                if 'input_ids' in batch:
                    logits = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        numerical_features=batch['numerical_features']
                    )
                else:
                    logits = model(numerical_features=batch['numerical_features'])
                
                loss = criterion(logits, batch['label'].squeeze())
                total_loss += loss.item()
                
                # Get predictions
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['label'].squeeze().cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'f1_score': f1_score(all_labels, all_predictions),
            'roc_auc': roc_auc_score(all_labels, all_probabilities),
            'loss': total_loss / len(data_loader)
        }
        
        return metrics
    
    def hyperparameter_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run hyperparameter optimization using Optuna"""
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            with mlflow.start_run(nested=True):
                # Log trial parameters
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("hidden_dim", hidden_dim)
                mlflow.log_param("dropout_rate", dropout_rate)
                
                # Quick training with reduced epochs
                original_epochs = self.config['training']['epochs']
                original_batch_size = self.config['training']['batch_size']
                original_lr = self.config['training']['learning_rate']
                
                self.config['training']['epochs'] = 5  # Quick evaluation
                self.config['training']['batch_size'] = batch_size
                self.config['training']['learning_rate'] = learning_rate
                
                try:
                    train_loader, val_loader, _ = self.prepare_datasets(df)
                    metrics = self.train_hybrid_model(train_loader, val_loader)
                    
                    # Log trial result
                    mlflow.log_metric("trial_f1_score", metrics['f1_score'])
                    
                    return metrics['f1_score']
                    
                except Exception as e:
                    logger.error(f"Trial failed: {e}")
                    return 0
                
                finally:
                    # Restore original config
                    self.config['training']['epochs'] = original_epochs
                    self.config['training']['batch_size'] = original_batch_size
                    self.config['training']['learning_rate'] = original_lr
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['hyperparameter_tuning']['n_trials'])
        
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting training pipeline")
        
        try:
            # Load data
            df = self.load_training_data()
            
            # Hyperparameter optimization (if enabled)
            if self.config['hyperparameter_tuning']['enabled']:
                logger.info("Running hyperparameter optimization")
                best_params = self.hyperparameter_optimization(df)
                
                # Update config with best parameters
                self.config['training'].update(best_params)
                logger.info(f"Using optimized parameters: {best_params}")
            
            # Train models
            if self.config['training']['model_type'] == 'simple':
                metrics = self.train_simple_model(df)
            elif self.config['training']['model_type'] == 'hybrid':
                train_loader, val_loader, test_loader = self.prepare_datasets(df)
                metrics = self.train_hybrid_model(train_loader, val_loader)
            else:
                # Train both for comparison
                logger.info("Training both models for comparison")
                simple_metrics = self.train_simple_model(df)
                
                train_loader, val_loader, test_loader = self.prepare_datasets(df)
                hybrid_metrics = self.train_hybrid_model(train_loader, val_loader)
                
                metrics = {
                    'simple_model': simple_metrics,
                    'hybrid_model': hybrid_metrics
                }
            
            logger.info("Training pipeline completed successfully")
            logger.info(f"Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

async def main():
    """Main entry point for training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bug Prediction Model Training")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--model-type', choices=['simple', 'hybrid', 'both'], 
                       help='Type of model to train')
    parser.add_argument('--tune-hyperparams', action='store_true', 
                       help='Enable hyperparameter tuning')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Validate configuration only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLflowTrainingPipeline(args.config)
    
    # Override config with CLI arguments
    if args.model_type:
        pipeline.config['training']['model_type'] = args.model_type
    
    if args.tune_hyperparams:
        pipeline.config['hyperparameter_tuning']['enabled'] = True
    
    if args.dry_run:
        logger.info("Dry run mode - validating configuration")
        logger.info(f"MLflow URI: {pipeline.config['mlflow']['tracking_uri']}")
        logger.info(f"Model type: {pipeline.config['training']['model_type']}")
        logger.info(f"Hyperparameter tuning: {pipeline.config['hyperparameter_tuning']['enabled']}")
        return
    
    # Run training
    try:
        metrics = pipeline.run_training_pipeline()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())