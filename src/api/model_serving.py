#!/usr/bin/env python3
"""
Model Serving API - Deployment Pipeline

This module demonstrates Stage 5: Model Deployment
- FastAPI-based model serving endpoint
- Real-time prediction API with validation
- Model loading and caching
- Health checks and monitoring endpoints
- A/B testing support for multiple model versions
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import mlflow
from mlflow.tracking import MlflowClient
import torch
import joblib
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
import boto3
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics for monitoring
REQUEST_COUNT = Counter('model_predictions_total', 'Total model predictions', ['model_name', 'status'])
REQUEST_DURATION = Histogram('model_prediction_duration_seconds', 'Model prediction duration')
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading duration')
ACTIVE_MODELS = Gauge('active_models_total', 'Number of active models loaded')

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    repository: str = Field(..., description="Repository name")
    commit_hash: Optional[str] = Field(None, description="Commit hash")
    
    # Code metrics
    lines_of_code: int = Field(0, ge=0, description="Lines of code")
    files_changed: int = Field(0, ge=0, description="Number of files changed")
    additions: int = Field(0, ge=0, description="Lines added")
    deletions: int = Field(0, ge=0, description="Lines deleted")
    cyclomatic_complexity: float = Field(0, ge=0, description="Cyclomatic complexity")
    cognitive_complexity: float = Field(0, ge=0, description="Cognitive complexity")
    halstead_volume: float = Field(0, ge=0, description="Halstead volume")
    halstead_difficulty: float = Field(0, ge=0, description="Halstead difficulty")
    function_count: int = Field(0, ge=0, description="Number of functions")
    class_count: int = Field(0, ge=0, description="Number of classes")
    comment_ratio: float = Field(0, ge=0, le=1, description="Comment ratio")
    docstring_ratio: float = Field(0, ge=0, le=1, description="Docstring ratio")
    test_file_ratio: float = Field(0, ge=0, le=1, description="Test file ratio")
    import_complexity: float = Field(0, ge=0, description="Import complexity")
    nested_depth: int = Field(0, ge=0, description="Maximum nesting depth")
    
    # Message and readability
    commit_message: Optional[str] = Field("", description="Commit message")
    commit_message_length: int = Field(0, ge=0, description="Commit message length")
    code_readability: float = Field(0.5, ge=0, le=1, description="Code readability score")
    
    # Quality indicators
    error_handling_ratio: float = Field(0, ge=0, le=1, description="Error handling ratio")
    todo_comment_count: int = Field(0, ge=0, description="TODO comment count")
    magic_number_count: int = Field(0, ge=0, description="Magic number count")
    
    # Historical features
    author_experience: int = Field(0, ge=0, description="Author experience score")
    file_change_frequency: float = Field(0, ge=0, description="File change frequency")
    
    @validator('commit_message_length', pre=True, always=True)
    def set_message_length(cls, v, values):
        if 'commit_message' in values and values['commit_message']:
            return len(values['commit_message'])
        return v

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Prediction result: 'bug_fix' or 'normal'")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    features_used: Dict[str, float] = Field(..., description="Feature values used for prediction")
    
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    models_loaded: int = Field(..., description="Number of models loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")
    last_prediction: Optional[datetime] = Field(None, description="Last prediction timestamp")

class ModelCache:
    """In-memory model cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
    
    def get(self, key: str):
        """Get model from cache"""
        if key in self.cache:
            current_time = time.time()
            if current_time - self.access_times[key] < self.ttl_seconds:
                self.access_times[key] = current_time
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key: str, model):
        """Store model in cache"""
        self.cache[key] = model
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached models"""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

class ModelManager:
    """Manages model loading, caching, and A/B testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_client = MlflowClient(tracking_uri=config['mlflow']['tracking_uri'])
        self.cache = ModelCache(ttl_seconds=config.get('model_cache_ttl', 3600))
        self.active_models = {}
        self.ab_test_config = config.get('ab_testing', {})
        self.redis_client = None
        
        # Initialize Redis for A/B testing if configured
        if config.get('redis', {}).get('enabled', False):
            try:
                self.redis_client = redis.Redis(
                    host=config['redis']['host'],
                    port=config['redis']['port'],
                    db=config['redis']['db']
                )
                logger.info("Redis client initialized for A/B testing")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
    
    async def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load model from MLflow registry with caching"""
        cache_key = f"{model_name}:{stage}"
        
        # Check cache first
        cached_model = self.cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        
        start_time = time.time()
        
        try:
            # Get latest model version
            model_versions = self.mlflow_client.get_latest_versions(model_name, stages=[stage])
            
            if not model_versions:
                raise ValueError(f"No model found for {model_name} in stage {stage}")
            
            model_version = model_versions[0]
            model_uri = f"models:/{model_name}/{model_version.version}"
            
            # Load model based on type
            if 'pytorch' in model_version.tags.get('mlflow.source.type', ''):
                model = mlflow.pytorch.load_model(model_uri)
            else:
                model = mlflow.sklearn.load_model(model_uri)
            
            # Cache the model
            self.cache.set(cache_key, {
                'model': model,
                'version': model_version.version,
                'metadata': {
                    'stage': stage,
                    'run_id': model_version.run_id,
                    'tags': model_version.tags
                }
            })
            
            self.active_models[cache_key] = {
                'model_name': model_name,
                'version': model_version.version,
                'stage': stage,
                'loaded_at': datetime.now()
            }
            
            # Update metrics
            load_duration = time.time() - start_time
            MODEL_LOAD_TIME.observe(load_duration)
            ACTIVE_MODELS.set(len(self.active_models))
            
            logger.info(f"Loaded model {model_name} v{model_version.version} in {load_duration:.2f}s")
            
            return self.cache.get(cache_key)
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    def select_model_for_request(self, request: PredictionRequest) -> str:
        """Select model based on A/B testing configuration"""
        default_model = self.config.get('default_model', 'simple_bug_predictor')
        
        if not self.ab_test_config.get('enabled', False):
            return default_model
        
        # Simple A/B testing based on repository hash
        if self.redis_client:
            try:
                # Get A/B test assignment from Redis
                test_key = f"ab_test:{request.repository}"
                assigned_model = self.redis_client.get(test_key)
                
                if assigned_model:
                    return assigned_model.decode('utf-8')
                
                # Assign to A/B test group
                traffic_split = self.ab_test_config.get('traffic_split', 0.5)
                hash_value = hash(request.repository) % 100
                
                if hash_value < traffic_split * 100:
                    selected_model = self.ab_test_config.get('model_a', default_model)
                else:
                    selected_model = self.ab_test_config.get('model_b', 'hybrid_bug_predictor')
                
                # Store assignment in Redis with TTL
                self.redis_client.setex(test_key, 86400, selected_model)  # 24 hours
                
                return selected_model
                
            except Exception as e:
                logger.warning(f"A/B testing failed, using default model: {e}")
        
        return default_model
    
    async def predict(self, model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using specified model"""
        model_data = await self.load_model(model_name)
        model = model_data['model']
        
        # Prepare features for prediction
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # Probabilistic prediction
            probabilities = model.predict_proba(feature_df)
            prediction = model.predict(feature_df)[0]
            confidence = float(probabilities[0][prediction])
        else:
            # Simple prediction
            prediction = model.predict(feature_df)[0]
            confidence = 0.8  # Default confidence for non-probabilistic models
        
        return {
            'prediction': 'bug_fix' if prediction == 1 else 'normal',
            'confidence': confidence,
            'model_version': model_data['version'],
            'features_used': features
        }

# Global variables
model_manager: Optional[ModelManager] = None
service_start_time = time.time()
last_prediction_time: Optional[datetime] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("Starting Model Serving API...")
    
    # Load configuration
    config = load_config()
    
    # Initialize model manager
    model_manager = ModelManager(config)
    
    # Pre-load default models
    try:
        default_models = config.get('preload_models', ['simple_bug_predictor'])
        for model_name in default_models:
            await model_manager.load_model(model_name)
            logger.info(f"Pre-loaded model: {model_name}")
    except Exception as e:
        logger.warning(f"Failed to pre-load models: {e}")
    
    logger.info("Model Serving API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Model Serving API...")
    if model_manager:
        model_manager.cache.clear()
    logger.info("Model Serving API shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Code Quality MLOps - Model Serving API",
    description="Real-time model serving for bug prediction in code repositories",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

def load_config() -> Dict[str, Any]:
    """Load service configuration"""
    config_path = os.getenv('CONFIG_PATH', 'config/serving.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        'mlflow': {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        },
        'default_model': os.getenv('DEFAULT_MODEL', 'simple_bug_predictor'),
        'model_cache_ttl': int(os.getenv('MODEL_CACHE_TTL', '3600')),
        'preload_models': os.getenv('PRELOAD_MODELS', 'simple_bug_predictor').split(','),
        'ab_testing': {
            'enabled': os.getenv('AB_TESTING_ENABLED', 'false').lower() == 'true',
            'traffic_split': float(os.getenv('AB_TRAFFIC_SPLIT', '0.5')),
            'model_a': os.getenv('AB_MODEL_A', 'simple_bug_predictor'),
            'model_b': os.getenv('AB_MODEL_B', 'hybrid_bug_predictor')
        },
        'redis': {
            'enabled': os.getenv('REDIS_ENABLED', 'false').lower() == 'true',
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0'))
        }
    }

def extract_features(request: PredictionRequest) -> Dict[str, float]:
    """Extract features from prediction request"""
    return {
        'lines_of_code': float(request.lines_of_code),
        'files_changed': float(request.files_changed),
        'additions': float(request.additions),
        'deletions': float(request.deletions),
        'cyclomatic_complexity': request.cyclomatic_complexity,
        'cognitive_complexity': request.cognitive_complexity,
        'halstead_volume': request.halstead_volume,
        'halstead_difficulty': request.halstead_difficulty,
        'function_count': float(request.function_count),
        'class_count': float(request.class_count),
        'comment_ratio': request.comment_ratio,
        'docstring_ratio': request.docstring_ratio,
        'test_file_ratio': request.test_file_ratio,
        'import_complexity': request.import_complexity,
        'nested_depth': float(request.nested_depth),
        'commit_message_length': float(request.commit_message_length),
        'code_readability': request.code_readability,
        'error_handling_ratio': request.error_handling_ratio,
        'todo_comment_count': float(request.todo_comment_count),
        'magic_number_count': float(request.magic_number_count),
        'author_experience': float(request.author_experience),
        'file_change_frequency': request.file_change_frequency
    }

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Code Quality MLOps - Model Serving API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "models": "/models"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a bug prediction for the given code metrics"""
    global last_prediction_time
    
    start_time = time.time()
    prediction_id = f"{request.repository}_{int(time.time())}"
    
    try:
        # Select model for this request (A/B testing)
        model_name = model_manager.select_model_for_request(request)
        
        # Extract features
        features = extract_features(request)
        
        # Make prediction
        result = await model_manager.predict(model_name, features)
        
        # Create response
        response = PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            model_name=model_name,
            model_version=result['model_version'],
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            features_used=result['features_used']
        )
        
        # Update metrics
        REQUEST_COUNT.labels(model_name=model_name, status='success').inc()
        REQUEST_DURATION.observe(time.time() - start_time)
        last_prediction_time = datetime.now()
        
        # Log prediction (background task)
        background_tasks.add_task(log_prediction, request, response)
        
        logger.info(f"Prediction completed: {prediction_id} -> {result['prediction']} ({result['confidence']:.3f})")
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(model_name='unknown', status='error').inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - service_start_time
    models_loaded = model_manager.cache.size() if model_manager else 0
    
    return HealthResponse(
        status="healthy" if model_manager else "unhealthy",
        models_loaded=models_loaded,
        uptime_seconds=uptime,
        version="1.0.0",
        last_prediction=last_prediction_time
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest().decode('utf-8')

@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List currently loaded models"""
    if not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    return {
        "active_models": model_manager.active_models,
        "cache_size": model_manager.cache.size(),
        "ab_testing_enabled": model_manager.ab_test_config.get('enabled', False)
    }

@app.post("/models/{model_name}/reload")
async def reload_model(model_name: str, stage: str = "Production"):
    """Reload a specific model"""
    if not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    try:
        # Clear from cache
        cache_key = f"{model_name}:{stage}"
        model_manager.cache.cache.pop(cache_key, None)
        model_manager.cache.access_times.pop(cache_key, None)
        
        # Reload model
        await model_manager.load_model(model_name, stage)
        
        return {"message": f"Model {model_name} reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reload model {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )

async def log_prediction(request: PredictionRequest, response: PredictionResponse):
    """Log prediction for monitoring and feedback (background task)"""
    try:
        log_entry = {
            'prediction_id': response.prediction_id,
            'timestamp': response.timestamp.isoformat(),
            'repository': request.repository,
            'commit_hash': request.commit_hash,
            'prediction': response.prediction,
            'confidence': response.confidence,
            'model_name': response.model_name,
            'model_version': response.model_version,
            'features': response.features_used
        }
        
        # Save to local file (in production, this would go to a database or logging service)
        log_dir = Path("prediction_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

if __name__ == "__main__":
    # Configuration for local development
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info"
    }
    
    # Override with environment variables
    config["host"] = os.getenv("HOST", config["host"])
    config["port"] = int(os.getenv("PORT", config["port"]))
    config["reload"] = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server at {config['host']}:{config['port']}")
    uvicorn.run("model_serving:app", **config)