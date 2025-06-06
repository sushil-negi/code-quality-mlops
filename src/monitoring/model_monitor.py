#!/usr/bin/env python3
"""
Model Monitoring and Feedback System

This module demonstrates Stage 6: Monitoring and Feedback
- Real-time model performance monitoring
- Data drift detection and alerting
- Model quality metrics tracking
- Feedback loop for continuous improvement
- A/B test analysis and reporting
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import redis
import boto3
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import scipy.stats as stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    model_version: str
    timestamp: datetime
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Operational metrics
    prediction_count: int
    average_latency: float
    error_rate: float
    
    # Business metrics
    cost_per_prediction: float = 0.0
    business_impact: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class DriftAlert:
    """Data drift alert"""
    feature_name: str
    drift_score: float
    threshold: float
    detection_method: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class DriftDetector:
    """Detects data drift using statistical methods"""
    
    def __init__(self, reference_window: int = 7, detection_window: int = 1):
        self.reference_window = reference_window  # days
        self.detection_window = detection_window  # days
        self.drift_thresholds = {
            'ks_test': 0.05,
            'psi': 0.1,
            'wasserstein': 0.3
        }
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect drift between reference and current data"""
        alerts = []
        
        numerical_features = reference_data.select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features:
            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(ref_values, curr_values)
            
            if ks_p_value < self.drift_thresholds['ks_test']:
                severity = self._calculate_severity(ks_statistic, [0.1, 0.3, 0.5])
                alerts.append(DriftAlert(
                    feature_name=feature,
                    drift_score=ks_statistic,
                    threshold=self.drift_thresholds['ks_test'],
                    detection_method='ks_test',
                    timestamp=datetime.now(),
                    severity=severity
                ))
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_values, curr_values)
            
            if psi_score > self.drift_thresholds['psi']:
                severity = self._calculate_severity(psi_score, [0.1, 0.25, 0.5])
                alerts.append(DriftAlert(
                    feature_name=feature,
                    drift_score=psi_score,
                    threshold=self.drift_thresholds['psi'],
                    detection_method='psi',
                    timestamp=datetime.now(),
                    severity=severity
                ))
            
            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(ref_values, curr_values)
            normalized_wasserstein = wasserstein_dist / (ref_values.std() + 1e-8)
            
            if normalized_wasserstein > self.drift_thresholds['wasserstein']:
                severity = self._calculate_severity(normalized_wasserstein, [0.3, 0.6, 1.0])
                alerts.append(DriftAlert(
                    feature_name=feature,
                    drift_score=normalized_wasserstein,
                    threshold=self.drift_thresholds['wasserstein'],
                    detection_method='wasserstein',
                    timestamp=datetime.now(),
                    severity=severity
                ))
        
        return alerts
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Calculate distributions
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            curr_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to probabilities
            ref_prob = ref_hist / len(reference)
            curr_prob = curr_hist / len(current)
            
            # Add small constant to avoid log(0)
            ref_prob = np.where(ref_prob == 0, 1e-8, ref_prob)
            curr_prob = np.where(curr_prob == 0, 1e-8, curr_prob)
            
            # Calculate PSI
            psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
            
            return psi
            
        except Exception as e:
            logger.error(f"Failed to calculate PSI: {e}")
            return 0.0
    
    def _calculate_severity(self, score: float, thresholds: List[float]) -> str:
        """Calculate alert severity based on score and thresholds"""
        if score >= thresholds[2]:
            return 'critical'
        elif score >= thresholds[1]:
            return 'high'
        elif score >= thresholds[0]:
            return 'medium'
        else:
            return 'low'

class PerformanceMonitor:
    """Monitors model performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_client = MlflowClient(config['mlflow']['tracking_uri'])
        
        # Initialize Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Initialize alerting
        self.alert_channels = config.get('alerting', {}).get('channels', [])
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions',
            ['model_name', 'model_version', 'prediction'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_duration_seconds',
            'Model prediction latency',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.model_drift_score = Gauge(
            'model_drift_score',
            'Data drift score',
            ['model_name', 'feature_name', 'method'],
            registry=self.registry
        )
        
        self.model_error_rate = Gauge(
            'model_error_rate',
            'Model error rate',
            ['model_name', 'model_version'],
            registry=self.registry
        )
    
    def calculate_performance_metrics(self, predictions_df: pd.DataFrame) -> ModelMetrics:
        """Calculate model performance metrics from predictions"""
        
        # Get the most recent model info
        model_name = predictions_df['model_name'].iloc[-1]
        model_version = predictions_df['model_version'].iloc[-1]
        
        # Filter to predictions with ground truth
        labeled_predictions = predictions_df.dropna(subset=['actual_label'])
        
        if len(labeled_predictions) == 0:
            logger.warning("No labeled predictions found for performance calculation")
            return None
        
        # Convert predictions to binary
        y_true = labeled_predictions['actual_label'].astype(int)
        y_pred = (labeled_predictions['prediction'] == 'bug_fix').astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Operational metrics
        prediction_count = len(predictions_df)
        avg_latency = predictions_df.get('latency', pd.Series([0])).mean()
        error_rate = predictions_df['error'].fillna(False).mean() if 'error' in predictions_df else 0
        
        # Create metrics object
        metrics = ModelMetrics(
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            prediction_count=prediction_count,
            average_latency=avg_latency,
            error_rate=error_rate
        )
        
        # Update Prometheus metrics
        self.model_accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
        
        self.model_error_rate.labels(
            model_name=model_name,
            model_version=model_version
        ).set(error_rate)
        
        return metrics
    
    def log_metrics_to_mlflow(self, metrics: ModelMetrics):
        """Log metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=f"monitoring_{metrics.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log performance metrics
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("precision", metrics.precision)
                mlflow.log_metric("recall", metrics.recall)
                mlflow.log_metric("f1_score", metrics.f1_score)
                
                # Log operational metrics
                mlflow.log_metric("prediction_count", metrics.prediction_count)
                mlflow.log_metric("average_latency", metrics.average_latency)
                mlflow.log_metric("error_rate", metrics.error_rate)
                
                # Add tags
                mlflow.set_tag("monitoring", "true")
                mlflow.set_tag("model_name", metrics.model_name)
                mlflow.set_tag("model_version", metrics.model_version)
                
                logger.info(f"Metrics logged to MLflow for {metrics.model_name} v{metrics.model_version}")
                
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def push_metrics_to_prometheus(self):
        """Push metrics to Prometheus Pushgateway"""
        try:
            gateway = self.config.get('prometheus', {}).get('pushgateway', 'localhost:9091')
            push_to_gateway(gateway, job='model_monitoring', registry=self.registry)
            logger.info("Metrics pushed to Prometheus")
            
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus: {e}")

class ABTestAnalyzer:
    """Analyzes A/B test results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        
        # Initialize Redis connection if enabled
        if config.get('redis', {}).get('enabled', False):
            try:
                self.redis_client = redis.Redis(
                    host=config['redis']['host'],
                    port=config['redis']['port'],
                    db=config['redis']['db']
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
    
    def analyze_ab_test(self, test_name: str, start_date: datetime, 
                       end_date: datetime) -> Dict[str, Any]:
        """Analyze A/B test results between two time periods"""
        
        # Load prediction data for the test period
        predictions_df = self._load_predictions_data(start_date, end_date)
        
        if predictions_df.empty:
            logger.warning(f"No prediction data found for A/B test {test_name}")
            return {}
        
        # Group by model
        results = {}
        
        for model_name, model_data in predictions_df.groupby('model_name'):
            # Calculate metrics for this model variant
            labeled_data = model_data.dropna(subset=['actual_label'])
            
            if len(labeled_data) == 0:
                continue
            
            y_true = labeled_data['actual_label'].astype(int)
            y_pred = (labeled_data['prediction'] == 'bug_fix').astype(int)
            
            metrics = {
                'sample_size': len(model_data),
                'labeled_sample_size': len(labeled_data),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'average_confidence': model_data['confidence'].mean(),
                'average_latency': model_data.get('latency', pd.Series([0])).mean(),
                'error_rate': model_data['error'].fillna(False).mean() if 'error' in model_data else 0
            }
            
            results[model_name] = metrics
        
        # Perform statistical significance testing
        if len(results) >= 2:
            significance_results = self._test_statistical_significance(predictions_df)
            results['statistical_significance'] = significance_results
        
        # Calculate business impact
        business_impact = self._calculate_business_impact(results)
        results['business_impact'] = business_impact
        
        return results
    
    def _load_predictions_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load prediction data from logs"""
        try:
            # Load from prediction logs (in production, this would be from a database)
            log_dir = Path("prediction_logs")
            all_data = []
            
            current_date = start_date
            while current_date <= end_date:
                log_file = log_dir / f"predictions_{current_date.strftime('%Y%m%d')}.jsonl"
                
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                all_data.append(data)
                            except json.JSONDecodeError:
                                continue
                
                current_date += timedelta(days=1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load predictions data: {e}")
            return pd.DataFrame()
    
    def _test_statistical_significance(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical significance between model variants"""
        
        model_groups = list(predictions_df.groupby('model_name'))
        
        if len(model_groups) < 2:
            return {}
        
        # Get the two main model groups
        model_a_name, model_a_data = model_groups[0]
        model_b_name, model_b_data = model_groups[1]
        
        # Filter to labeled data
        model_a_labeled = model_a_data.dropna(subset=['actual_label'])
        model_b_labeled = model_b_data.dropna(subset=['actual_label'])
        
        if len(model_a_labeled) == 0 or len(model_b_labeled) == 0:
            return {}
        
        # Convert to binary outcomes
        model_a_outcomes = (model_a_labeled['prediction'] == 'bug_fix').astype(int)
        model_b_outcomes = (model_b_labeled['prediction'] == 'bug_fix').astype(int)
        
        model_a_actual = model_a_labeled['actual_label'].astype(int)
        model_b_actual = model_b_labeled['actual_label'].astype(int)
        
        # Calculate accuracy for each model
        model_a_accuracy = accuracy_score(model_a_actual, model_a_outcomes)
        model_b_accuracy = accuracy_score(model_b_actual, model_b_outcomes)
        
        # Perform two-proportion z-test
        n_a = len(model_a_labeled)
        n_b = len(model_b_labeled)
        
        correct_a = (model_a_outcomes == model_a_actual).sum()
        correct_b = (model_b_outcomes == model_b_actual).sum()
        
        # Combined proportion
        p_combined = (correct_a + correct_b) / (n_a + n_b)
        
        # Standard error
        se = np.sqrt(p_combined * (1 - p_combined) * (1/n_a + 1/n_b))
        
        # Z-statistic
        if se > 0:
            z_stat = (model_a_accuracy - model_b_accuracy) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0
        
        return {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'model_a_accuracy': model_a_accuracy,
            'model_b_accuracy': model_b_accuracy,
            'accuracy_difference': model_a_accuracy - model_b_accuracy,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'sample_size_a': n_a,
            'sample_size_b': n_b
        }
    
    def _calculate_business_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business impact of A/B test results"""
        
        if len(results) < 2:
            return {}
        
        # Estimate cost savings from improved accuracy
        baseline_accuracy = 0.7  # Assume baseline accuracy
        
        business_impact = {}
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                accuracy_improvement = metrics['accuracy'] - baseline_accuracy
                
                # Estimate cost per false positive/negative
                cost_per_error = 100  # Example: $100 per error
                predictions_per_day = 1000  # Example volume
                
                daily_savings = accuracy_improvement * predictions_per_day * cost_per_error
                monthly_savings = daily_savings * 30
                
                business_impact[model_name] = {
                    'accuracy_improvement': accuracy_improvement,
                    'daily_cost_savings': daily_savings,
                    'monthly_cost_savings': monthly_savings,
                    'annual_cost_savings': monthly_savings * 12
                }
        
        return business_impact

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = []
    
    def send_drift_alert(self, alerts: List[DriftAlert]):
        """Send drift detection alerts"""
        if not alerts:
            return
        
        # Group alerts by severity
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        high_alerts = [a for a in alerts if a.severity == 'high']
        
        if critical_alerts:
            self._send_immediate_alert(critical_alerts, "CRITICAL: Data Drift Detected")
        
        if high_alerts:
            self._send_alert(high_alerts, "HIGH: Data Drift Detected")
        
        # Store alert history
        self.alert_history.extend(alerts)
    
    def send_performance_alert(self, metrics: ModelMetrics, thresholds: Dict[str, float]):
        """Send performance degradation alerts"""
        alerts = []
        
        if metrics.accuracy < thresholds.get('accuracy', 0.7):
            alerts.append(f"Accuracy dropped to {metrics.accuracy:.3f}")
        
        if metrics.error_rate > thresholds.get('error_rate', 0.1):
            alerts.append(f"Error rate increased to {metrics.error_rate:.3f}")
        
        if metrics.average_latency > thresholds.get('latency', 1.0):
            alerts.append(f"Average latency increased to {metrics.average_latency:.3f}s")
        
        if alerts:
            message = f"Performance degradation detected for {metrics.model_name} v{metrics.model_version}:\n" + \
                     "\n".join(f"- {alert}" for alert in alerts)
            
            self._send_alert([], message)
    
    def _send_immediate_alert(self, alerts: List[DriftAlert], subject: str):
        """Send immediate alert for critical issues"""
        self._send_email_alert(alerts, subject)
        self._send_slack_alert(alerts, subject)
    
    def _send_alert(self, alerts: List[DriftAlert], message: str):
        """Send regular alert"""
        self._send_slack_alert(alerts, message)
    
    def _send_email_alert(self, alerts: List[DriftAlert], subject: str):
        """Send email alert"""
        try:
            email_config = self.config.get('email', {})
            
            if not email_config.get('enabled', False):
                return
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = subject
            
            # Create alert details
            alert_details = "\n".join([
                f"Feature: {alert.feature_name}\n"
                f"Method: {alert.detection_method}\n"
                f"Score: {alert.drift_score:.4f}\n"
                f"Threshold: {alert.threshold:.4f}\n"
                f"Severity: {alert.severity}\n"
                f"Time: {alert.timestamp}\n"
                for alert in alerts
            ])
            
            body = f"Data drift detected:\n\n{alert_details}"
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                
                if email_config.get('username'):
                    server.login(email_config['username'], email_config['password'])
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alerts: List[DriftAlert], message: str):
        """Send Slack alert"""
        try:
            slack_config = self.config.get('slack', {})
            
            if not slack_config.get('enabled', False):
                return
            
            webhook_url = slack_config['webhook_url']
            
            # Create Slack message
            slack_message = {
                "text": message,
                "username": "MLOps Monitor",
                "icon_emoji": ":warning:",
                "attachments": []
            }
            
            # Add alert details as attachments
            for alert in alerts:
                color = {
                    'critical': 'danger',
                    'high': 'warning',
                    'medium': 'good',
                    'low': '#439FE0'
                }.get(alert.severity, 'good')
                
                attachment = {
                    "color": color,
                    "fields": [
                        {"title": "Feature", "value": alert.feature_name, "short": True},
                        {"title": "Method", "value": alert.detection_method, "short": True},
                        {"title": "Score", "value": f"{alert.drift_score:.4f}", "short": True},
                        {"title": "Severity", "value": alert.severity.upper(), "short": True}
                    ],
                    "footer": "MLOps Monitor",
                    "ts": int(alert.timestamp.timestamp())
                }
                
                slack_message["attachments"].append(attachment)
            
            # Send to Slack
            response = requests.post(webhook_url, json=slack_message)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class ModelMonitor:
    """Main monitoring orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.drift_detector = DriftDetector(
            reference_window=self.config.get('drift_detection', {}).get('reference_window', 7),
            detection_window=self.config.get('drift_detection', {}).get('detection_window', 1)
        )
        
        self.performance_monitor = PerformanceMonitor(self.config)
        self.ab_test_analyzer = ABTestAnalyzer(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.last_drift_check = None
        self.last_performance_check = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'mlflow': {
                'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            },
            'monitoring_interval': int(os.getenv('MONITORING_INTERVAL', '300')),  # 5 minutes
            'drift_detection': {
                'enabled': True,
                'reference_window': 7,
                'detection_window': 1,
                'check_interval': 3600  # 1 hour
            },
            'performance_thresholds': {
                'accuracy': 0.7,
                'error_rate': 0.1,
                'latency': 1.0
            },
            'alerting': {
                'channels': ['slack'],
                'email': {
                    'enabled': False
                },
                'slack': {
                    'enabled': os.getenv('SLACK_ENABLED', 'false').lower() == 'true',
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL', '')
                }
            }
        }
    
    async def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        logger.info("Starting monitoring cycle")
        
        try:
            # 1. Performance monitoring
            await self._monitor_performance()
            
            # 2. Drift detection
            await self._monitor_drift()
            
            # 3. A/B test analysis
            await self._analyze_ab_tests()
            
            # 4. Push metrics
            self.performance_monitor.push_metrics_to_prometheus()
            
            logger.info("Monitoring cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            raise
    
    async def _monitor_performance(self):
        """Monitor model performance"""
        logger.info("Monitoring model performance")
        
        try:
            # Load recent predictions
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)  # Last hour
            
            predictions_df = self.ab_test_analyzer._load_predictions_data(start_time, end_time)
            
            if predictions_df.empty:
                logger.info("No recent predictions found for performance monitoring")
                return
            
            # Calculate metrics for each model
            for model_name, model_data in predictions_df.groupby('model_name'):
                metrics = self.performance_monitor.calculate_performance_metrics(model_data)
                
                if metrics:
                    # Log to MLflow
                    self.performance_monitor.log_metrics_to_mlflow(metrics)
                    
                    # Check thresholds and send alerts
                    thresholds = self.config['performance_thresholds']
                    self.alert_manager.send_performance_alert(metrics, thresholds)
                    
                    logger.info(f"Performance metrics calculated for {model_name}: "
                               f"Accuracy={metrics.accuracy:.3f}, F1={metrics.f1_score:.3f}")
            
            self.last_performance_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
    
    async def _monitor_drift(self):
        """Monitor for data drift"""
        if not self.config['drift_detection']['enabled']:
            return
        
        logger.info("Monitoring for data drift")
        
        try:
            # Get reference and current data
            current_time = datetime.now()
            
            # Reference data (last week)
            ref_start = current_time - timedelta(days=7)
            ref_end = current_time - timedelta(days=1)
            reference_data = self.ab_test_analyzer._load_predictions_data(ref_start, ref_end)
            
            # Current data (last day)
            curr_start = current_time - timedelta(days=1)
            curr_end = current_time
            current_data = self.ab_test_analyzer._load_predictions_data(curr_start, curr_end)
            
            if reference_data.empty or current_data.empty:
                logger.info("Insufficient data for drift detection")
                return
            
            # Extract feature columns for drift detection
            feature_columns = [
                'lines_of_code', 'files_changed', 'additions', 'deletions',
                'cyclomatic_complexity', 'cognitive_complexity', 'halstead_volume',
                'halstead_difficulty', 'function_count', 'class_count',
                'comment_ratio', 'docstring_ratio', 'test_file_ratio',
                'import_complexity', 'nested_depth', 'commit_message_length',
                'code_readability', 'error_handling_ratio', 'todo_comment_count',
                'magic_number_count', 'author_experience', 'file_change_frequency'
            ]
            
            # Select only available feature columns
            available_features = [col for col in feature_columns if col in reference_data.columns]
            
            if not available_features:
                logger.warning("No feature columns available for drift detection")
                return
            
            ref_features = reference_data[available_features]
            curr_features = current_data[available_features]
            
            # Detect drift
            drift_alerts = self.drift_detector.detect_drift(ref_features, curr_features)
            
            if drift_alerts:
                logger.warning(f"Drift detected in {len(drift_alerts)} features")
                
                # Update Prometheus metrics
                for alert in drift_alerts:
                    self.performance_monitor.model_drift_score.labels(
                        model_name='all',
                        feature_name=alert.feature_name,
                        method=alert.detection_method
                    ).set(alert.drift_score)
                
                # Send alerts
                self.alert_manager.send_drift_alert(drift_alerts)
            else:
                logger.info("No significant drift detected")
            
            self.last_drift_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Drift monitoring failed: {e}")
    
    async def _analyze_ab_tests(self):
        """Analyze ongoing A/B tests"""
        logger.info("Analyzing A/B test results")
        
        try:
            # Analyze last 7 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            
            results = self.ab_test_analyzer.analyze_ab_test(
                "model_comparison",
                start_time,
                end_time
            )
            
            if results:
                logger.info("A/B test analysis completed")
                
                # Log results to MLflow
                with mlflow.start_run(run_name=f"ab_test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_dict(results, "ab_test_results.json")
                    mlflow.set_tag("analysis_type", "ab_test")
                
                # Check for significant results
                if 'statistical_significance' in results:
                    sig_results = results['statistical_significance']
                    if sig_results.get('significant', False):
                        logger.info(f"Significant difference found: "
                                   f"{sig_results['accuracy_difference']:.3f} "
                                   f"(p={sig_results['p_value']:.4f})")
            else:
                logger.info("No A/B test data available for analysis")
                
        except Exception as e:
            logger.error(f"A/B test analysis failed: {e}")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("Starting continuous model monitoring")
        
        monitoring_interval = self.config['monitoring_interval']
        
        while True:
            try:
                await self.run_monitoring_cycle()
                await asyncio.sleep(monitoring_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

async def main():
    """Main entry point for model monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Monitoring System")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--mode', choices=['continuous', 'single'], 
                       default='continuous', help='Monitoring mode')
    parser.add_argument('--analyze-ab-test', help='Analyze specific A/B test')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ModelMonitor(args.config)
    
    try:
        if args.mode == 'single':
            await monitor.run_monitoring_cycle()
        elif args.analyze_ab_test:
            # Analyze specific A/B test
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            
            results = monitor.ab_test_analyzer.analyze_ab_test(
                args.analyze_ab_test,
                start_time,
                end_time
            )
            
            print(json.dumps(results, indent=2, default=str))
        else:
            await monitor.start_monitoring()
            
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())