#!/usr/bin/env python3
"""
Model Validation and Testing Framework

This module demonstrates Stage 4: Model Validation
- Comprehensive model testing and validation
- Performance benchmarking and regression testing
- Data drift detection and model stability analysis
- A/B testing framework for model comparison
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import torch
import joblib
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation framework"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.mlflow_client = MlflowClient(tracking_uri=self.config['mlflow']['tracking_uri'])
        self.validation_results = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load validation configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'mlflow': {
                'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
                'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'bug_prediction')
            },
            'validation': {
                'test_data_path': os.getenv('TEST_DATA_PATH', 'test_data/'),
                'performance_thresholds': {
                    'accuracy': 0.8,
                    'precision': 0.75,
                    'recall': 0.7,
                    'f1_score': 0.72,
                    'roc_auc': 0.8
                },
                'drift_detection': {
                    'enabled': True,
                    'significance_level': 0.05,
                    'reference_period_days': 30
                },
                'stability_tests': {
                    'enabled': True,
                    'perturbation_ratio': 0.1,
                    'num_perturbations': 100
                }
            },
            'output': {
                'reports_path': 'validation_reports/',
                'plots_path': 'validation_plots/'
            }
        }
    
    def load_model_from_mlflow(self, model_name: str, stage: str = "Production") -> Any:
        """Load model from MLflow registry"""
        try:
            model_version = self.mlflow_client.get_latest_versions(
                model_name, stages=[stage]
            )[0]
            
            model_uri = f"models:/{model_name}/{model_version.version}"
            
            # Determine model type and load appropriately
            if 'pytorch' in model_version.tags.get('mlflow.source.type', ''):
                model = mlflow.pytorch.load_model(model_uri)
            else:
                model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(f"Loaded model {model_name} version {model_version.version}")
            return model, model_version
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset"""
        test_data_path = Path(self.config['validation']['test_data_path'])
        
        if test_data_path.is_file():
            if test_data_path.suffix == '.csv':
                return pd.read_csv(test_data_path)
            elif test_data_path.suffix == '.json':
                return pd.read_json(test_data_path)
            elif test_data_path.suffix == '.jsonl':
                return pd.read_json(test_data_path, lines=True)
        else:
            # Load from multiple files
            all_data = []
            for file_path in test_data_path.glob("*.jsonl"):
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            all_data.append(data)
                        except json.JSONDecodeError:
                            continue
            
            return pd.DataFrame(all_data)
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'support': len(y_true)
        }
        
        if y_pred_proba is not None:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                
                # Calculate additional metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics.update({
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
                })
        
        return metrics
    
    def cross_validation_analysis(self, model, X: pd.DataFrame, y: pd.Series, 
                                cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation analysis"""
        logger.info("Running cross-validation analysis")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Define scoring metrics
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'confidence_interval': (
                    scores.mean() - 1.96 * scores.std() / np.sqrt(cv_folds),
                    scores.mean() + 1.96 * scores.std() / np.sqrt(cv_folds)
                )
            }
        
        logger.info(f"Cross-validation completed. Mean F1 Score: {cv_results['f1_weighted']['mean']:.4f}")
        return cv_results
    
    def stability_analysis(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze model stability with data perturbations"""
        logger.info("Running stability analysis")
        
        baseline_pred = model.predict(X)
        baseline_metrics = self.calculate_performance_metrics(y, baseline_pred)
        
        perturbation_results = []
        num_perturbations = self.config['validation']['stability_tests']['num_perturbations']
        perturbation_ratio = self.config['validation']['stability_tests']['perturbation_ratio']
        
        for i in range(num_perturbations):
            # Create perturbed dataset
            X_perturbed = X.copy()
            
            # Add random noise to numerical features
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                noise = np.random.normal(0, X[col].std() * perturbation_ratio, len(X))
                X_perturbed[col] += noise
            
            # Make predictions on perturbed data
            perturbed_pred = model.predict(X_perturbed)
            perturbed_metrics = self.calculate_performance_metrics(y, perturbed_pred)
            
            # Calculate stability metrics
            prediction_agreement = (baseline_pred == perturbed_pred).mean()
            
            perturbation_results.append({
                'iteration': i + 1,
                'prediction_agreement': prediction_agreement,
                'f1_score': perturbed_metrics['f1_score'],
                'accuracy': perturbed_metrics['accuracy']
            })
        
        # Aggregate results
        perturbation_df = pd.DataFrame(perturbation_results)
        
        stability_metrics = {
            'mean_prediction_agreement': perturbation_df['prediction_agreement'].mean(),
            'std_prediction_agreement': perturbation_df['prediction_agreement'].std(),
            'mean_f1_stability': perturbation_df['f1_score'].mean(),
            'std_f1_stability': perturbation_df['f1_score'].std(),
            'baseline_f1': baseline_metrics['f1_score'],
            'f1_degradation': baseline_metrics['f1_score'] - perturbation_df['f1_score'].mean(),
            'perturbation_results': perturbation_results
        }
        
        logger.info(f"Stability analysis completed. Mean prediction agreement: {stability_metrics['mean_prediction_agreement']:.4f}")
        return stability_metrics
    
    def calibration_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration"""
        logger.info("Running calibration analysis")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Calculate Brier score
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        
        # Calculate reliability (calibration error)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
    
    def data_drift_detection(self, reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        logger.info("Running data drift detection")
        
        drift_results = {}
        significance_level = self.config['validation']['drift_detection']['significance_level']
        
        numerical_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in current_data.columns:
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.ks_2samp(
                    reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                # Mann-Whitney U test (for non-parametric comparison)
                mw_statistic, mw_p_value = stats.mannwhitneyu(
                    reference_data[col].dropna(),
                    current_data[col].dropna(),
                    alternative='two-sided'
                )
                
                drift_results[col] = {
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'ks_drift_detected': ks_p_value < significance_level,
                    'mw_statistic': mw_statistic,
                    'mw_p_value': mw_p_value,
                    'mw_drift_detected': mw_p_value < significance_level,
                    'reference_mean': reference_data[col].mean(),
                    'current_mean': current_data[col].mean(),
                    'mean_shift': (current_data[col].mean() - reference_data[col].mean()) / reference_data[col].std()
                }
        
        # Overall drift assessment
        drift_detected_features = [
            col for col, results in drift_results.items() 
            if results['ks_drift_detected'] or results['mw_drift_detected']
        ]
        
        overall_drift = {
            'total_features': len(drift_results),
            'features_with_drift': len(drift_detected_features),
            'drift_ratio': len(drift_detected_features) / len(drift_results) if drift_results else 0,
            'drift_detected_features': drift_detected_features,
            'overall_drift_detected': len(drift_detected_features) / len(drift_results) > 0.3 if drift_results else False
        }
        
        logger.info(f"Data drift analysis completed. Drift detected in {len(drift_detected_features)} features")
        return {'feature_drift': drift_results, 'overall_drift': overall_drift}
    
    def bias_fairness_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             sensitive_features: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze model bias and fairness"""
        logger.info("Running bias and fairness analysis")
        
        if sensitive_features is None:
            logger.warning("No sensitive features provided for bias analysis")
            return {}
        
        bias_results = {}
        
        for feature in sensitive_features.columns:
            unique_values = sensitive_features[feature].unique()
            
            if len(unique_values) <= 10:  # Categorical feature
                feature_bias = {}
                
                for value in unique_values:
                    mask = sensitive_features[feature] == value
                    group_y_true = y_true[mask]
                    group_y_pred = y_pred[mask]
                    
                    if len(group_y_true) > 0:
                        group_metrics = self.calculate_performance_metrics(group_y_true, group_y_pred)
                        feature_bias[str(value)] = {
                            'count': len(group_y_true),
                            'metrics': group_metrics
                        }
                
                # Calculate fairness metrics
                if len(feature_bias) >= 2:
                    groups = list(feature_bias.keys())
                    group1_f1 = feature_bias[groups[0]]['metrics']['f1_score']
                    group2_f1 = feature_bias[groups[1]]['metrics']['f1_score']
                    
                    feature_bias['fairness_metrics'] = {
                        'equalized_odds_difference': abs(group1_f1 - group2_f1),
                        'fairness_satisfied': abs(group1_f1 - group2_f1) < 0.1  # 10% threshold
                    }
                
                bias_results[feature] = feature_bias
        
        return bias_results
    
    def feature_importance_analysis(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance using SHAP"""
        logger.info("Running feature importance analysis")
        
        try:
            # Use SHAP for model explainability
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X)
            else:
                explainer = shap.Explainer(model.predict, X)
            
            # Calculate SHAP values for a sample of data (to avoid memory issues)
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            
            shap_values = explainer(X_sample)
            
            # Calculate feature importance
            if isinstance(shap_values.values, list):
                # Multi-class case - use values for positive class
                feature_importance = np.abs(shap_values.values[1]).mean(axis=0)
            else:
                # Binary or regression case
                feature_importance = np.abs(shap_values.values).mean(axis=0)
            
            # Create importance dictionary
            importance_dict = dict(zip(X.columns, feature_importance))
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': dict(sorted_importance),
                'top_10_features': sorted_importance[:10],
                'shap_analysis_completed': True
            }
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            
            # Fallback to model-based feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                return {
                    'feature_importance': dict(sorted_importance),
                    'top_10_features': sorted_importance[:10],
                    'method': 'model_based'
                }
            
            return {'error': str(e)}
    
    def performance_regression_test(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check for performance regression against thresholds"""
        thresholds = self.config['validation']['performance_thresholds']
        
        regression_results = {}
        overall_pass = True
        
        for metric, threshold in thresholds.items():
            if metric in current_metrics:
                passed = current_metrics[metric] >= threshold
                regression_results[metric] = {
                    'current_value': current_metrics[metric],
                    'threshold': threshold,
                    'passed': passed,
                    'difference': current_metrics[metric] - threshold
                }
                
                if not passed:
                    overall_pass = False
            else:
                regression_results[metric] = {
                    'status': 'metric_not_available'
                }
        
        return {
            'overall_pass': overall_pass,
            'individual_results': regression_results,
            'passed_count': sum(1 for r in regression_results.values() if r.get('passed', False)),
            'total_count': len(thresholds)
        }
    
    def generate_validation_report(self, model_name: str, validation_results: Dict[str, Any]):
        """Generate comprehensive validation report"""
        # Create output directories
        reports_path = Path(self.config['output']['reports_path'])
        plots_path = Path(self.config['output']['plots_path'])
        reports_path.mkdir(exist_ok=True)
        plots_path.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create detailed report
        report = {
            'model_name': model_name,
            'validation_timestamp': timestamp,
            'summary': {
                'overall_performance': validation_results.get('performance_metrics', {}),
                'performance_regression': validation_results.get('regression_test', {}),
                'stability_assessment': validation_results.get('stability_analysis', {}),
                'drift_detection': validation_results.get('drift_analysis', {}),
                'calibration_quality': validation_results.get('calibration_analysis', {})
            },
            'detailed_results': validation_results
        }
        
        # Save report
        report_file = reports_path / f"{model_name}_validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved: {report_file}")
        
        # Generate plots
        self._generate_validation_plots(validation_results, plots_path, timestamp)
        
        return report_file
    
    def _generate_validation_plots(self, results: Dict[str, Any], plots_path: Path, timestamp: str):
        """Generate validation plots"""
        try:
            # Confusion matrix
            if 'confusion_matrix' in results:
                plt.figure(figsize=(8, 6))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(plots_path / f'confusion_matrix_{timestamp}.png')
                plt.close()
            
            # Feature importance
            if 'feature_importance' in results and 'top_10_features' in results['feature_importance']:
                top_features = results['feature_importance']['top_10_features']
                features, importance = zip(*top_features)
                
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(features)), importance)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Feature Importance')
                plt.tight_layout()
                plt.savefig(plots_path / f'feature_importance_{timestamp}.png')
                plt.close()
            
            # Calibration plot
            if 'calibration_analysis' in results:
                cal_data = results['calibration_analysis']
                if 'fraction_of_positives' in cal_data:
                    plt.figure(figsize=(8, 6))
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                    plt.plot(cal_data['mean_predicted_value'], 
                            cal_data['fraction_of_positives'], 
                            marker='o', label='Model calibration')
                    plt.xlabel('Mean Predicted Probability')
                    plt.ylabel('Fraction of Positives')
                    plt.title('Calibration Plot')
                    plt.legend()
                    plt.savefig(plots_path / f'calibration_plot_{timestamp}.png')
                    plt.close()
            
            logger.info(f"Validation plots saved to: {plots_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def run_comprehensive_validation(self, model_name: str, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run comprehensive model validation"""
        logger.info(f"Starting comprehensive validation for model: {model_name}")
        
        # Load model
        model, model_version = self.load_model_from_mlflow(model_name)
        
        # Load test data if not provided
        if test_data is None:
            test_data = self.load_test_data()
        
        # Prepare features and labels
        feature_cols = [col for col in test_data.columns if col not in ['is_bug_fix', 'commit_hash', 'repository', 'timestamp']]
        X = test_data[feature_cols]
        y = test_data['is_bug_fix'].astype(int)
        
        validation_results = {}
        
        try:
            # 1. Basic performance metrics
            logger.info("Calculating performance metrics")
            y_pred = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                y_pred_proba = None
            
            performance_metrics = self.calculate_performance_metrics(y, y_pred, y_pred_proba)
            validation_results['performance_metrics'] = performance_metrics
            
            # 2. Performance regression test
            logger.info("Running performance regression test")
            regression_test = self.performance_regression_test(performance_metrics)
            validation_results['regression_test'] = regression_test
            
            # 3. Cross-validation analysis
            if hasattr(model, 'fit'):  # sklearn models
                logger.info("Running cross-validation analysis")
                cv_results = self.cross_validation_analysis(model, X, y)
                validation_results['cross_validation'] = cv_results
            
            # 4. Stability analysis
            if self.config['validation']['stability_tests']['enabled'] and hasattr(model, 'fit'):
                logger.info("Running stability analysis")
                stability_results = self.stability_analysis(model, X, y)
                validation_results['stability_analysis'] = stability_results
            
            # 5. Calibration analysis
            if y_pred_proba is not None:
                logger.info("Running calibration analysis")
                calibration_results = self.calibration_analysis(y, y_pred_proba)
                validation_results['calibration_analysis'] = calibration_results
            
            # 6. Feature importance analysis
            logger.info("Running feature importance analysis")
            importance_results = self.feature_importance_analysis(model, X, y)
            validation_results['feature_importance'] = importance_results
            
            # 7. Data drift detection (if reference data available)
            if self.config['validation']['drift_detection']['enabled']:
                # Use training data as reference (simplified approach)
                reference_data = X.sample(frac=0.5, random_state=42)
                current_data = X.drop(reference_data.index)
                
                if len(current_data) > 0:
                    logger.info("Running data drift detection")
                    drift_results = self.data_drift_detection(reference_data, current_data)
                    validation_results['drift_analysis'] = drift_results
            
            # 8. Add confusion matrix for binary classification
            if len(np.unique(y)) == 2:
                cm = confusion_matrix(y, y_pred)
                validation_results['confusion_matrix'] = cm.tolist()
            
            # 9. Add model metadata
            validation_results['model_metadata'] = {
                'model_name': model_name,
                'model_version': model_version.version,
                'model_stage': model_version.current_stage,
                'model_tags': model_version.tags,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Log results to MLflow
            with mlflow.start_run(run_name=f"validation_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log performance metrics
                for metric_name, metric_value in performance_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"validation_{metric_name}", metric_value)
                
                # Log validation status
                mlflow.log_metric("validation_passed", int(regression_test['overall_pass']))
                
                # Log artifacts
                validation_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(validation_file, 'w') as f:
                    json.dump(validation_results, f, indent=2, default=str)
                mlflow.log_artifact(validation_file)
                
                # Clean up temporary file
                os.remove(validation_file)
            
            # Generate comprehensive report
            report_file = self.generate_validation_report(model_name, validation_results)
            
            logger.info("Comprehensive validation completed successfully")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['error'] = str(e)
            return validation_results

async def main():
    """Main entry point for model validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Validation Framework")
    parser.add_argument('--model-name', required=True, help='Name of model to validate')
    parser.add_argument('--model-stage', default='Production', help='Model stage (Production, Staging)')
    parser.add_argument('--test-data', help='Path to test data file')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--generate-report', action='store_true', help='Generate validation report')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ModelValidator(args.config)
    
    # Load test data if provided
    test_data = None
    if args.test_data:
        test_data = pd.read_csv(args.test_data)
    
    # Run validation
    try:
        results = validator.run_comprehensive_validation(args.model_name, test_data)
        
        # Print summary
        if 'performance_metrics' in results:
            logger.info("Validation Results Summary:")
            for metric, value in results['performance_metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        if 'regression_test' in results:
            regression_passed = results['regression_test']['overall_pass']
            logger.info(f"Performance Regression Test: {'PASSED' if regression_passed else 'FAILED'}")
        
        # Print overall assessment
        logger.info("Validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())