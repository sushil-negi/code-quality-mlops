#!/usr/bin/env python3
"""
Deployment Manager - Automated Model Deployment Pipeline

This module handles automated deployment of trained models to various serving environments:
- Kubernetes deployment with auto-scaling
- Docker containerization
- AWS ECS/Fargate deployment
- Lambda serverless deployment
- Blue-green deployment strategies
"""

import os
import json
import yaml
import logging
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict

import boto3
import docker
from kubernetes import client, config
import mlflow
from mlflow.tracking import MlflowClient
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    model_version: str
    deployment_target: str  # 'kubernetes', 'ecs', 'lambda', 'local'
    environment: str  # 'dev', 'staging', 'prod'
    
    # Resource configuration
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    
    # Scaling configuration
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Deployment strategy
    strategy: str = "rolling"  # 'rolling', 'blue_green', 'canary'
    
    # Health check configuration
    health_check_path: str = "/health"
    readiness_probe_delay: int = 30
    liveness_probe_delay: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ContainerBuilder:
    """Builds Docker containers for model serving"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
    
    def build_serving_image(self, model_name: str, model_version: str) -> str:
        """Build Docker image for model serving"""
        logger.info(f"Building Docker image for {model_name} v{model_version}")
        
        # Create temporary directory for build context
        build_dir = Path(f"build/{model_name}-{model_version}")
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Dockerfile
        dockerfile_content = self._generate_dockerfile(model_name, model_version)
        dockerfile_path = build_dir / "Dockerfile"
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Copy application code
        self._copy_application_code(build_dir)
        
        # Copy model artifacts
        self._copy_model_artifacts(build_dir, model_name, model_version)
        
        # Build image
        image_tag = f"{model_name}:{model_version}"
        
        try:
            image, logs = self.docker_client.images.build(
                path=str(build_dir),
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            logger.info(f"Successfully built image: {image_tag}")
            
            # Push to registry if configured
            if self.config.get('container_registry'):
                self._push_to_registry(image_tag)
            
            return image_tag
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise
    
    def _generate_dockerfile(self, model_name: str, model_version: str) -> str:
        """Generate Dockerfile for model serving"""
        return f"""
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Copy model artifacts
COPY models/ ./models/

# Set environment variables
ENV MODEL_NAME={model_name}
ENV MODEL_VERSION={model_version}
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.model_serving:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    def _copy_application_code(self, build_dir: Path):
        """Copy application source code to build directory"""
        import shutil
        
        # Copy source code
        src_dir = Path("src")
        if src_dir.exists():
            shutil.copytree(src_dir, build_dir / "src", dirs_exist_ok=True)
        
        # Copy configuration
        config_dir = Path("config")
        if config_dir.exists():
            shutil.copytree(config_dir, build_dir / "config", dirs_exist_ok=True)
        
        # Generate requirements.txt
        requirements = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "mlflow==2.8.1",
            "pandas==2.1.3",
            "numpy==1.24.3",
            "scikit-learn==1.3.2",
            "torch==2.1.1",
            "prometheus-client==0.19.0",
            "redis==5.0.1",
            "pydantic==2.5.0",
            "boto3==1.34.0"
        ]
        
        with open(build_dir / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
    
    def _copy_model_artifacts(self, build_dir: Path, model_name: str, model_version: str):
        """Copy model artifacts to build directory"""
        # Create models directory
        models_dir = build_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Download model from MLflow
        try:
            mlflow_client = MlflowClient()
            model_uri = f"models:/{model_name}/{model_version}"
            
            # Download model to temporary location
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Copy to build directory
            import shutil
            shutil.copytree(local_path, models_dir / f"{model_name}-{model_version}", dirs_exist_ok=True)
            
            logger.info(f"Model artifacts copied for {model_name} v{model_version}")
            
        except Exception as e:
            logger.warning(f"Failed to copy model artifacts: {e}")
    
    def _push_to_registry(self, image_tag: str):
        """Push image to container registry"""
        registry = self.config['container_registry']
        
        # Tag for registry
        registry_tag = f"{registry['url']}/{registry['namespace']}/{image_tag}"
        
        # Tag image
        image = self.docker_client.images.get(image_tag)
        image.tag(registry_tag)
        
        # Push to registry
        try:
            push_logs = self.docker_client.images.push(registry_tag)
            logger.info(f"Pushed image to registry: {registry_tag}")
            return registry_tag
            
        except Exception as e:
            logger.error(f"Failed to push to registry: {e}")
            raise

class KubernetesDeployer:
    """Deploys models to Kubernetes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load Kubernetes configuration
        try:
            if os.path.exists(os.path.expanduser("~/.kube/config")):
                config.load_kube_config()
            else:
                config.load_incluster_config()
            
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_autoscaling_v2 = client.AutoscalingV2Api()
            
            logger.info("Kubernetes client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def deploy_model(self, deployment_config: DeploymentConfig, image_tag: str) -> Dict[str, Any]:
        """Deploy model to Kubernetes"""
        logger.info(f"Deploying {deployment_config.model_name} to Kubernetes")
        
        namespace = f"mlops-{deployment_config.environment}"
        app_name = f"{deployment_config.model_name}-serving"
        
        try:
            # Create namespace if it doesn't exist
            self._ensure_namespace(namespace)
            
            # Deploy based on strategy
            if deployment_config.strategy == "blue_green":
                return self._blue_green_deploy(deployment_config, image_tag, namespace, app_name)
            elif deployment_config.strategy == "canary":
                return self._canary_deploy(deployment_config, image_tag, namespace, app_name)
            else:
                return self._rolling_deploy(deployment_config, image_tag, namespace, app_name)
                
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    def _rolling_deploy(self, deployment_config: DeploymentConfig, image_tag: str, 
                       namespace: str, app_name: str) -> Dict[str, Any]:
        """Perform rolling deployment"""
        
        # Create or update deployment
        deployment = self._create_deployment_manifest(deployment_config, image_tag, app_name)
        
        try:
            # Try to get existing deployment
            existing_deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=app_name, namespace=namespace
            )
            
            # Update existing deployment
            deployment.metadata.resource_version = existing_deployment.metadata.resource_version
            updated_deployment = self.k8s_apps_v1.replace_namespaced_deployment(
                name=app_name, namespace=namespace, body=deployment
            )
            
            logger.info(f"Updated deployment: {app_name}")
            
        except client.ApiException as e:
            if e.status == 404:
                # Create new deployment
                created_deployment = self.k8s_apps_v1.create_namespaced_deployment(
                    namespace=namespace, body=deployment
                )
                logger.info(f"Created deployment: {app_name}")
            else:
                raise
        
        # Create or update service
        service = self._create_service_manifest(deployment_config, app_name)
        self._ensure_service(service, namespace, app_name)
        
        # Create or update HPA
        hpa = self._create_hpa_manifest(deployment_config, app_name)
        self._ensure_hpa(hpa, namespace, app_name)
        
        # Wait for deployment to be ready
        self._wait_for_deployment(namespace, app_name)
        
        return {
            "deployment_name": app_name,
            "namespace": namespace,
            "image": image_tag,
            "strategy": "rolling",
            "status": "deployed"
        }
    
    def _blue_green_deploy(self, deployment_config: DeploymentConfig, image_tag: str,
                          namespace: str, app_name: str) -> Dict[str, Any]:
        """Perform blue-green deployment"""
        
        # Determine current and new versions
        current_color = self._get_current_color(namespace, app_name)
        new_color = "blue" if current_color == "green" else "green"
        
        new_app_name = f"{app_name}-{new_color}"
        
        # Deploy new version
        deployment = self._create_deployment_manifest(deployment_config, image_tag, new_app_name)
        
        self.k8s_apps_v1.create_namespaced_deployment(
            namespace=namespace, body=deployment
        )
        
        # Wait for new deployment to be ready
        self._wait_for_deployment(namespace, new_app_name)
        
        # Update service to point to new deployment
        service = self._create_service_manifest(deployment_config, app_name)
        service.spec.selector = {"app": new_app_name}
        
        self.k8s_core_v1.replace_namespaced_service(
            name=app_name, namespace=namespace, body=service
        )
        
        # Clean up old deployment
        old_app_name = f"{app_name}-{current_color}"
        try:
            self.k8s_apps_v1.delete_namespaced_deployment(
                name=old_app_name, namespace=namespace
            )
        except client.ApiException:
            pass  # Old deployment might not exist
        
        return {
            "deployment_name": new_app_name,
            "namespace": namespace,
            "image": image_tag,
            "strategy": "blue_green",
            "active_color": new_color,
            "status": "deployed"
        }
    
    def _canary_deploy(self, deployment_config: DeploymentConfig, image_tag: str,
                      namespace: str, app_name: str) -> Dict[str, Any]:
        """Perform canary deployment"""
        
        canary_app_name = f"{app_name}-canary"
        
        # Deploy canary version with reduced replicas
        canary_config = deployment_config
        canary_config.min_replicas = 1
        canary_config.max_replicas = 2
        
        deployment = self._create_deployment_manifest(canary_config, image_tag, canary_app_name)
        
        self.k8s_apps_v1.create_namespaced_deployment(
            namespace=namespace, body=deployment
        )
        
        # Wait for canary deployment
        self._wait_for_deployment(namespace, canary_app_name)
        
        return {
            "deployment_name": canary_app_name,
            "namespace": namespace,
            "image": image_tag,
            "strategy": "canary",
            "canary_replicas": 1,
            "status": "canary_deployed"
        }
    
    def _create_deployment_manifest(self, deployment_config: DeploymentConfig, 
                                  image_tag: str, app_name: str) -> client.V1Deployment:
        """Create Kubernetes deployment manifest"""
        
        # Container specification
        container = client.V1Container(
            name="model-serving",
            image=image_tag,
            ports=[client.V1ContainerPort(container_port=8000)],
            env=[
                client.V1EnvVar(name="MODEL_NAME", value=deployment_config.model_name),
                client.V1EnvVar(name="MODEL_VERSION", value=deployment_config.model_version),
                client.V1EnvVar(name="ENVIRONMENT", value=deployment_config.environment)
            ],
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": deployment_config.cpu_request,
                    "memory": deployment_config.memory_request
                },
                limits={
                    "cpu": deployment_config.cpu_limit,
                    "memory": deployment_config.memory_limit
                }
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=deployment_config.health_check_path,
                    port=8000
                ),
                initial_delay_seconds=deployment_config.readiness_probe_delay,
                period_seconds=10
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=deployment_config.health_check_path,
                    port=8000
                ),
                initial_delay_seconds=deployment_config.liveness_probe_delay,
                period_seconds=30
            )
        )
        
        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"app": app_name}
            ),
            spec=client.V1PodSpec(
                containers=[container]
            )
        )
        
        # Deployment spec
        spec = client.V1DeploymentSpec(
            replicas=deployment_config.min_replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": app_name}
            ),
            template=template,
            strategy=client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge="25%",
                    max_unavailable="25%"
                )
            )
        )
        
        # Deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=app_name,
                labels={"app": app_name}
            ),
            spec=spec
        )
        
        return deployment
    
    def _create_service_manifest(self, deployment_config: DeploymentConfig, 
                               app_name: str) -> client.V1Service:
        """Create Kubernetes service manifest"""
        
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=app_name,
                labels={"app": app_name}
            ),
            spec=client.V1ServiceSpec(
                selector={"app": app_name},
                ports=[
                    client.V1ServicePort(
                        port=80,
                        target_port=8000,
                        protocol="TCP"
                    )
                ],
                type="LoadBalancer"
            )
        )
        
        return service
    
    def _create_hpa_manifest(self, deployment_config: DeploymentConfig, 
                           app_name: str) -> client.V2HorizontalPodAutoscaler:
        """Create Kubernetes HPA manifest"""
        
        hpa = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name=app_name,
                labels={"app": app_name}
            ),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=app_name
                ),
                min_replicas=deployment_config.min_replicas,
                max_replicas=deployment_config.max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=deployment_config.target_cpu_utilization
                            )
                        )
                    )
                ]
            )
        )
        
        return hpa
    
    def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists"""
        try:
            self.k8s_core_v1.read_namespace(namespace)
        except client.ApiException as e:
            if e.status == 404:
                # Create namespace
                ns = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.k8s_core_v1.create_namespace(ns)
                logger.info(f"Created namespace: {namespace}")
    
    def _ensure_service(self, service: client.V1Service, namespace: str, app_name: str):
        """Ensure service exists"""
        try:
            self.k8s_core_v1.read_namespaced_service(name=app_name, namespace=namespace)
            # Update existing service
            self.k8s_core_v1.replace_namespaced_service(
                name=app_name, namespace=namespace, body=service
            )
        except client.ApiException as e:
            if e.status == 404:
                # Create new service
                self.k8s_core_v1.create_namespaced_service(
                    namespace=namespace, body=service
                )
                logger.info(f"Created service: {app_name}")
    
    def _ensure_hpa(self, hpa: client.V2HorizontalPodAutoscaler, namespace: str, app_name: str):
        """Ensure HPA exists"""
        try:
            self.k8s_autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                name=app_name, namespace=namespace
            )
            # Update existing HPA
            self.k8s_autoscaling_v2.replace_namespaced_horizontal_pod_autoscaler(
                name=app_name, namespace=namespace, body=hpa
            )
        except client.ApiException as e:
            if e.status == 404:
                # Create new HPA
                self.k8s_autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                    namespace=namespace, body=hpa
                )
                logger.info(f"Created HPA: {app_name}")
    
    def _wait_for_deployment(self, namespace: str, app_name: str, timeout: int = 300):
        """Wait for deployment to be ready"""
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=app_name, namespace=namespace
                )
                
                status = deployment.status
                if (status.ready_replicas and 
                    status.ready_replicas == status.replicas and
                    status.updated_replicas == status.replicas):
                    logger.info(f"Deployment {app_name} is ready")
                    return
                
                logger.info(f"Waiting for deployment {app_name} to be ready...")
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking deployment status: {e}")
                time.sleep(10)
        
        raise TimeoutError(f"Deployment {app_name} did not become ready within {timeout} seconds")
    
    def _get_current_color(self, namespace: str, app_name: str) -> str:
        """Get current active color for blue-green deployment"""
        try:
            service = self.k8s_core_v1.read_namespaced_service(
                name=app_name, namespace=namespace
            )
            
            selector_app = service.spec.selector.get("app", "")
            if "blue" in selector_app:
                return "blue"
            elif "green" in selector_app:
                return "green"
            else:
                return "blue"  # Default to blue
                
        except client.ApiException:
            return "blue"  # Default to blue if service doesn't exist

class DeploymentManager:
    """Main deployment manager orchestrating the deployment process"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.container_builder = ContainerBuilder(self.config)
        
        # Initialize deployers based on configuration
        self.deployers = {}
        
        if self.config.get('kubernetes', {}).get('enabled', False):
            self.deployers['kubernetes'] = KubernetesDeployer(self.config)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'container_registry': {
                'url': os.getenv('CONTAINER_REGISTRY_URL', 'localhost:5000'),
                'namespace': os.getenv('CONTAINER_REGISTRY_NAMESPACE', 'mlops')
            },
            'kubernetes': {
                'enabled': os.getenv('KUBERNETES_ENABLED', 'true').lower() == 'true'
            },
            'mlflow': {
                'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            }
        }
    
    async def deploy_model(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy a model to the specified target"""
        logger.info(f"Starting deployment: {deployment_config.model_name} v{deployment_config.model_version}")
        
        try:
            # Validate model exists in MLflow
            await self._validate_model(deployment_config.model_name, deployment_config.model_version)
            
            # Build container image
            image_tag = self.container_builder.build_serving_image(
                deployment_config.model_name,
                deployment_config.model_version
            )
            
            # Deploy to target environment
            if deployment_config.deployment_target not in self.deployers:
                raise ValueError(f"Unsupported deployment target: {deployment_config.deployment_target}")
            
            deployer = self.deployers[deployment_config.deployment_target]
            deployment_result = deployer.deploy_model(deployment_config, image_tag)
            
            # Record deployment in MLflow
            await self._record_deployment(deployment_config, deployment_result)
            
            logger.info(f"Deployment completed successfully: {deployment_result}")
            
            return {
                'status': 'success',
                'deployment_config': deployment_config.to_dict(),
                'image_tag': image_tag,
                'deployment_result': deployment_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'deployment_config': deployment_config.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _validate_model(self, model_name: str, model_version: str):
        """Validate that the model exists in MLflow"""
        try:
            mlflow_client = MlflowClient(self.config['mlflow']['tracking_uri'])
            model_version_obj = mlflow_client.get_model_version(model_name, model_version)
            
            if model_version_obj.current_stage not in ['Production', 'Staging']:
                logger.warning(f"Model {model_name} v{model_version} is not in Production or Staging stage")
            
            logger.info(f"Model validation successful: {model_name} v{model_version}")
            
        except Exception as e:
            raise ValueError(f"Model validation failed: {e}")
    
    async def _record_deployment(self, deployment_config: DeploymentConfig, 
                               deployment_result: Dict[str, Any]):
        """Record deployment information in MLflow"""
        try:
            with mlflow.start_run(run_name=f"deployment_{deployment_config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log deployment parameters
                mlflow.log_param("model_name", deployment_config.model_name)
                mlflow.log_param("model_version", deployment_config.model_version)
                mlflow.log_param("deployment_target", deployment_config.deployment_target)
                mlflow.log_param("environment", deployment_config.environment)
                mlflow.log_param("strategy", deployment_config.strategy)
                
                # Log deployment result
                mlflow.log_dict(deployment_result, "deployment_result.json")
                
                # Tag the run
                mlflow.set_tag("deployment", "true")
                mlflow.set_tag("environment", deployment_config.environment)
                
                logger.info("Deployment recorded in MLflow")
                
        except Exception as e:
            logger.warning(f"Failed to record deployment in MLflow: {e}")

async def main():
    """Main entry point for deployment manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Deployment Manager")
    parser.add_argument('--model-name', required=True, help='Name of model to deploy')
    parser.add_argument('--model-version', required=True, help='Version of model to deploy')
    parser.add_argument('--target', choices=['kubernetes', 'ecs', 'lambda'], 
                       default='kubernetes', help='Deployment target')
    parser.add_argument('--environment', choices=['dev', 'staging', 'prod'], 
                       default='dev', help='Deployment environment')
    parser.add_argument('--strategy', choices=['rolling', 'blue_green', 'canary'], 
                       default='rolling', help='Deployment strategy')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create deployment configuration
    deployment_config = DeploymentConfig(
        model_name=args.model_name,
        model_version=args.model_version,
        deployment_target=args.target,
        environment=args.environment,
        strategy=args.strategy
    )
    
    # Initialize deployment manager
    manager = DeploymentManager(args.config)
    
    # Deploy model
    try:
        result = await manager.deploy_model(deployment_config)
        
        if result['status'] == 'success':
            logger.info("Deployment completed successfully!")
            print(json.dumps(result, indent=2))
        else:
            logger.error("Deployment failed!")
            print(json.dumps(result, indent=2))
            exit(1)
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())