# Output values for the modular MLOps infrastructure

# Network Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

# Storage Outputs
output "data_lake_bucket" {
  description = "Name of the data lake S3 bucket"
  value       = module.storage.data_lake_bucket_name
}

output "mlflow_artifacts_bucket" {
  description = "Name of the MLflow artifacts S3 bucket"
  value       = module.storage.mlflow_artifacts_bucket_name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.storage.rds_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.storage.redis_endpoint
  sensitive   = true
}

# Compute Outputs (Conditional based on serving type)
output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = var.modules.serving.type == "kubernetes" ? module.compute.eks_cluster_endpoint : null
  sensitive   = true
}

output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = var.modules.serving.type == "kubernetes" ? module.compute.eks_cluster_name : null
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = var.modules.serving.type == "ecs" ? module.compute.ecs_cluster_name : null
}

# Data Pipeline Outputs (Conditional)
output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value = var.modules.data_pipeline.enabled && var.modules.data_pipeline.type == "kafka" ? (
    length(module.data_pipeline_kafka) > 0 ? module.data_pipeline_kafka[0].bootstrap_servers : null
  ) : null
  sensitive = true
}

output "kinesis_stream_names" {
  description = "Kinesis stream names"
  value = var.modules.data_pipeline.enabled && var.modules.data_pipeline.type == "kinesis" ? (
    length(module.data_pipeline_kinesis) > 0 ? module.data_pipeline_kinesis[0].stream_names : null
  ) : null
}

# ML Platform Outputs (Conditional)
output "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  value = var.modules.ml_platform.enabled && var.modules.ml_platform.type == "mlflow" ? (
    length(module.ml_platform_mlflow) > 0 ? module.ml_platform_mlflow[0].tracking_uri : null
  ) : null
  sensitive = true
}

output "kubeflow_dashboard_url" {
  description = "Kubeflow dashboard URL"
  value = var.modules.ml_platform.enabled && var.modules.ml_platform.type == "kubeflow" ? (
    length(module.ml_platform_kubeflow) > 0 ? module.ml_platform_kubeflow[0].dashboard_url : null
  ) : null
  sensitive = true
}

# Monitoring Outputs (Conditional)
output "prometheus_endpoint" {
  description = "Prometheus server endpoint"
  value = var.modules.monitoring.enabled && var.modules.monitoring.type == "prometheus" ? (
    length(module.monitoring_prometheus) > 0 ? module.monitoring_prometheus[0].prometheus_endpoint : null
  ) : null
  sensitive = true
}

output "grafana_dashboard_url" {
  description = "Grafana dashboard URL"
  value = var.modules.monitoring.enabled && var.modules.monitoring.type == "prometheus" ? (
    length(module.monitoring_prometheus) > 0 ? module.monitoring_prometheus[0].grafana_url : null
  ) : null
  sensitive = true
}

# Security Outputs
output "security_group_ids" {
  description = "Map of security group IDs"
  value = {
    eks_cluster = module.security_groups.eks_cluster_sg_id
    database   = module.security_groups.database_sg_id
    cache      = module.security_groups.cache_sg_id
    kafka      = module.security_groups.kafka_sg_id
    mlflow     = module.security_groups.mlflow_sg_id
  }
}

# IAM Outputs
output "iam_role_arns" {
  description = "ARNs of created IAM roles"
  value = {
    eks_cluster_role    = module.iam.eks_cluster_role_arn
    eks_node_group_role = module.iam.eks_node_group_role_arn
    mlflow_role        = module.iam.mlflow_role_arn
    lambda_role        = module.iam.lambda_role_arn
  }
}

# Configuration Summary
output "deployed_modules" {
  description = "Summary of deployed modules"
  value = {
    data_pipeline = {
      enabled = var.modules.data_pipeline.enabled
      type    = var.modules.data_pipeline.type
    }
    ml_platform = {
      enabled = var.modules.ml_platform.enabled
      type    = var.modules.ml_platform.type
    }
    monitoring = {
      enabled = var.modules.monitoring.enabled
      type    = var.modules.monitoring.type
    }
    serving = {
      enabled = var.modules.serving.enabled
      type    = var.modules.serving.type
    }
  }
}

# Connection Information for Applications
output "connection_info" {
  description = "Connection information for deployed services"
  value = {
    # Data Pipeline connections
    data_pipeline_endpoint = var.modules.data_pipeline.enabled ? (
      var.modules.data_pipeline.type == "kafka" ? (
        length(module.data_pipeline_kafka) > 0 ? module.data_pipeline_kafka[0].bootstrap_servers : null
      ) : var.modules.data_pipeline.type == "kinesis" ? (
        length(module.data_pipeline_kinesis) > 0 ? module.data_pipeline_kinesis[0].stream_names : null
      ) : null
    ) : null
    
    # ML Platform connections
    ml_platform_endpoint = var.modules.ml_platform.enabled ? (
      var.modules.ml_platform.type == "mlflow" ? (
        length(module.ml_platform_mlflow) > 0 ? module.ml_platform_mlflow[0].tracking_uri : null
      ) : var.modules.ml_platform.type == "kubeflow" ? (
        length(module.ml_platform_kubeflow) > 0 ? module.ml_platform_kubeflow[0].dashboard_url : null
      ) : null
    ) : null
    
    # Storage connections
    storage = {
      data_lake_bucket = module.storage.data_lake_bucket_name
      database_endpoint = module.storage.rds_endpoint
      cache_endpoint = module.storage.redis_endpoint
    }
    
    # Compute connections
    compute_endpoint = var.modules.serving.type == "kubernetes" ? module.compute.eks_cluster_endpoint : (
      var.modules.serving.type == "ecs" ? module.compute.ecs_cluster_name : null
    )
  }
  sensitive = true
}

# Cost Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    compute = var.modules.serving.type == "kubernetes" ? "~$300-800" : "~$200-500"
    storage = "~$100-300"
    data_pipeline = var.modules.data_pipeline.type == "kafka" ? "~$200-400" : "~$50-150"
    ml_platform = var.modules.ml_platform.type == "mlflow" ? "~$100-200" : "~$300-500"
    monitoring = var.modules.monitoring.type == "prometheus" ? "~$50-100" : "~$200-400"
    total_estimated = "~$750-2000"
  }
}