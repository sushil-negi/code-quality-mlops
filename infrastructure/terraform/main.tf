# MLOps Pipeline - Modular Infrastructure
# This creates a base infrastructure that supports swappable components

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = var.terraform_state_bucket
    key    = "mlops-pipeline/terraform.tfstate"
    region = var.aws_region
  }
}

# Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "MLOps-CodeQuality"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values for consistent naming
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    Owner       = var.owner
  }
  
  # Module configuration based on user selection
  modules_config = {
    data_pipeline = {
      enabled = var.modules.data_pipeline.enabled
      type    = var.modules.data_pipeline.type # kafka, pulsar, kinesis
    }
    ml_platform = {
      enabled = var.modules.ml_platform.enabled
      type    = var.modules.ml_platform.type # mlflow, kubeflow, sagemaker
    }
    monitoring = {
      enabled = var.modules.monitoring.enabled
      type    = var.modules.monitoring.type # prometheus, datadog, cloudwatch
    }
    serving = {
      enabled = var.modules.serving.enabled
      type    = var.modules.serving.type # kubernetes, ecs, cloudrun
    }
  }
}

# Core Network Infrastructure
module "vpc" {
  source = "./modules/networking"
  
  name_prefix = local.name_prefix
  cidr_block  = var.vpc_cidr
  
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
  
  # Subnets configuration
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  enable_nat_gateway = var.enable_nat_gateway
  enable_vpn_gateway = var.enable_vpn_gateway
  
  tags = local.common_tags
}

# Security Groups
module "security_groups" {
  source = "./modules/security"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  
  # CIDR blocks for access control
  allowed_cidr_blocks = var.allowed_cidr_blocks
  office_cidr_blocks  = var.office_cidr_blocks
  
  tags = local.common_tags
}

# Storage Infrastructure
module "storage" {
  source = "./modules/storage"
  
  name_prefix = local.name_prefix
  
  # S3 buckets for different data types
  data_lake_config = {
    raw_data_retention      = var.storage.raw_data_retention
    processed_data_retention = var.storage.processed_data_retention
    model_artifacts_retention = var.storage.model_artifacts_retention
  }
  
  # RDS for metadata
  rds_config = {
    instance_class    = var.storage.rds_instance_class
    allocated_storage = var.storage.rds_storage_gb
    multi_az         = var.storage.rds_multi_az
  }
  
  # Redis for caching
  redis_config = {
    instance_type = var.storage.redis_instance_type
    num_cache_nodes = var.storage.redis_num_nodes
  }
  
  subnet_group_name = module.vpc.database_subnet_group_name
  security_group_ids = [
    module.security_groups.database_sg_id,
    module.security_groups.cache_sg_id
  ]
  
  tags = local.common_tags
}

# Compute Infrastructure (conditional based on serving type)
module "compute" {
  source = "./modules/compute"
  
  name_prefix = local.name_prefix
  
  # EKS Configuration (when kubernetes serving is selected)
  eks_config = local.modules_config.serving.type == "kubernetes" ? {
    enabled        = true
    cluster_version = var.compute.eks_version
    node_groups = var.compute.eks_node_groups
    subnet_ids = module.vpc.private_subnet_ids
    security_group_ids = [module.security_groups.eks_cluster_sg_id]
  } : { enabled = false }
  
  # ECS Configuration (when ecs serving is selected)
  ecs_config = local.modules_config.serving.type == "ecs" ? {
    enabled = true
    capacity_providers = var.compute.ecs_capacity_providers
  } : { enabled = false }
  
  tags = local.common_tags
}

# Conditional Data Pipeline Modules
module "data_pipeline_kafka" {
  count  = local.modules_config.data_pipeline.enabled && local.modules_config.data_pipeline.type == "kafka" ? 1 : 0
  source = "./modules/data-pipeline/kafka"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnet_ids
  
  kafka_config = var.data_pipeline.kafka
  security_group_ids = [module.security_groups.kafka_sg_id]
  
  tags = local.common_tags
}

module "data_pipeline_kinesis" {
  count  = local.modules_config.data_pipeline.enabled && local.modules_config.data_pipeline.type == "kinesis" ? 1 : 0
  source = "./modules/data-pipeline/kinesis"
  
  name_prefix = local.name_prefix
  
  kinesis_config = var.data_pipeline.kinesis
  
  tags = local.common_tags
}

# Conditional ML Platform Modules
module "ml_platform_mlflow" {
  count  = local.modules_config.ml_platform.enabled && local.modules_config.ml_platform.type == "mlflow" ? 1 : 0
  source = "./modules/ml-platform/mlflow"
  
  name_prefix = local.name_prefix
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnet_ids
  
  mlflow_config = var.ml_platform.mlflow
  storage_bucket = module.storage.mlflow_artifacts_bucket
  database_endpoint = module.storage.rds_endpoint
  
  security_group_ids = [module.security_groups.mlflow_sg_id]
  
  tags = local.common_tags
  
  depends_on = [module.compute]
}

module "ml_platform_kubeflow" {
  count  = local.modules_config.ml_platform.enabled && local.modules_config.ml_platform.type == "kubeflow" ? 1 : 0
  source = "./modules/ml-platform/kubeflow"
  
  name_prefix = local.name_prefix
  
  kubeflow_config = var.ml_platform.kubeflow
  cluster_name = local.modules_config.serving.type == "kubernetes" ? module.compute.eks_cluster_name : null
  
  tags = local.common_tags
  
  depends_on = [module.compute]
}

# Conditional Monitoring Modules
module "monitoring_prometheus" {
  count  = local.modules_config.monitoring.enabled && local.modules_config.monitoring.type == "prometheus" ? 1 : 0
  source = "./modules/monitoring/prometheus"
  
  name_prefix = local.name_prefix
  
  prometheus_config = var.monitoring.prometheus
  cluster_name = local.modules_config.serving.type == "kubernetes" ? module.compute.eks_cluster_name : null
  
  tags = local.common_tags
  
  depends_on = [module.compute]
}

module "monitoring_datadog" {
  count  = local.modules_config.monitoring.enabled && local.modules_config.monitoring.type == "datadog" ? 1 : 0
  source = "./modules/monitoring/datadog"
  
  name_prefix = local.name_prefix
  
  datadog_config = var.monitoring.datadog
  datadog_api_key = var.datadog_api_key
  
  tags = local.common_tags
}

# IAM Roles and Policies
module "iam" {
  source = "./modules/iam"
  
  name_prefix = local.name_prefix
  
  # Service roles based on enabled modules
  create_eks_roles = local.modules_config.serving.type == "kubernetes"
  create_ecs_roles = local.modules_config.serving.type == "ecs"
  create_lambda_roles = var.enable_serverless_functions
  
  # Data access roles
  s3_bucket_arns = [
    module.storage.data_lake_bucket_arn,
    module.storage.mlflow_artifacts_bucket_arn
  ]
  
  tags = local.common_tags
}