# Infrastructure Variables for Modular MLOps Pipeline

# Core Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "code-quality-mlops"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "mlops-team"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway"
  type        = bool
  default     = false
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access resources"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "office_cidr_blocks" {
  description = "Office CIDR blocks for admin access"
  type        = list(string)
  default     = []
}

# MODULE SELECTION CONFIGURATION
# This is the key to modularity - users can enable/disable and choose implementations
variable "modules" {
  description = "Configuration for which modules to enable and their types"
  type = object({
    data_pipeline = object({
      enabled = bool
      type    = string # kafka, pulsar, kinesis, pubsub
    })
    ml_platform = object({
      enabled = bool
      type    = string # mlflow, kubeflow, sagemaker, vertex
    })
    monitoring = object({
      enabled = bool
      type    = string # prometheus, datadog, cloudwatch, newrelic
    })
    serving = object({
      enabled = bool
      type    = string # kubernetes, ecs, cloudrun, lambda
    })
  })
  
  default = {
    data_pipeline = {
      enabled = true
      type    = "kafka"
    }
    ml_platform = {
      enabled = true
      type    = "mlflow"
    }
    monitoring = {
      enabled = true
      type    = "prometheus"
    }
    serving = {
      enabled = true
      type    = "kubernetes"
    }
  }
  
  validation {
    condition = contains(["kafka", "pulsar", "kinesis", "pubsub"], var.modules.data_pipeline.type)
    error_message = "Data pipeline type must be: kafka, pulsar, kinesis, or pubsub."
  }
  
  validation {
    condition = contains(["mlflow", "kubeflow", "sagemaker", "vertex"], var.modules.ml_platform.type)
    error_message = "ML platform type must be: mlflow, kubeflow, sagemaker, or vertex."
  }
  
  validation {
    condition = contains(["prometheus", "datadog", "cloudwatch", "newrelic"], var.modules.monitoring.type)
    error_message = "Monitoring type must be: prometheus, datadog, cloudwatch, or newrelic."
  }
  
  validation {
    condition = contains(["kubernetes", "ecs", "cloudrun", "lambda"], var.modules.serving.type)
    error_message = "Serving type must be: kubernetes, ecs, cloudrun, or lambda."
  }
}

# Storage Configuration
variable "storage" {
  description = "Storage configuration"
  type = object({
    # Data lake retention policies
    raw_data_retention      = number
    processed_data_retention = number
    model_artifacts_retention = number
    
    # RDS configuration
    rds_instance_class = string
    rds_storage_gb    = number
    rds_multi_az      = bool
    
    # Redis configuration
    redis_instance_type = string
    redis_num_nodes    = number
  })
  
  default = {
    raw_data_retention        = 90
    processed_data_retention  = 365
    model_artifacts_retention = 1095
    
    rds_instance_class = "db.t3.medium"
    rds_storage_gb    = 100
    rds_multi_az      = false
    
    redis_instance_type = "cache.t3.medium"
    redis_num_nodes    = 1
  }
}

# Compute Configuration
variable "compute" {
  description = "Compute configuration for different serving types"
  type = object({
    # EKS Configuration (for kubernetes serving)
    eks_version = string
    eks_node_groups = map(object({
      instance_types = list(string)
      scaling_config = object({
        desired_size = number
        max_size     = number
        min_size     = number
      })
      capacity_type = string # ON_DEMAND or SPOT
    }))
    
    # ECS Configuration (for ecs serving)
    ecs_capacity_providers = list(string)
  })
  
  default = {
    eks_version = "1.27"
    eks_node_groups = {
      general = {
        instance_types = ["t3.medium"]
        scaling_config = {
          desired_size = 2
          max_size     = 10
          min_size     = 1
        }
        capacity_type = "ON_DEMAND"
      }
      ml_workloads = {
        instance_types = ["m5.large", "m5.xlarge"]
        scaling_config = {
          desired_size = 1
          max_size     = 5
          min_size     = 0
        }
        capacity_type = "SPOT"
      }
    }
    
    ecs_capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  }
}

# Data Pipeline Specific Configurations
variable "data_pipeline" {
  description = "Configuration for different data pipeline types"
  type = object({
    kafka = object({
      broker_instance_type = string
      num_brokers         = number
      ebs_volume_size     = number
      kafka_version       = string
    })
    
    kinesis = object({
      shard_count = number
      retention_period = number
    })
    
    pulsar = object({
      broker_instance_type = string
      num_brokers         = number
      zookeeper_instance_type = string
      num_zookeepers     = number
    })
  })
  
  default = {
    kafka = {
      broker_instance_type = "kafka.t3.small"
      num_brokers         = 3
      ebs_volume_size     = 100
      kafka_version       = "2.8.1"
    }
    
    kinesis = {
      shard_count = 2
      retention_period = 24
    }
    
    pulsar = {
      broker_instance_type    = "t3.medium"
      num_brokers           = 3
      zookeeper_instance_type = "t3.small"
      num_zookeepers        = 3
    }
  }
}

# ML Platform Specific Configurations
variable "ml_platform" {
  description = "Configuration for different ML platforms"
  type = object({
    mlflow = object({
      instance_type = string
      storage_size  = number
      enable_auth   = bool
    })
    
    kubeflow = object({
      version = string
      enable_istio = bool
      enable_cert_manager = bool
    })
    
    sagemaker = object({
      enable_studio = bool
      enable_projects = bool
    })
  })
  
  default = {
    mlflow = {
      instance_type = "t3.medium"
      storage_size  = 100
      enable_auth   = true
    }
    
    kubeflow = {
      version = "1.7.0"
      enable_istio = true
      enable_cert_manager = true
    }
    
    sagemaker = {
      enable_studio = true
      enable_projects = true
    }
  }
}

# Monitoring Specific Configurations
variable "monitoring" {
  description = "Configuration for different monitoring solutions"
  type = object({
    prometheus = object({
      storage_size = number
      retention_days = number
      enable_grafana = bool
      enable_alertmanager = bool
    })
    
    datadog = object({
      enable_apm = bool
      enable_logs = bool
      enable_synthetics = bool
    })
    
    cloudwatch = object({
      log_retention_days = number
      enable_insights = bool
    })
  })
  
  default = {
    prometheus = {
      storage_size = 50
      retention_days = 15
      enable_grafana = true
      enable_alertmanager = true
    }
    
    datadog = {
      enable_apm = true
      enable_logs = true
      enable_synthetics = false
    }
    
    cloudwatch = {
      log_retention_days = 30
      enable_insights = true
    }
  }
}

# External Service Configurations
variable "datadog_api_key" {
  description = "Datadog API key (required if monitoring.type = datadog)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_serverless_functions" {
  description = "Enable Lambda functions for serverless components"
  type        = bool
  default     = false
}

# Cost Optimization Features
variable "cost_optimization" {
  description = "Cost optimization features"
  type = object({
    enable_spot_instances = bool
    enable_scheduled_scaling = bool
    enable_rightsizing = bool
  })
  
  default = {
    enable_spot_instances = true
    enable_scheduled_scaling = true
    enable_rightsizing = true
  }
}