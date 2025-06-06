# Kafka Data Pipeline Module Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where Kafka will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for Kafka brokers"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for Kafka cluster"
  type        = list(string)
}

variable "kafka_config" {
  description = "Kafka cluster configuration"
  type = object({
    broker_instance_type = string
    num_brokers         = number
    ebs_volume_size     = number
    kafka_version       = string
  })
  
  default = {
    broker_instance_type = "kafka.t3.small"
    num_brokers         = 3
    ebs_volume_size     = 100
    kafka_version       = "2.8.1"
  }
}

variable "enable_topic_automation" {
  description = "Enable automatic topic creation via Lambda"
  type        = bool
  default     = true
}

variable "default_topics" {
  description = "Default topics to create"
  type = list(object({
    name               = string
    num_partitions     = number
    replication_factor = number
    config             = map(string)
  }))
  
  default = [
    {
      name               = "code-commits"
      num_partitions     = 10
      replication_factor = 3
      config = {
        "retention.ms" = "604800000"  # 7 days
        "compression.type" = "lz4"
      }
    },
    {
      name               = "code-metrics"
      num_partitions     = 5
      replication_factor = 3
      config = {
        "retention.ms" = "2592000000"  # 30 days
        "compression.type" = "lz4"
      }
    },
    {
      name               = "bug-reports"
      num_partitions     = 3
      replication_factor = 3
      config = {
        "retention.ms" = "7776000000"  # 90 days
        "compression.type" = "lz4"
      }
    },
    {
      name               = "model-predictions"
      num_partitions     = 5
      replication_factor = 3
      config = {
        "retention.ms" = "604800000"  # 7 days
        "compression.type" = "snappy"
      }
    }
  ]
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}