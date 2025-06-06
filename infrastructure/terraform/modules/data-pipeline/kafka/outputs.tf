# Kafka Data Pipeline Module Outputs

output "cluster_arn" {
  description = "Amazon Resource Name (ARN) of the MSK cluster"
  value       = aws_msk_cluster.main.arn
}

output "cluster_name" {
  description = "Name of the MSK cluster"
  value       = aws_msk_cluster.main.cluster_name
}

output "bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value       = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
  sensitive   = true
}

output "bootstrap_servers_tls" {
  description = "Kafka bootstrap servers (TLS)"
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
  sensitive   = true
}

output "zookeeper_connect_string" {
  description = "Zookeeper connection string"
  value       = aws_msk_cluster.main.zookeeper_connect_string
  sensitive   = true
}

output "kafka_version" {
  description = "Kafka version"
  value       = aws_msk_cluster.main.kafka_version
}

output "current_version" {
  description = "Current version of the MSK Cluster"
  value       = aws_msk_cluster.main.current_version
}

output "encryption_kms_key_id" {
  description = "KMS key ID used for encryption"
  value       = aws_kms_key.kafka.key_id
}

output "logs_bucket" {
  description = "S3 bucket for Kafka logs"
  value       = aws_s3_bucket.kafka_logs.bucket
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for Kafka"
  value       = aws_cloudwatch_log_group.kafka.name
}

output "topic_creator_function_name" {
  description = "Lambda function name for topic creation"
  value       = var.enable_topic_automation ? aws_lambda_function.topic_creator[0].function_name : null
}

output "default_topics" {
  description = "Default topics configuration"
  value       = var.default_topics
}

output "cluster_configuration" {
  description = "Kafka cluster configuration details"
  value = {
    num_brokers         = var.kafka_config.num_brokers
    instance_type       = var.kafka_config.broker_instance_type
    ebs_volume_size     = var.kafka_config.ebs_volume_size
    kafka_version       = var.kafka_config.kafka_version
    encryption_enabled  = true
    enhanced_monitoring = true
  }
}

output "connection_info" {
  description = "Connection information for applications"
  value = {
    bootstrap_servers = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
    security_protocol = "SASL_SSL"
    sasl_mechanism   = "AWS_MSK_IAM"
    ssl_ca_location  = "/tmp/kafka-ca-cert"
    
    # Environment variables for applications
    env_vars = {
      KAFKA_BOOTSTRAP_SERVERS = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
      KAFKA_SECURITY_PROTOCOL = "SASL_SSL"
      KAFKA_SASL_MECHANISM   = "AWS_MSK_IAM"
    }
  }
  sensitive = true
}