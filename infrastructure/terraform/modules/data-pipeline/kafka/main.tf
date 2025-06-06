# Kafka Data Pipeline Module

# MSK Cluster Configuration
resource "aws_msk_configuration" "main" {
  kafka_versions = [var.kafka_config.kafka_version]
  name           = "${var.name_prefix}-kafka-config"

  server_properties = <<PROPERTIES
auto.create.topics.enable=true
default.replication.factor=3
min.insync.replicas=2
num.network.threads=8
num.io.threads=16
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
num.partitions=3
num.recovery.threads.per.data.dir=1
offsets.topic.replication.factor=3
transaction.state.log.replication.factor=3
transaction.state.log.min.isr=2
log.retention.hours=168
log.segment.bytes=1073741824
log.retention.check.interval.ms=300000
log.cleanup.policy=delete
compression.type=producer
PROPERTIES
}

# MSK Cluster
resource "aws_msk_cluster" "main" {
  cluster_name           = "${var.name_prefix}-kafka"
  kafka_version          = var.kafka_config.kafka_version
  number_of_broker_nodes = var.kafka_config.num_brokers
  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  broker_node_group_info {
    instance_type   = var.kafka_config.broker_instance_type
    client_subnets  = var.subnet_ids
    security_groups = var.security_group_ids
    
    storage_info {
      ebs_storage_info {
        volume_size = var.kafka_config.ebs_volume_size
      }
    }
  }

  # Encryption settings
  encryption_info {
    encryption_at_rest_kms_key_id = aws_kms_key.kafka.arn
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  # Enhanced monitoring
  enhanced_monitoring = "PER_TOPIC_PER_BROKER"

  # Logging
  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.kafka.name
      }
      firehose {
        enabled         = false
      }
      s3 {
        enabled = true
        bucket  = aws_s3_bucket.kafka_logs.bucket
        prefix  = "kafka-logs/"
      }
    }
  }

  # Client authentication
  client_authentication {
    unauthenticated = false
    sasl {
      scram = true
      iam   = true
    }
    tls {
      certificate_authority_arns = []
    }
  }

  tags = var.tags
}

# KMS Key for Kafka encryption
resource "aws_kms_key" "kafka" {
  description             = "KMS key for Kafka encryption"
  deletion_window_in_days = 7

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-kafka-kms"
  })
}

resource "aws_kms_alias" "kafka" {
  name          = "alias/${var.name_prefix}-kafka"
  target_key_id = aws_kms_key.kafka.key_id
}

# CloudWatch Log Group for Kafka
resource "aws_cloudwatch_log_group" "kafka" {
  name              = "/aws/msk/${var.name_prefix}-kafka"
  retention_in_days = 14

  tags = var.tags
}

# S3 Bucket for Kafka logs
resource "aws_s3_bucket" "kafka_logs" {
  bucket = "${var.name_prefix}-kafka-logs-${random_id.bucket_suffix.hex}"

  tags = var.tags
}

resource "aws_s3_bucket_versioning" "kafka_logs" {
  bucket = aws_s3_bucket.kafka_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "kafka_logs" {
  bucket = aws_s3_bucket.kafka_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "kafka_logs" {
  bucket = aws_s3_bucket.kafka_logs.id

  rule {
    id     = "log_lifecycle"
    status = "Enabled"

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 8
}

# Kafka Topics (created via Terraform for initial setup)
resource "aws_msk_cluster_policy" "main" {
  cluster_arn = aws_msk_cluster.main.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowKafkaTopicActions"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:AlterCluster",
          "kafka-cluster:DescribeCluster"
        ]
        Resource = aws_msk_cluster.main.arn
      },
      {
        Sid    = "AllowTopicActions"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action = [
          "kafka-cluster:*Topic*",
          "kafka-cluster:WriteData",
          "kafka-cluster:ReadData"
        ]
        Resource = "${aws_msk_cluster.main.arn}/*"
      }
    ]
  })
}

# Lambda function for topic creation (optional automation)
resource "aws_lambda_function" "topic_creator" {
  count = var.enable_topic_automation ? 1 : 0
  
  filename         = data.archive_file.topic_creator[0].output_path
  function_name    = "${var.name_prefix}-kafka-topic-creator"
  role            = aws_iam_role.topic_creator[0].arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.topic_creator[0].output_base64sha256
  runtime         = "python3.9"
  timeout         = 60

  environment {
    variables = {
      BOOTSTRAP_SERVERS = aws_msk_cluster.main.bootstrap_brokers_sasl_iam
      TOPICS_CONFIG = jsonencode(var.default_topics)
    }
  }

  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = var.security_group_ids
  }

  tags = var.tags
}

# Lambda function code for topic creation
data "archive_file" "topic_creator" {
  count = var.enable_topic_automation ? 1 : 0
  
  type        = "zip"
  output_path = "/tmp/topic_creator.zip"
  
  source {
    content = templatefile("${path.module}/topic_creator.py.tpl", {
      topics = var.default_topics
    })
    filename = "index.py"
  }
}

# IAM role for Lambda
resource "aws_iam_role" "topic_creator" {
  count = var.enable_topic_automation ? 1 : 0
  
  name = "${var.name_prefix}-topic-creator-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM policy for Lambda
resource "aws_iam_role_policy" "topic_creator" {
  count = var.enable_topic_automation ? 1 : 0
  
  name = "${var.name_prefix}-topic-creator-policy"
  role = aws_iam_role.topic_creator[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "kafka-cluster:Connect",
          "kafka-cluster:AlterCluster",
          "kafka-cluster:DescribeCluster",
          "kafka-cluster:CreateTopic",
          "kafka-cluster:DescribeTopic",
          "kafka-cluster:WriteData",
          "kafka-cluster:ReadData"
        ]
        Resource = [
          aws_msk_cluster.main.arn,
          "${aws_msk_cluster.main.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Alarms for Kafka monitoring
resource "aws_cloudwatch_metric_alarm" "kafka_cpu" {
  alarm_name          = "${var.name_prefix}-kafka-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CpuUser"
  namespace           = "AWS/Kafka"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors kafka cpu utilization"
  alarm_actions       = [aws_sns_topic.kafka_alerts.arn]

  dimensions = {
    "Cluster Name" = aws_msk_cluster.main.cluster_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "kafka_disk" {
  alarm_name          = "${var.name_prefix}-kafka-high-disk"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "KafkaDataLogsDiskUsed"
  namespace           = "AWS/Kafka"
  period              = "300"
  statistic           = "Average"
  threshold           = "85"
  alarm_description   = "This metric monitors kafka disk utilization"
  alarm_actions       = [aws_sns_topic.kafka_alerts.arn]

  dimensions = {
    "Cluster Name" = aws_msk_cluster.main.cluster_name
  }

  tags = var.tags
}

# SNS Topic for alerts
resource "aws_sns_topic" "kafka_alerts" {
  name = "${var.name_prefix}-kafka-alerts"

  tags = var.tags
}