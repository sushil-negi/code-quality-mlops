# Kinesis Data Pipeline Module

# Kinesis Data Streams
resource "aws_kinesis_stream" "code_commits" {
  name             = "${var.name_prefix}-code-commits"
  shard_count      = var.kinesis_config.shard_count
  retention_period = var.kinesis_config.retention_period

  shard_level_metrics = [
    "IncomingRecords",
    "OutgoingRecords",
  ]

  stream_mode_details {
    stream_mode = var.kinesis_config.stream_mode
  }

  encryption_type = "KMS"
  kms_key_id      = aws_kms_key.kinesis.arn

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-code-commits"
    Type = "DataStream"
  })
}

resource "aws_kinesis_stream" "code_metrics" {
  name             = "${var.name_prefix}-code-metrics"
  shard_count      = var.kinesis_config.shard_count
  retention_period = var.kinesis_config.retention_period

  shard_level_metrics = [
    "IncomingRecords",
    "OutgoingRecords",
  ]

  stream_mode_details {
    stream_mode = var.kinesis_config.stream_mode
  }

  encryption_type = "KMS"
  kms_key_id      = aws_kms_key.kinesis.arn

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-code-metrics"
    Type = "DataStream"
  })
}

resource "aws_kinesis_stream" "bug_reports" {
  name             = "${var.name_prefix}-bug-reports"
  shard_count      = ceil(var.kinesis_config.shard_count / 2)  # Lower throughput expected
  retention_period = var.kinesis_config.retention_period * 3  # Longer retention for bugs

  shard_level_metrics = [
    "IncomingRecords",
    "OutgoingRecords",
  ]

  stream_mode_details {
    stream_mode = var.kinesis_config.stream_mode
  }

  encryption_type = "KMS"
  kms_key_id      = aws_kms_key.kinesis.arn

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-bug-reports"
    Type = "DataStream"
  })
}

resource "aws_kinesis_stream" "model_predictions" {
  name             = "${var.name_prefix}-model-predictions"
  shard_count      = var.kinesis_config.shard_count
  retention_period = var.kinesis_config.retention_period

  shard_level_metrics = [
    "IncomingRecords",
    "OutgoingRecords",
  ]

  stream_mode_details {
    stream_mode = var.kinesis_config.stream_mode
  }

  encryption_type = "KMS"
  kms_key_id      = aws_kms_key.kinesis.arn

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-model-predictions"
    Type = "DataStream"
  })
}

# KMS Key for Kinesis encryption
resource "aws_kms_key" "kinesis" {
  description             = "KMS key for Kinesis encryption"
  deletion_window_in_days = 7

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow Kinesis service"
        Effect = "Allow"
        Principal = {
          Service = "kinesis.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:Encrypt",
          "kms:GenerateDataKey*",
          "kms:ReEncrypt*"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-kinesis-kms"
  })
}

resource "aws_kms_alias" "kinesis" {
  name          = "alias/${var.name_prefix}-kinesis"
  target_key_id = aws_kms_key.kinesis.key_id
}

# Data source for account ID
data "aws_caller_identity" "current" {}

# Kinesis Data Firehose for archival (optional)
resource "aws_kinesis_firehose_delivery_stream" "archive" {
  count = var.enable_archival ? 1 : 0
  
  name        = "${var.name_prefix}-kinesis-archive"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose[0].arn
    bucket_arn = aws_s3_bucket.kinesis_archive[0].arn
    prefix     = "year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/"
    
    buffer_size     = 5
    buffer_interval = 60
    
    compression_format = "GZIP"
    
    # Data transformation
    processing_configuration {
      enabled = false
    }
    
    # CloudWatch logging
    cloudwatch_logging_options {
      enabled         = true
      log_group_name  = aws_cloudwatch_log_group.firehose[0].name
      log_stream_name = "S3Delivery"
    }
  }

  tags = var.tags
}

# S3 bucket for Kinesis archival
resource "aws_s3_bucket" "kinesis_archive" {
  count = var.enable_archival ? 1 : 0
  
  bucket = "${var.name_prefix}-kinesis-archive-${random_id.bucket_suffix.hex}"

  tags = var.tags
}

resource "aws_s3_bucket_versioning" "kinesis_archive" {
  count = var.enable_archival ? 1 : 0
  
  bucket = aws_s3_bucket.kinesis_archive[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "kinesis_archive" {
  count = var.enable_archival ? 1 : 0
  
  bucket = aws_s3_bucket.kinesis_archive[0].id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.kinesis.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "kinesis_archive" {
  count = var.enable_archival ? 1 : 0
  
  bucket = aws_s3_bucket.kinesis_archive[0].id

  rule {
    id     = "archive_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 2555  # 7 years
    }
  }
}

# IAM Role for Firehose
resource "aws_iam_role" "firehose" {
  count = var.enable_archival ? 1 : 0
  
  name = "${var.name_prefix}-firehose-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "firehose.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "firehose" {
  count = var.enable_archival ? 1 : 0
  
  name = "${var.name_prefix}-firehose-policy"
  role = aws_iam_role.firehose[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:PutObject"
        ]
        Resource = [
          aws_s3_bucket.kinesis_archive[0].arn,
          "${aws_s3_bucket.kinesis_archive[0].arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.kinesis.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:PutLogEvents"
        ]
        Resource = var.enable_archival ? aws_cloudwatch_log_group.firehose[0].arn : ""
      }
    ]
  })
}

# CloudWatch Log Group for Firehose
resource "aws_cloudwatch_log_group" "firehose" {
  count = var.enable_archival ? 1 : 0
  
  name              = "/aws/kinesisfirehose/${var.name_prefix}-archive"
  retention_in_days = 14

  tags = var.tags
}

# CloudWatch Alarms for Kinesis monitoring
resource "aws_cloudwatch_metric_alarm" "incoming_records_high" {
  for_each = local.streams

  alarm_name          = "${var.name_prefix}-${each.key}-high-incoming-records"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "IncomingRecords"
  namespace           = "AWS/Kinesis"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10000"
  alarm_description   = "This metric monitors incoming records to ${each.key} stream"
  treat_missing_data  = "notBreaching"

  dimensions = {
    StreamName = each.value.name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "iterator_age_high" {
  for_each = local.streams

  alarm_name          = "${var.name_prefix}-${each.key}-high-iterator-age"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "GetRecords.IteratorAgeMilliseconds"
  namespace           = "AWS/Kinesis"
  period              = "300"
  statistic           = "Maximum"
  threshold           = "60000"  # 1 minute
  alarm_description   = "This metric monitors iterator age for ${each.key} stream"
  treat_missing_data  = "notBreaching"

  dimensions = {
    StreamName = each.value.name
  }

  tags = var.tags
}

# Local values for stream references
locals {
  streams = {
    code_commits      = aws_kinesis_stream.code_commits
    code_metrics      = aws_kinesis_stream.code_metrics
    bug_reports       = aws_kinesis_stream.bug_reports
    model_predictions = aws_kinesis_stream.model_predictions
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 8
}