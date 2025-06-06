# Kinesis Data Pipeline Module Outputs

output "stream_names" {
  description = "Map of stream names"
  value = {
    code_commits      = aws_kinesis_stream.code_commits.name
    code_metrics      = aws_kinesis_stream.code_metrics.name
    bug_reports       = aws_kinesis_stream.bug_reports.name
    model_predictions = aws_kinesis_stream.model_predictions.name
  }
}

output "stream_arns" {
  description = "Map of stream ARNs"
  value = {
    code_commits      = aws_kinesis_stream.code_commits.arn
    code_metrics      = aws_kinesis_stream.code_metrics.arn
    bug_reports       = aws_kinesis_stream.bug_reports.arn
    model_predictions = aws_kinesis_stream.model_predictions.arn
  }
}

output "kms_key_id" {
  description = "KMS key ID used for encryption"
  value       = aws_kms_key.kinesis.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN used for encryption"
  value       = aws_kms_key.kinesis.arn
}

output "firehose_delivery_stream_name" {
  description = "Firehose delivery stream name for archival"
  value       = var.enable_archival ? aws_kinesis_firehose_delivery_stream.archive[0].name : null
}

output "archive_bucket_name" {
  description = "S3 bucket name for archival"
  value       = var.enable_archival ? aws_s3_bucket.kinesis_archive[0].bucket : null
}

output "connection_info" {
  description = "Connection information for applications"
  value = {
    region = data.aws_region.current.name
    streams = {
      code_commits      = aws_kinesis_stream.code_commits.name
      code_metrics      = aws_kinesis_stream.code_metrics.name
      bug_reports       = aws_kinesis_stream.bug_reports.name
      model_predictions = aws_kinesis_stream.model_predictions.name
    }
    
    # Environment variables for applications
    env_vars = {
      KINESIS_REGION = data.aws_region.current.name
      KINESIS_STREAM_CODE_COMMITS = aws_kinesis_stream.code_commits.name
      KINESIS_STREAM_CODE_METRICS = aws_kinesis_stream.code_metrics.name
      KINESIS_STREAM_BUG_REPORTS = aws_kinesis_stream.bug_reports.name
      KINESIS_STREAM_MODEL_PREDICTIONS = aws_kinesis_stream.model_predictions.name
    }
  }
}

output "stream_configuration" {
  description = "Stream configuration details"
  value = {
    shard_count      = var.kinesis_config.shard_count
    retention_period = var.kinesis_config.retention_period
    stream_mode      = var.kinesis_config.stream_mode
    encryption_type  = "KMS"
    
    streams_info = {
      code_commits = {
        name         = aws_kinesis_stream.code_commits.name
        shard_count  = aws_kinesis_stream.code_commits.shard_count
        retention    = aws_kinesis_stream.code_commits.retention_period
      }
      code_metrics = {
        name         = aws_kinesis_stream.code_metrics.name
        shard_count  = aws_kinesis_stream.code_metrics.shard_count
        retention    = aws_kinesis_stream.code_metrics.retention_period
      }
      bug_reports = {
        name         = aws_kinesis_stream.bug_reports.name
        shard_count  = aws_kinesis_stream.bug_reports.shard_count
        retention    = aws_kinesis_stream.bug_reports.retention_period
      }
      model_predictions = {
        name         = aws_kinesis_stream.model_predictions.name
        shard_count  = aws_kinesis_stream.model_predictions.shard_count
        retention    = aws_kinesis_stream.model_predictions.retention_period
      }
    }
  }
}

# Data source for current region
data "aws_region" "current" {}