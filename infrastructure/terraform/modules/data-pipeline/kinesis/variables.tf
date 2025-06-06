# Kinesis Data Pipeline Module Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "kinesis_config" {
  description = "Kinesis streams configuration"
  type = object({
    shard_count      = number
    retention_period = number
    stream_mode      = string
  })
  
  default = {
    shard_count      = 2
    retention_period = 24
    stream_mode      = "PROVISIONED"  # or "ON_DEMAND"
  }
  
  validation {
    condition     = contains(["PROVISIONED", "ON_DEMAND"], var.kinesis_config.stream_mode)
    error_message = "Stream mode must be either PROVISIONED or ON_DEMAND."
  }
  
  validation {
    condition     = var.kinesis_config.retention_period >= 24 && var.kinesis_config.retention_period <= 8760
    error_message = "Retention period must be between 24 hours and 8760 hours (1 year)."
  }
}

variable "enable_archival" {
  description = "Enable Kinesis Data Firehose for long-term archival"
  type        = bool
  default     = true
}

variable "enable_analytics" {
  description = "Enable Kinesis Data Analytics applications"
  type        = bool
  default     = false
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}