# Networking Module Outputs

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "vpc_arn" {
  description = "ARN of the VPC"
  value       = aws_vpc.main.arn
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_cidrs" {
  description = "CIDR blocks of the public subnets"
  value       = aws_subnet.public[*].cidr_block
}

output "private_subnet_cidrs" {
  description = "CIDR blocks of the private subnets"
  value       = aws_subnet.private[*].cidr_block
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].id
}

output "nat_gateway_ips" {
  description = "Elastic IP addresses of the NAT Gateways"
  value       = aws_eip.nat[*].public_ip
}

output "database_subnet_group_name" {
  description = "Name of the database subnet group"
  value       = aws_db_subnet_group.main.name
}

output "cache_subnet_group_name" {
  description = "Name of the cache subnet group"
  value       = aws_elasticache_subnet_group.main.name
}

output "public_route_table_id" {
  description = "ID of the public route table"
  value       = aws_route_table.public.id
}

output "private_route_table_ids" {
  description = "IDs of the private route tables"
  value       = aws_route_table.private[*].id
}

output "vpc_endpoints" {
  description = "VPC endpoint information"
  value = {
    s3 = {
      id           = aws_vpc_endpoint.s3.id
      service_name = aws_vpc_endpoint.s3.service_name
    }
    ecr_api = {
      id           = aws_vpc_endpoint.ecr_api.id
      service_name = aws_vpc_endpoint.ecr_api.service_name
      dns_entry    = aws_vpc_endpoint.ecr_api.dns_entry
    }
    ecr_dkr = {
      id           = aws_vpc_endpoint.ecr_dkr.id
      service_name = aws_vpc_endpoint.ecr_dkr.service_name
      dns_entry    = aws_vpc_endpoint.ecr_dkr.dns_entry
    }
  }
}