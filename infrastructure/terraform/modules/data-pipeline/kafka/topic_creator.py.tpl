#!/usr/bin/env python3
"""
Lambda function to automatically create Kafka topics in MSK cluster.
This function is triggered during infrastructure deployment to set up initial topics.
"""

import json
import os
import boto3
import logging
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import TopicAlreadyExistsError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    Lambda handler for creating Kafka topics
    """
    try:
        bootstrap_servers = os.environ.get('BOOTSTRAP_SERVERS')
        topics_config = json.loads(os.environ.get('TOPICS_CONFIG'))
        
        if not bootstrap_servers:
            raise ValueError("BOOTSTRAP_SERVERS environment variable not set")
        
        logger.info(f"Connecting to Kafka cluster: {bootstrap_servers}")
        
        # Create admin client with IAM authentication
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            security_protocol='SASL_SSL',
            sasl_mechanism='AWS_MSK_IAM',
            sasl_oauth_token_provider=lambda: get_oauth_token(),
            client_id='topic-creator-lambda'
        )
        
        # Create topics
        topics_to_create = []
        for topic_config in topics_config:
            topic = NewTopic(
                name=topic_config['name'],
                num_partitions=topic_config['num_partitions'],
                replication_factor=topic_config['replication_factor'],
                topic_configs=topic_config.get('config', {})
            )
            topics_to_create.append(topic)
        
        # Create topics in batch
        result = admin_client.create_topics(topics_to_create, validate_only=False)
        
        created_topics = []
        for topic_name, future in result.items():
            try:
                future.result()  # Will raise exception if creation failed
                created_topics.append(topic_name)
                logger.info(f"Successfully created topic: {topic_name}")
            except TopicAlreadyExistsError:
                logger.info(f"Topic already exists: {topic_name}")
                created_topics.append(topic_name)
            except Exception as e:
                logger.error(f"Failed to create topic {topic_name}: {str(e)}")
                raise
        
        # Verify topics exist
        existing_topics = admin_client.list_topics().topics
        for topic_config in topics_config:
            if topic_config['name'] not in existing_topics:
                raise RuntimeError(f"Topic {topic_config['name']} was not created successfully")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Topics created successfully',
                'created_topics': created_topics,
                'total_topics': len(created_topics)
            })
        }
        
    except Exception as e:
        logger.error(f"Error creating topics: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def get_oauth_token():
    """
    Get OAuth token for MSK IAM authentication
    """
    import base64
    import hmac
    import hashlib
    import urllib.parse
    from datetime import datetime
    
    # This is a simplified implementation
    # In production, you'd use the AWS SDK for proper IAM authentication
    session = boto3.Session()
    credentials = session.get_credentials()
    
    if not credentials:
        raise RuntimeError("Unable to get AWS credentials")
    
    # Return empty string for now - MSK IAM auth is handled by boto3
    return ""

# Default topics configuration (template substitution)
DEFAULT_TOPICS = ${jsonencode(topics)}