#!/usr/bin/env python3
"""
Data Ingestion Pipeline - GitHub Repository Analyzer

This module demonstrates Stage 1: Data Ingestion
- Collects code from GitHub repositories
- Extracts commit data, file changes, and issue reports
- Sends data to the configured data pipeline (Kafka/Kinesis)
"""

import os
import json
import logging
import asyncio
import aiohttp
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import git
from github import Github
from kafka import KafkaProducer
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeCommit:
    """Represents a code commit with metadata"""
    repository: str
    commit_hash: str
    author: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    additions: int
    deletions: int
    file_contents: Dict[str, str]
    bug_labels: List[str] = None  # For training data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class IssueReport:
    """Represents a bug/issue report"""
    repository: str
    issue_id: str
    title: str
    description: str
    labels: List[str]
    state: str
    created_at: datetime
    closed_at: Optional[datetime]
    linked_commits: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['closed_at'] = self.closed_at.isoformat() if self.closed_at else None
        return data

class DataPipelinePublisher:
    """Abstraction layer for different data pipeline backends"""
    
    def __init__(self, pipeline_type: str, config: Dict[str, Any]):
        self.pipeline_type = pipeline_type
        self.config = config
        self.producer = None
        
        if pipeline_type == "kafka":
            self._init_kafka()
        elif pipeline_type == "kinesis":
            self._init_kinesis()
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    
    def _init_kafka(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config['bootstrap_servers'],
                security_protocol='SASL_SSL',
                sasl_mechanism='AWS_MSK_IAM',
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def _init_kinesis(self):
        """Initialize Kinesis client"""
        try:
            self.producer = boto3.client(
                'kinesis',
                region_name=self.config.get('region', 'us-west-2')
            )
            logger.info("Kinesis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kinesis client: {e}")
            raise
    
    async def publish_commit(self, commit: CodeCommit):
        """Publish commit data to the pipeline"""
        try:
            if self.pipeline_type == "kafka":
                self.producer.send(
                    topic='code-commits',
                    key=commit.commit_hash,
                    value=commit.to_dict()
                )
            elif self.pipeline_type == "kinesis":
                self.producer.put_record(
                    StreamName=self.config['streams']['code_commits'],
                    Data=json.dumps(commit.to_dict()),
                    PartitionKey=commit.commit_hash
                )
            logger.info(f"Published commit: {commit.commit_hash}")
        except Exception as e:
            logger.error(f"Failed to publish commit {commit.commit_hash}: {e}")
            raise
    
    async def publish_issue(self, issue: IssueReport):
        """Publish issue data to the pipeline"""
        try:
            if self.pipeline_type == "kafka":
                self.producer.send(
                    topic='bug-reports',
                    key=issue.issue_id,
                    value=issue.to_dict()
                )
            elif self.pipeline_type == "kinesis":
                self.producer.put_record(
                    StreamName=self.config['streams']['bug_reports'],
                    Data=json.dumps(issue.to_dict()),
                    PartitionKey=issue.issue_id
                )
            logger.info(f"Published issue: {issue.issue_id}")
        except Exception as e:
            logger.error(f"Failed to publish issue {issue.issue_id}: {e}")
            raise

class GitHubCollector:
    """Collects data from GitHub repositories"""
    
    def __init__(self, github_token: str, publisher: DataPipelinePublisher):
        self.github = Github(github_token)
        self.publisher = publisher
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_repository_data(self, repo_name: str, days_back: int = 30) -> Dict[str, int]:
        """Collect comprehensive data from a repository"""
        logger.info(f"Starting data collection for repository: {repo_name}")
        
        try:
            repo = self.github.get_repo(repo_name)
            
            # Calculate date range
            since_date = datetime.now() - timedelta(days=days_back)
            
            # Collect commits
            commits_processed = await self._collect_commits(repo, since_date)
            
            # Collect issues
            issues_processed = await self._collect_issues(repo, since_date)
            
            logger.info(f"Collection completed for {repo_name}: "
                       f"{commits_processed} commits, {issues_processed} issues")
            
            return {
                'repository': repo_name,
                'commits_processed': commits_processed,
                'issues_processed': issues_processed,
                'collection_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect data from {repo_name}: {e}")
            raise
    
    async def _collect_commits(self, repo, since_date: datetime) -> int:
        """Collect commit data"""
        commits_processed = 0
        
        try:
            commits = repo.get_commits(since=since_date)
            
            for commit in commits:
                try:
                    # Get detailed commit information
                    commit_data = await self._extract_commit_data(repo, commit)
                    
                    # Publish to data pipeline
                    await self.publisher.publish_commit(commit_data)
                    
                    commits_processed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to process commit {commit.sha}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to collect commits: {e}")
            raise
        
        return commits_processed
    
    async def _extract_commit_data(self, repo, commit) -> CodeCommit:
        """Extract detailed data from a commit"""
        # Get file contents for analysis
        file_contents = {}
        files_changed = []
        
        try:
            # Limit to Python files for this example
            for file in commit.files:
                if file.filename.endswith('.py') and len(file_contents) < 10:  # Limit files
                    files_changed.append(file.filename)
                    
                    # Get file content if it's not too large
                    if file.additions + file.deletions < 1000:  # Skip very large files
                        try:
                            file_content = repo.get_contents(file.filename, ref=commit.sha)
                            if hasattr(file_content, 'decoded_content'):
                                content = file_content.decoded_content.decode('utf-8')
                                file_contents[file.filename] = content
                        except Exception:
                            # File might be deleted or binary
                            pass
        except Exception as e:
            logger.warning(f"Failed to extract file contents: {e}")
        
        # Determine if this is a bug-fix commit (for training labels)
        bug_keywords = ['fix', 'bug', 'issue', 'error', 'patch', 'resolve']
        is_bug_fix = any(keyword in commit.commit.message.lower() for keyword in bug_keywords)
        bug_labels = ['bug_fix'] if is_bug_fix else []
        
        return CodeCommit(
            repository=repo.full_name,
            commit_hash=commit.sha,
            author=commit.commit.author.name if commit.commit.author else "Unknown",
            timestamp=commit.commit.author.date,
            message=commit.commit.message,
            files_changed=files_changed,
            additions=sum(f.additions for f in commit.files if f.additions),
            deletions=sum(f.deletions for f in commit.files if f.deletions),
            file_contents=file_contents,
            bug_labels=bug_labels
        )
    
    async def _collect_issues(self, repo, since_date: datetime) -> int:
        """Collect issue/bug report data"""
        issues_processed = 0
        
        try:
            # Get issues that are labeled as bugs or closed recently
            issues = repo.get_issues(state='all', since=since_date)
            
            for issue in issues:
                try:
                    # Get linked commits (if issue mentions commits)
                    linked_commits = self._find_linked_commits(issue)
                    
                    issue_data = IssueReport(
                        repository=repo.full_name,
                        issue_id=str(issue.number),
                        title=issue.title,
                        description=issue.body or "",
                        labels=[label.name for label in issue.labels],
                        state=issue.state,
                        created_at=issue.created_at,
                        closed_at=issue.closed_at,
                        linked_commits=linked_commits
                    )
                    
                    await self.publisher.publish_issue(issue_data)
                    issues_processed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to process issue {issue.number}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to collect issues: {e}")
            raise
        
        return issues_processed
    
    def _find_linked_commits(self, issue) -> List[str]:
        """Find commits mentioned in issue description or comments"""
        linked_commits = []
        
        # Simple regex to find commit hashes in text
        import re
        commit_pattern = r'\b[a-f0-9]{7,40}\b'
        
        # Check issue body
        if issue.body:
            commits = re.findall(commit_pattern, issue.body.lower())
            linked_commits.extend(commits)
        
        # Check comments (limit to avoid rate limiting)
        try:
            comments = list(issue.get_comments()[:5])  # Limit to first 5 comments
            for comment in comments:
                commits = re.findall(commit_pattern, comment.body.lower())
                linked_commits.extend(commits)
        except Exception:
            pass
        
        return list(set(linked_commits))  # Remove duplicates

class DataIngestionPipeline:
    """Main orchestrator for data ingestion"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.publisher = self._init_publisher()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration from environment variables
        return {
            'github_token': os.getenv('GITHUB_TOKEN'),
            'pipeline_type': os.getenv('DATA_PIPELINE_TYPE', 'kinesis'),
            'kafka': {
                'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', '').split(',')
            },
            'kinesis': {
                'region': os.getenv('AWS_REGION', 'us-west-2'),
                'streams': {
                    'code_commits': os.getenv('KINESIS_STREAM_CODE_COMMITS', 'code-commits'),
                    'bug_reports': os.getenv('KINESIS_STREAM_BUG_REPORTS', 'bug-reports')
                }
            },
            'repositories': os.getenv('TARGET_REPOSITORIES', '').split(','),
            'collection_days': int(os.getenv('COLLECTION_DAYS', '30'))
        }
    
    def _init_publisher(self) -> DataPipelinePublisher:
        """Initialize data pipeline publisher"""
        pipeline_type = self.config['pipeline_type']
        
        if pipeline_type == 'kafka':
            config = self.config['kafka']
        elif pipeline_type == 'kinesis':
            config = self.config['kinesis']
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
        
        return DataPipelinePublisher(pipeline_type, config)
    
    async def run_collection(self, repositories: List[str] = None) -> Dict[str, Any]:
        """Run data collection for specified repositories"""
        repos_to_collect = repositories or self.config.get('repositories', [])
        
        if not repos_to_collect:
            raise ValueError("No repositories specified for collection")
        
        if not self.config.get('github_token'):
            raise ValueError("GitHub token not configured")
        
        collection_results = []
        
        async with GitHubCollector(self.config['github_token'], self.publisher) as collector:
            for repo_name in repos_to_collect:
                try:
                    result = await collector.collect_repository_data(
                        repo_name, 
                        self.config.get('collection_days', 30)
                    )
                    collection_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to collect from {repo_name}: {e}")
                    collection_results.append({
                        'repository': repo_name,
                        'error': str(e),
                        'collection_date': datetime.now().isoformat()
                    })
        
        return {
            'total_repositories': len(repos_to_collect),
            'successful_collections': len([r for r in collection_results if 'error' not in r]),
            'results': collection_results,
            'pipeline_type': self.config['pipeline_type']
        }

async def main():
    """Main entry point for data ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Data Ingestion Pipeline")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--repositories', nargs='+', help='Repositories to collect')
    parser.add_argument('--days', type=int, default=30, help='Days of history to collect')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without running')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(args.config)
    
    if args.dry_run:
        logger.info("Dry run mode - validating configuration")
        logger.info(f"Pipeline type: {pipeline.config['pipeline_type']}")
        logger.info(f"Repositories: {args.repositories or pipeline.config.get('repositories', [])}")
        return
    
    # Run collection
    try:
        results = await pipeline.run_collection(args.repositories)
        
        logger.info("Data collection completed!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        # Save results
        output_file = f"collection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())