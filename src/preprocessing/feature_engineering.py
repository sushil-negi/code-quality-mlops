#!/usr/bin/env python3
"""
Data Processing Pipeline - Feature Engineering

This module demonstrates Stage 2: Data Processing
- Processes raw commit and issue data from the data pipeline
- Extracts meaningful features for ML models
- Handles code analysis and complexity metrics
- Publishes processed features to the feature store
"""

import os
import json
import logging
import asyncio
import ast
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from kafka import KafkaConsumer
import boto3

# Code analysis imports
import radon.complexity as radon_complexity
import radon.metrics as radon_metrics
from radon.visitors import ComplexityVisitor
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeFeatures:
    """Represents extracted features from code"""
    commit_hash: str
    repository: str
    timestamp: datetime
    
    # Basic metrics
    lines_of_code: int
    files_changed: int
    additions: int
    deletions: int
    
    # Complexity metrics
    cyclomatic_complexity: float
    cognitive_complexity: float
    halstead_volume: float
    halstead_difficulty: float
    
    # Code quality indicators
    function_count: int
    class_count: int
    comment_ratio: float
    docstring_ratio: float
    
    # Pattern-based features
    test_file_ratio: float
    import_complexity: float
    nested_depth: int
    
    # Text-based features
    commit_message_sentiment: float
    commit_message_length: int
    code_readability: float
    
    # Bug indicators
    error_handling_ratio: float
    todo_comment_count: int
    magic_number_count: int
    
    # Historical features
    author_experience: int
    file_change_frequency: float
    
    # Labels (for training)
    is_bug_fix: bool = False
    has_linked_issues: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return data

class CodeAnalyzer:
    """Analyzes code to extract features"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def analyze_code_content(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """Extract features from code content"""
        if not file_contents:
            return self._empty_code_features()
        
        # Combine all code content
        all_code = "\n".join(file_contents.values())
        
        # Basic metrics
        total_lines = len(all_code.split('\n'))
        non_empty_lines = len([line for line in all_code.split('\n') if line.strip()])
        
        features = {
            'lines_of_code': non_empty_lines,
            'total_characters': len(all_code)
        }
        
        # Analyze each file
        complexity_scores = []
        function_counts = []
        class_counts = []
        comment_ratios = []
        docstring_ratios = []
        
        for filename, content in file_contents.items():
            file_features = self._analyze_single_file(content)
            
            complexity_scores.append(file_features.get('cyclomatic_complexity', 0))
            function_counts.append(file_features.get('function_count', 0))
            class_counts.append(file_features.get('class_count', 0))
            comment_ratios.append(file_features.get('comment_ratio', 0))
            docstring_ratios.append(file_features.get('docstring_ratio', 0))
        
        # Aggregate features
        features.update({
            'cyclomatic_complexity': np.mean(complexity_scores) if complexity_scores else 0,
            'cognitive_complexity': self._calculate_cognitive_complexity(all_code),
            'halstead_volume': self._calculate_halstead_metrics(all_code).get('volume', 0),
            'halstead_difficulty': self._calculate_halstead_metrics(all_code).get('difficulty', 0),
            'function_count': sum(function_counts),
            'class_count': sum(class_counts),
            'comment_ratio': np.mean(comment_ratios) if comment_ratios else 0,
            'docstring_ratio': np.mean(docstring_ratios) if docstring_ratios else 0,
            'test_file_ratio': self._calculate_test_file_ratio(file_contents),
            'import_complexity': self._calculate_import_complexity(all_code),
            'nested_depth': self._calculate_nested_depth(all_code),
            'code_readability': self._calculate_code_readability(all_code),
            'error_handling_ratio': self._calculate_error_handling_ratio(all_code),
            'todo_comment_count': self._count_todo_comments(all_code),
            'magic_number_count': self._count_magic_numbers(all_code)
        })
        
        return features
    
    def _analyze_single_file(self, content: str) -> Dict[str, Any]:
        """Analyze a single Python file"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._empty_file_features()
        
        # Count functions and classes
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        # Calculate complexity using radon
        try:
            complexity_visitor = ComplexityVisitor.from_code(content)
            avg_complexity = complexity_visitor.complexity if complexity_visitor.complexity else 0
        except:
            avg_complexity = 0
        
        # Comment and docstring analysis
        lines = content.split('\n')
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # Count docstrings
        docstring_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (ast.get_docstring(node) is not None):
                    docstring_count += 1
        
        return {
            'function_count': function_count,
            'class_count': class_count,
            'cyclomatic_complexity': avg_complexity,
            'comment_ratio': comment_lines / len(lines) if lines else 0,
            'docstring_ratio': docstring_count / max(function_count + class_count, 1)
        }
    
    def _calculate_cognitive_complexity(self, code: str) -> float:
        """Calculate cognitive complexity (simplified)"""
        # Simplified cognitive complexity based on nesting and control flow
        complexity = 0
        nesting_level = 0
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Increase nesting for certain constructs
            if any(stripped.startswith(keyword) for keyword in ['if', 'for', 'while', 'try', 'with']):
                nesting_level += 1
                complexity += nesting_level
            
            # Decrease nesting
            if stripped in ['else:', 'elif', 'except:', 'finally:', 'end']:
                nesting_level = max(0, nesting_level - 1)
        
        return complexity / max(len(code.split('\n')), 1)
    
    def _calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        """Calculate Halstead complexity metrics"""
        try:
            # Use radon for Halstead metrics
            halstead = radon_metrics.h_visit(code)
            return {
                'volume': halstead.volume if halstead else 0,
                'difficulty': halstead.difficulty if halstead else 0,
                'effort': halstead.effort if halstead else 0
            }
        except:
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
    
    def _calculate_test_file_ratio(self, file_contents: Dict[str, str]) -> float:
        """Calculate ratio of test files"""
        if not file_contents:
            return 0
        
        test_files = sum(1 for filename in file_contents.keys() 
                        if 'test' in filename.lower() or filename.startswith('test_'))
        
        return test_files / len(file_contents)
    
    def _calculate_import_complexity(self, code: str) -> float:
        """Calculate import complexity based on number and type of imports"""
        try:
            tree = ast.parse(code)
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            # Weight different types of imports
            complexity = 0
            for imp in imports:
                if isinstance(imp, ast.ImportFrom):
                    complexity += 2  # from imports are more complex
                else:
                    complexity += 1
            
            return complexity / max(len(code.split('\n')), 1)
        except:
            return 0
    
    def _calculate_nested_depth(self, code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Count indentation
            if stripped:
                indent = len(line) - len(line.lstrip())
                depth = indent // 4  # Assuming 4-space indentation
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_code_readability(self, code: str) -> float:
        """Calculate code readability score (simplified)"""
        try:
            # Remove code-specific syntax and calculate readability on comments/strings
            text_content = ""
            
            # Extract comments and docstrings
            for line in code.split('\n'):
                if '#' in line:
                    comment = line[line.index('#')+1:].strip()
                    text_content += comment + " "
            
            if not text_content:
                return 0.5  # Neutral score if no text content
            
            return min(flesch_reading_ease(text_content) / 100, 1.0)
        except:
            return 0.5
    
    def _calculate_error_handling_ratio(self, code: str) -> float:
        """Calculate ratio of error handling constructs"""
        try:
            tree = ast.parse(code)
            
            # Count try-except blocks
            try_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
            
            # Count total functions (as proxy for places where error handling could be)
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            
            return try_blocks / max(functions, 1)
        except:
            return 0
    
    def _count_todo_comments(self, code: str) -> int:
        """Count TODO/FIXME comments"""
        todo_pattern = r'#.*\b(TODO|FIXME|HACK|XXX)\b'
        return len(re.findall(todo_pattern, code, re.IGNORECASE))
    
    def _count_magic_numbers(self, code: str) -> int:
        """Count magic numbers in code"""
        try:
            tree = ast.parse(code)
            magic_numbers = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    # Exclude common non-magic numbers
                    if node.value not in [0, 1, -1, 2, 10, 100, 1000]:
                        magic_numbers += 1
            
            return magic_numbers
        except:
            return 0
    
    def _empty_code_features(self) -> Dict[str, Any]:
        """Return empty feature set"""
        return {
            'lines_of_code': 0,
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'halstead_volume': 0,
            'halstead_difficulty': 0,
            'function_count': 0,
            'class_count': 0,
            'comment_ratio': 0,
            'docstring_ratio': 0,
            'test_file_ratio': 0,
            'import_complexity': 0,
            'nested_depth': 0,
            'code_readability': 0,
            'error_handling_ratio': 0,
            'todo_comment_count': 0,
            'magic_number_count': 0
        }
    
    def _empty_file_features(self) -> Dict[str, Any]:
        """Return empty file feature set"""
        return {
            'function_count': 0,
            'class_count': 0,
            'cyclomatic_complexity': 0,
            'comment_ratio': 0,
            'docstring_ratio': 0
        }

class HistoricalAnalyzer:
    """Analyzes historical patterns and trends"""
    
    def __init__(self):
        self.author_stats = {}
        self.file_change_history = {}
    
    def update_author_stats(self, author: str, commit_hash: str):
        """Update author experience tracking"""
        if author not in self.author_stats:
            self.author_stats[author] = {
                'commit_count': 0,
                'first_commit': None,
                'recent_commits': []
            }
        
        self.author_stats[author]['commit_count'] += 1
        self.author_stats[author]['recent_commits'].append(commit_hash)
        
        # Keep only recent commits (last 100)
        if len(self.author_stats[author]['recent_commits']) > 100:
            self.author_stats[author]['recent_commits'] = \
                self.author_stats[author]['recent_commits'][-100:]
    
    def get_author_experience(self, author: str) -> int:
        """Get author experience score"""
        return self.author_stats.get(author, {}).get('commit_count', 0)
    
    def update_file_change_frequency(self, files: List[str], commit_hash: str):
        """Update file change frequency tracking"""
        for file in files:
            if file not in self.file_change_history:
                self.file_change_history[file] = []
            
            self.file_change_history[file].append(commit_hash)
            
            # Keep only recent changes (last 50)
            if len(self.file_change_history[file]) > 50:
                self.file_change_history[file] = \
                    self.file_change_history[file][-50:]
    
    def get_file_change_frequency(self, files: List[str]) -> float:
        """Calculate average file change frequency"""
        if not files:
            return 0
        
        frequencies = []
        for file in files:
            freq = len(self.file_change_history.get(file, []))
            frequencies.append(freq)
        
        return np.mean(frequencies) if frequencies else 0

class CommitMessageAnalyzer:
    """Analyzes commit messages for sentiment and patterns"""
    
    def __init__(self):
        # Simple sentiment keywords
        self.positive_words = [
            'fix', 'improve', 'add', 'enhance', 'optimize', 'update', 
            'refactor', 'clean', 'better', 'faster'
        ]
        self.negative_words = [
            'bug', 'error', 'fail', 'break', 'issue', 'problem', 
            'wrong', 'bad', 'slow', 'crash'
        ]
    
    def analyze_sentiment(self, message: str) -> float:
        """Calculate sentiment score for commit message"""
        if not message:
            return 0.5
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in message_lower)
        negative_count = sum(1 for word in self.negative_words if word in message_lower)
        
        total_words = len(message.split())
        
        if total_words == 0:
            return 0.5
        
        # Normalize sentiment score
        sentiment = (positive_count - negative_count) / total_words
        return max(0, min(1, 0.5 + sentiment * 2))  # Scale to [0, 1]

class FeatureEngineeringPipeline:
    """Main feature engineering pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.code_analyzer = CodeAnalyzer()
        self.historical_analyzer = HistoricalAnalyzer()
        self.message_analyzer = CommitMessageAnalyzer()
        self.consumer = None
        self.processed_count = 0
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration from environment variables
        return {
            'pipeline_type': os.getenv('DATA_PIPELINE_TYPE', 'kinesis'),
            'kafka': {
                'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', '').split(','),
                'topics': ['code-commits', 'bug-reports']
            },
            'kinesis': {
                'region': os.getenv('AWS_REGION', 'us-west-2'),
                'streams': {
                    'code_commits': os.getenv('KINESIS_STREAM_CODE_COMMITS', 'code-commits'),
                    'bug_reports': os.getenv('KINESIS_STREAM_BUG_REPORTS', 'bug-reports')
                }
            },
            'output': {
                'pipeline_type': os.getenv('OUTPUT_PIPELINE_TYPE', 'kinesis'),
                'stream_name': os.getenv('PROCESSED_FEATURES_STREAM', 'code-metrics')
            }
        }
    
    async def start_processing(self):
        """Start the feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline")
        
        if self.config['pipeline_type'] == 'kafka':
            await self._process_kafka_messages()
        elif self.config['pipeline_type'] == 'kinesis':
            await self._process_kinesis_streams()
        else:
            raise ValueError(f"Unsupported pipeline type: {self.config['pipeline_type']}")
    
    async def _process_kafka_messages(self):
        """Process messages from Kafka"""
        from kafka import KafkaConsumer
        
        consumer = KafkaConsumer(
            'code-commits',
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            security_protocol='SASL_SSL',
            sasl_mechanism='AWS_MSK_IAM',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='feature-engineering-group'
        )
        
        logger.info("Connected to Kafka, starting message processing")
        
        for message in consumer:
            try:
                commit_data = message.value
                features = await self._process_commit_data(commit_data)
                await self._publish_features(features)
                
                self.processed_count += 1
                if self.processed_count % 10 == 0:
                    logger.info(f"Processed {self.processed_count} commits")
                    
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                continue
    
    async def _process_kinesis_streams(self):
        """Process messages from Kinesis"""
        kinesis_client = boto3.client('kinesis', region_name=self.config['kinesis']['region'])
        
        # Get stream information
        stream_name = self.config['kinesis']['streams']['code_commits']
        
        try:
            response = kinesis_client.describe_stream(StreamName=stream_name)
            shards = response['StreamDescription']['Shards']
            
            logger.info(f"Processing Kinesis stream: {stream_name} with {len(shards)} shards")
            
            # Process each shard
            for shard in shards:
                await self._process_kinesis_shard(kinesis_client, stream_name, shard['ShardId'])
                
        except Exception as e:
            logger.error(f"Failed to process Kinesis stream: {e}")
            raise
    
    async def _process_kinesis_shard(self, client, stream_name: str, shard_id: str):
        """Process a single Kinesis shard"""
        try:
            # Get shard iterator
            iterator_response = client.get_shard_iterator(
                StreamName=stream_name,
                ShardId=shard_id,
                ShardIteratorType='LATEST'
            )
            
            shard_iterator = iterator_response['ShardIterator']
            
            while shard_iterator:
                # Get records
                response = client.get_records(ShardIterator=shard_iterator, Limit=10)
                
                records = response['Records']
                shard_iterator = response.get('NextShardIterator')
                
                for record in records:
                    try:
                        commit_data = json.loads(record['Data'])
                        features = await self._process_commit_data(commit_data)
                        await self._publish_features(features)
                        
                        self.processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process record: {e}")
                        continue
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to process shard {shard_id}: {e}")
    
    async def _process_commit_data(self, commit_data: Dict[str, Any]) -> CodeFeatures:
        """Process a single commit to extract features"""
        try:
            # Extract basic information
            commit_hash = commit_data['commit_hash']
            repository = commit_data['repository']
            timestamp = datetime.fromisoformat(commit_data['timestamp'])
            author = commit_data['author']
            message = commit_data['message']
            files_changed = commit_data['files_changed']
            file_contents = commit_data.get('file_contents', {})
            
            # Update historical analyzers
            self.historical_analyzer.update_author_stats(author, commit_hash)
            self.historical_analyzer.update_file_change_frequency(files_changed, commit_hash)
            
            # Analyze code content
            code_features = self.code_analyzer.analyze_code_content(file_contents)
            
            # Analyze commit message
            message_sentiment = self.message_analyzer.analyze_sentiment(message)
            
            # Get historical features
            author_experience = self.historical_analyzer.get_author_experience(author)
            file_change_freq = self.historical_analyzer.get_file_change_frequency(files_changed)
            
            # Determine labels
            bug_labels = commit_data.get('bug_labels', [])
            is_bug_fix = 'bug_fix' in bug_labels
            
            # Create feature object
            features = CodeFeatures(
                commit_hash=commit_hash,
                repository=repository,
                timestamp=timestamp,
                lines_of_code=code_features.get('lines_of_code', 0),
                files_changed=len(files_changed),
                additions=commit_data.get('additions', 0),
                deletions=commit_data.get('deletions', 0),
                cyclomatic_complexity=code_features.get('cyclomatic_complexity', 0),
                cognitive_complexity=code_features.get('cognitive_complexity', 0),
                halstead_volume=code_features.get('halstead_volume', 0),
                halstead_difficulty=code_features.get('halstead_difficulty', 0),
                function_count=code_features.get('function_count', 0),
                class_count=code_features.get('class_count', 0),
                comment_ratio=code_features.get('comment_ratio', 0),
                docstring_ratio=code_features.get('docstring_ratio', 0),
                test_file_ratio=code_features.get('test_file_ratio', 0),
                import_complexity=code_features.get('import_complexity', 0),
                nested_depth=code_features.get('nested_depth', 0),
                commit_message_sentiment=message_sentiment,
                commit_message_length=len(message),
                code_readability=code_features.get('code_readability', 0),
                error_handling_ratio=code_features.get('error_handling_ratio', 0),
                todo_comment_count=code_features.get('todo_comment_count', 0),
                magic_number_count=code_features.get('magic_number_count', 0),
                author_experience=author_experience,
                file_change_frequency=file_change_freq,
                is_bug_fix=is_bug_fix,
                has_linked_issues=len(commit_data.get('linked_issues', [])) > 0
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to process commit data: {e}")
            raise
    
    async def _publish_features(self, features: CodeFeatures):
        """Publish processed features to output pipeline"""
        try:
            output_config = self.config['output']
            
            if output_config['pipeline_type'] == 'kinesis':
                kinesis_client = boto3.client('kinesis', region_name=self.config['kinesis']['region'])
                
                kinesis_client.put_record(
                    StreamName=output_config['stream_name'],
                    Data=json.dumps(features.to_dict()),
                    PartitionKey=features.commit_hash
                )
            
            # Also save to local file for debugging
            await self._save_features_locally(features)
            
        except Exception as e:
            logger.error(f"Failed to publish features: {e}")
            raise
    
    async def _save_features_locally(self, features: CodeFeatures):
        """Save features to local file for debugging"""
        output_dir = Path("processed_features")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"features_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(features.to_dict()) + '\n')

async def main():
    """Main entry point for feature engineering pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(args.config)
    
    if args.dry_run:
        logger.info("Dry run mode - validating configuration")
        logger.info(f"Pipeline type: {pipeline.config['pipeline_type']}")
        logger.info(f"Output type: {pipeline.config['output']['pipeline_type']}")
        return
    
    # Start processing
    try:
        await pipeline.start_processing()
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
        logger.info(f"Total commits processed: {pipeline.processed_count}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())