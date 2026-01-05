#!/usr/bin/env python3
"""
Dataset Registry CLI - Task 2.6

Command-line interface for dataset registry operations.

Commands:
- list: List datasets/versions
- search: Search with filters
- show: Show dataset/version details
- lineage: Show lineage tree
- verify: Verify file integrity
- stats: Calculate statistics
- init-schema: Initialize database schema

Integrates:
- Task 2.1: Postgres Schema
- Task 2.2: Semantic Versioning
- Task 2.3: SHA256 Integrity
- Task 2.4: Lineage Tracking
- Task 2.5: Search/Filter Queries

Phase A1 Week 3-4: Task 2.6 COMPLETE
"""

import argparse
import sys
import json
from typing import Optional
from pathlib import Path
from loguru import logger

from training.registry.dataset_registry_db import (
    DatasetRegistryDB,
    get_db_connection
)
from training.registry.semantic_version import (
    SemanticVersion,
    VersionManager
)
from training.registry.file_integrity import (
    FileIntegrityChecker,
    DatasetIntegrityManager
)
from training.registry.lineage_tracker import (
    LineageTracker,
    TransformationType
)
from training.registry.dataset_search import (
    DatasetSearchEngine,
    SearchFilter,
    SortField,
    SortOrder
)


class RegistryCLI:
    """
    CLI for dataset registry operations
    """
    
    def __init__(self, db_config: dict):
        """
        Initialize CLI
        
        Args:
            db_config: Database configuration
        """
        self.db_config = db_config
        self.db = None
        logger.info("RegistryCLI initialized")
    
    def connect(self):
        """Connect to database"""
        if not self.db:
            self.db = DatasetRegistryDB(self.db_config)
            self.db.connect()
            logger.info("Connected to database")
    
    def disconnect(self):
        """Disconnect from database"""
        if self.db:
            self.db.disconnect()
            logger.info("Disconnected from database")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    # ========================================================================
    # Schema Commands
    # ========================================================================
    
    def init_schema(self, schema_file: Optional[str] = None):
        """
        Initialize database schema
        
        Args:
            schema_file: Path to schema SQL file
        """
        logger.info("Initializing database schema")
        
        self.connect()
        self.db.init_schema(schema_file)
        
        print("✓ Database schema initialized successfully")
    
    def check_schema(self):
        """Check schema version"""
        self.connect()
        
        version = self.db.check_schema_version()
        
        if version:
            print(f"Schema version: {version}")
        else:
            print("Schema not initialized")
    
    # ========================================================================
    # List Commands
    # ========================================================================
    
    def list_datasets(
        self,
        agent_type: Optional[str] = None,
        format: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        List datasets
        
        Args:
            agent_type: Filter by agent type
            format: Filter by format
            status: Filter by status
            limit: Maximum number of results
        """
        self.connect()
        
        logger.info(f"Listing datasets (agent_type={agent_type}, format={format})")
        
        # Build search filter
        search_filter = SearchFilter(
            agent_type=agent_type,
            format=format,
            status=status,
            limit=limit,
            sort_by=SortField.UPDATED_AT,
            sort_order=SortOrder.DESC
        )
        
        # Search
        engine = DatasetSearchEngine(self.db)
        result = engine.search_datasets(search_filter)
        
        if not result.datasets:
            print("No datasets found")
            return
        
        print(f"\nFound {result.total_count} dataset(s):\n")
        
        for dataset in result.datasets:
            print(f"  {dataset['agent_type']:15} {dataset['name']}")
            print(f"    ID: {dataset['id']}")
            print(f"    Format: {dataset['format']}, Type: {dataset['dataset_type']}")
            print(f"    Status: {dataset['status']}")
            print(f"    Current version: {dataset['current_version'] or 'None'}")
            print(f"    Total examples: {dataset['total_examples']}")
            print(f"    Updated: {dataset['updated_at']}")
            print()
    
    def list_versions(
        self,
        dataset_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        is_latest: Optional[bool] = None,
        min_quality: Optional[float] = None,
        limit: Optional[int] = None
    ):
        """
        List dataset versions
        
        Args:
            dataset_id: Filter by dataset ID
            agent_type: Filter by agent type
            is_latest: Filter for latest versions only
            min_quality: Minimum quality score
            limit: Maximum number of results
        """
        self.connect()
        
        logger.info(f"Listing versions (dataset_id={dataset_id}, agent_type={agent_type})")
        
        if dataset_id:
            # List versions for specific dataset
            versions = self.db.list_versions(dataset_id, limit=limit)
            
            if not versions:
                print("No versions found")
                return
            
            print(f"\nFound {len(versions)} version(s):\n")
            
            for version in versions:
                print(f"  {version.version:10} ({version.example_count} examples)")
                print(f"    ID: {version.id}")
                print(f"    File: {version.file_path}")
                print(f"    Size: {version.file_size_bytes} bytes")
                print(f"    Quality: {version.avg_quality_score or 'N/A'}")
                print(f"    Latest: {version.is_latest}")
                print(f"    Created: {version.created_at}")
                print()
        
        else:
            # Search all versions
            search_filter = SearchFilter(
                agent_type=agent_type,
                is_latest=is_latest,
                min_quality_score=min_quality,
                limit=limit,
                sort_by=SortField.CREATED_AT,
                sort_order=SortOrder.DESC
            )
            
            engine = DatasetSearchEngine(self.db)
            result = engine.search_versions(search_filter)
            
            if not result.versions:
                print("No versions found")
                return
            
            print(f"\nFound {result.total_count} version(s):\n")
            
            for version in result.versions:
                print(f"  {version['dataset_name']:20} v{version['version']}")
                print(f"    Agent: {version['agent_type']}")
                print(f"    Examples: {version['example_count']}")
                print(f"    Quality: {version.get('avg_quality_score', 'N/A')}")
                print(f"    Created: {version['created_at']}")
                print()
    
    # ========================================================================
    # Search Commands
    # ========================================================================
    
    def search_datasets(
        self,
        query: str,
        agent_type: Optional[str] = None,
        format: Optional[str] = None,
        limit: int = 10
    ):
        """
        Search datasets by text query
        
        Args:
            query: Search query
            agent_type: Filter by agent type
            format: Filter by format
            limit: Maximum number of results
        """
        self.connect()
        
        logger.info(f"Searching datasets: {query}")
        
        search_filter = SearchFilter(
            query=query,
            agent_type=agent_type,
            format=format,
            limit=limit,
            sort_by=SortField.UPDATED_AT,
            sort_order=SortOrder.DESC
        )
        
        engine = DatasetSearchEngine(self.db)
        result = engine.search_datasets(search_filter)
        
        if not result.datasets:
            print(f"No datasets found matching '{query}'")
            return
        
        print(f"\nFound {result.total_count} dataset(s) matching '{query}':\n")
        
        for dataset in result.datasets:
            print(f"  {dataset['name']}")
            print(f"    Agent: {dataset['agent_type']}, Format: {dataset['format']}")
            print(f"    Version: {dataset['current_version'] or 'None'}")
            print(f"    Examples: {dataset['total_examples']}")
            print()
    
    def search_versions(
        self,
        agent_type: Optional[str] = None,
        min_quality: Optional[float] = None,
        min_examples: Optional[int] = None,
        is_latest: bool = False,
        limit: int = 10
    ):
        """
        Search versions with filters
        
        Args:
            agent_type: Filter by agent type
            min_quality: Minimum quality score
            min_examples: Minimum example count
            is_latest: Filter for latest versions only
            limit: Maximum number of results
        """
        self.connect()
        
        logger.info(f"Searching versions with filters")
        
        search_filter = SearchFilter(
            agent_type=agent_type,
            min_quality_score=min_quality,
            min_example_count=min_examples,
            is_latest=is_latest,
            limit=limit,
            sort_by=SortField.QUALITY_SCORE,
            sort_order=SortOrder.DESC
        )
        
        engine = DatasetSearchEngine(self.db)
        result = engine.search_versions(search_filter)
        
        if not result.versions:
            print("No versions found matching criteria")
            return
        
        print(f"\nFound {result.total_count} version(s):\n")
        
        for version in result.versions:
            print(f"  {version['dataset_name']} v{version['version']}")
            print(f"    Quality: {version.get('avg_quality_score', 'N/A')}")
            print(f"    Examples: {version['example_count']}")
            print(f"    Latest: {version['is_latest']}")
            print()
    
    # ========================================================================
    # Show Commands
    # ========================================================================
    
    def show_dataset(self, dataset_id: str):
        """
        Show dataset details
        
        Args:
            dataset_id: Dataset UUID
        """
        self.connect()
        
        logger.info(f"Showing dataset: {dataset_id}")
        
        dataset = self.db.get_dataset(dataset_id)
        
        if not dataset:
            print(f"Dataset not found: {dataset_id}")
            return
        
        print(f"\nDataset: {dataset.name}")
        print(f"  ID: {dataset.id}")
        print(f"  Agent type: {dataset.agent_type}")
        print(f"  Format: {dataset.format}")
        print(f"  Type: {dataset.dataset_type}")
        print(f"  Status: {dataset.status}")
        print(f"  Current version: {dataset.current_version or 'None'}")
        print(f"  Total examples: {dataset.total_examples}")
        print(f"  Total size: {dataset.total_size_bytes} bytes")
        print(f"  Created: {dataset.created_at}")
        print(f"  Updated: {dataset.updated_at}")
        
        if dataset.description:
            print(f"  Description: {dataset.description}")
        
        print()
    
    def show_version(self, version_id: str):
        """
        Show version details
        
        Args:
            version_id: Version UUID
        """
        self.connect()
        
        logger.info(f"Showing version: {version_id}")
        
        version = self.db.get_version(version_id)
        
        if not version:
            print(f"Version not found: {version_id}")
            return
        
        print(f"\nVersion: {version.version}")
        print(f"  ID: {version.id}")
        print(f"  Dataset ID: {version.dataset_id}")
        print(f"  File: {version.file_path}")
        print(f"  Size: {version.file_size_bytes} bytes")
        print(f"  SHA256: {version.sha256_hash}")
        print(f"  Examples: {version.example_count}")
        
        if version.avg_tokens_per_example:
            print(f"  Avg tokens/example: {version.avg_tokens_per_example:.1f}")
        
        if version.total_tokens:
            print(f"  Total tokens: {version.total_tokens}")
        
        if version.avg_quality_score:
            print(f"  Quality score: {version.avg_quality_score:.3f}")
            print(f"    Min: {version.min_quality_score:.3f}")
            print(f"    Max: {version.max_quality_score:.3f}")
        
        if version.avg_judge_score:
            print(f"  Judge score: {version.avg_judge_score:.3f}")
            print(f"  Judge pass rate: {version.judge_pass_rate:.1%}")
        
        print(f"  Latest: {version.is_latest}")
        print(f"  Created: {version.created_at}")
        
        if version.created_by:
            print(f"  Created by: {version.created_by}")
        
        if version.description:
            print(f"  Description: {version.description}")
        
        if version.tags:
            print(f"  Tags: {', '.join(version.tags)}")
        
        print()
    
    # ========================================================================
    # Lineage Commands
    # ========================================================================
    
    def show_lineage(
        self,
        dataset_id: str,
        format: str = "text"
    ):
        """
        Show lineage tree
        
        Args:
            dataset_id: Dataset UUID
            format: Output format (text, dot, mermaid)
        """
        self.connect()
        
        logger.info(f"Showing lineage for dataset: {dataset_id}")
        
        tracker = LineageTracker(self.db)
        visualization = tracker.visualize_lineage_tree(dataset_id, format)
        
        print(visualization)
    
    def show_lineage_stats(self, dataset_id: str):
        """
        Show lineage statistics
        
        Args:
            dataset_id: Dataset UUID
        """
        self.connect()
        
        logger.info(f"Calculating lineage stats for dataset: {dataset_id}")
        
        tracker = LineageTracker(self.db)
        stats = tracker.calculate_lineage_stats(dataset_id)
        
        print(f"\nLineage Statistics:")
        print(f"  Total versions: {stats['total_versions']}")
        print(f"  Root versions: {stats['root_versions']}")
        print(f"  Leaf versions: {stats['leaf_versions']}")
        print(f"  Max depth: {stats['max_depth']}")
        
        if stats['transformation_counts']:
            print(f"\n  Transformation counts:")
            for t_type, count in stats['transformation_counts'].items():
                print(f"    {t_type}: {count}")
        
        print(f"\n  Example changes:")
        print(f"    Inherited: {stats['total_examples_inherited']}")
        print(f"    New: {stats['total_examples_new']}")
        print(f"    Modified: {stats['total_examples_modified']}")
        print(f"    Removed: {stats['total_examples_removed']}")
        print()
    
    # ========================================================================
    # Integrity Commands
    # ========================================================================
    
    def verify_integrity(
        self,
        version_id: str,
        file_path: Optional[str] = None
    ):
        """
        Verify file integrity
        
        Args:
            version_id: Version UUID
            file_path: Path to file (optional, uses version.file_path if not provided)
        """
        self.connect()
        
        logger.info(f"Verifying integrity for version: {version_id}")
        
        version = self.db.get_version(version_id)
        
        if not version:
            print(f"Version not found: {version_id}")
            return
        
        file_to_check = file_path or version.file_path
        
        print(f"\nVerifying: {file_to_check}")
        print(f"Expected hash: {version.sha256_hash}")
        
        manager = DatasetIntegrityManager()
        result = manager.verify_dataset_integrity(file_to_check, version.sha256_hash)
        
        if result.is_valid:
            print(f"✓ Integrity verified")
            print(f"  Actual hash: {result.actual_hash}")
            print(f"  File size: {result.size_bytes} bytes")
        else:
            print(f"✗ Integrity check failed")
            print(f"  Expected: {result.expected_hash}")
            print(f"  Actual: {result.actual_hash}")
            print(f"  Error: {result.error}")
        
        print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Dataset Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Database connection options
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", type=int, default=5432, help="Database port")
    parser.add_argument("--database", default="stock_agent", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="", help="Database password")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Schema commands
    init_parser = subparsers.add_parser("init-schema", help="Initialize database schema")
    init_parser.add_argument("--schema-file", help="Path to schema SQL file")
    
    subparsers.add_parser("check-schema", help="Check schema version")
    
    # List commands
    list_datasets_parser = subparsers.add_parser("list-datasets", help="List datasets")
    list_datasets_parser.add_argument("--agent-type", help="Filter by agent type")
    list_datasets_parser.add_argument("--format", help="Filter by format")
    list_datasets_parser.add_argument("--status", help="Filter by status")
    list_datasets_parser.add_argument("--limit", type=int, help="Maximum results")
    
    list_versions_parser = subparsers.add_parser("list-versions", help="List versions")
    list_versions_parser.add_argument("--dataset-id", help="Filter by dataset ID")
    list_versions_parser.add_argument("--agent-type", help="Filter by agent type")
    list_versions_parser.add_argument("--latest", action="store_true", help="Latest only")
    list_versions_parser.add_argument("--min-quality", type=float, help="Min quality")
    list_versions_parser.add_argument("--limit", type=int, help="Maximum results")
    
    # Search commands
    search_datasets_parser = subparsers.add_parser("search-datasets", help="Search datasets")
    search_datasets_parser.add_argument("query", help="Search query")
    search_datasets_parser.add_argument("--agent-type", help="Filter by agent type")
    search_datasets_parser.add_argument("--format", help="Filter by format")
    search_datasets_parser.add_argument("--limit", type=int, default=10, help="Max results")
    
    search_versions_parser = subparsers.add_parser("search-versions", help="Search versions")
    search_versions_parser.add_argument("--agent-type", help="Filter by agent type")
    search_versions_parser.add_argument("--min-quality", type=float, help="Min quality")
    search_versions_parser.add_argument("--min-examples", type=int, help="Min examples")
    search_versions_parser.add_argument("--latest", action="store_true", help="Latest only")
    search_versions_parser.add_argument("--limit", type=int, default=10, help="Max results")
    
    # Show commands
    show_dataset_parser = subparsers.add_parser("show-dataset", help="Show dataset details")
    show_dataset_parser.add_argument("dataset_id", help="Dataset UUID")
    
    show_version_parser = subparsers.add_parser("show-version", help="Show version details")
    show_version_parser.add_argument("version_id", help="Version UUID")
    
    # Lineage commands
    lineage_parser = subparsers.add_parser("show-lineage", help="Show lineage tree")
    lineage_parser.add_argument("dataset_id", help="Dataset UUID")
    lineage_parser.add_argument("--format", default="text", choices=["text", "dot", "mermaid"])
    
    lineage_stats_parser = subparsers.add_parser("lineage-stats", help="Show lineage stats")
    lineage_stats_parser.add_argument("dataset_id", help="Dataset UUID")
    
    # Integrity commands
    verify_parser = subparsers.add_parser("verify", help="Verify file integrity")
    verify_parser.add_argument("version_id", help="Version UUID")
    verify_parser.add_argument("--file", help="File path (optional)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Build database config
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    # Execute command
    cli = RegistryCLI(db_config)
    
    try:
        if args.command == "init-schema":
            cli.init_schema(args.schema_file)
        
        elif args.command == "check-schema":
            cli.check_schema()
        
        elif args.command == "list-datasets":
            cli.list_datasets(
                agent_type=args.agent_type,
                format=args.format,
                status=args.status,
                limit=args.limit
            )
        
        elif args.command == "list-versions":
            cli.list_versions(
                dataset_id=args.dataset_id,
                agent_type=args.agent_type,
                is_latest=args.latest,
                min_quality=args.min_quality,
                limit=args.limit
            )
        
        elif args.command == "search-datasets":
            cli.search_datasets(
                query=args.query,
                agent_type=args.agent_type,
                format=args.format,
                limit=args.limit
            )
        
        elif args.command == "search-versions":
            cli.search_versions(
                agent_type=args.agent_type,
                min_quality=args.min_quality,
                min_examples=args.min_examples,
                is_latest=args.latest,
                limit=args.limit
            )
        
        elif args.command == "show-dataset":
            cli.show_dataset(args.dataset_id)
        
        elif args.command == "show-version":
            cli.show_version(args.version_id)
        
        elif args.command == "show-lineage":
            cli.show_lineage(args.dataset_id, args.format)
        
        elif args.command == "lineage-stats":
            cli.show_lineage_stats(args.dataset_id)
        
        elif args.command == "verify":
            cli.verify_integrity(args.version_id, args.file)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        cli.disconnect()


if __name__ == "__main__":
    main()
