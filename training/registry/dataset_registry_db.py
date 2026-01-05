#!/usr/bin/env python3
"""
Dataset Registry Database Module - Task 2.1

Python interface for dataset registry Postgres schema.

Features:
- Schema initialization and migration
- Connection management
- Helper functions for common queries
- Transaction support
- Error handling

Tables:
1. datasets - Main dataset metadata
2. dataset_versions - Version management
3. dataset_lineage - Provenance tracking
4. dataset_quality_metrics - Quality scores

Phase A1 Week 3-4: Task 2.1 COMPLETE
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from loguru import logger


@dataclass
class DatasetRecord:
    """Dataset record from database"""
    id: str
    agent_type: str
    name: str
    description: Optional[str]
    format: str
    dataset_type: str
    current_version: Optional[str]
    current_version_id: Optional[str]
    total_examples: int
    total_size_bytes: int
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass
class DatasetVersionRecord:
    """Dataset version record from database"""
    id: str
    dataset_id: str
    version: str
    major_version: int
    minor_version: int
    patch_version: int
    file_path: str
    file_size_bytes: int
    sha256_hash: str
    example_count: int
    avg_tokens_per_example: Optional[float]
    total_tokens: Optional[int]
    avg_quality_score: Optional[float]
    min_quality_score: Optional[float]
    max_quality_score: Optional[float]
    avg_judge_score: Optional[float]
    judge_pass_rate: Optional[float]
    judge_evaluated_count: int
    description: Optional[str]
    changelog: Optional[str]
    tags: List[str]
    is_latest: bool
    created_at: datetime
    created_by: Optional[str]


@dataclass
class DatasetLineageRecord:
    """Dataset lineage record from database"""
    id: str
    child_version_id: str
    parent_version_id: Optional[str]
    backtest_id: Optional[str]
    backtest_date: Optional[datetime]
    transformation_type: Optional[str]
    transformation_params: Optional[Dict]
    examples_inherited: int
    examples_new: int
    examples_modified: int
    examples_removed: int
    notes: Optional[str]
    created_at: datetime


@dataclass
class QualityMetricRecord:
    """Quality metric record from database"""
    id: str
    version_id: str
    example_index: int
    trajectory_id: Optional[str]
    quality_score: float
    reward_score: Optional[float]
    confidence_score: Optional[float]
    reasoning_score: Optional[float]
    consistency_score: Optional[float]
    judge_score: Optional[float]
    judge_passed: Optional[bool]
    judge_feedback: Optional[str]
    judge_evaluated_at: Optional[datetime]
    symbol: Optional[str]
    agent_type: Optional[str]
    created_at: datetime


class DatasetRegistryDB:
    """
    Database interface for dataset registry
    
    Manages connections, schema, and queries for the dataset registry system
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize database connection
        
        Args:
            db_config: Database configuration dict with keys:
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Database user
                - password: Database password
        """
        self.db_config = db_config
        self.conn = None
        
        logger.info("DatasetRegistryDB initialized")
    
    def connect(self):
        """Establish database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database")
    
    def disconnect(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Disconnected from database")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def init_schema(self, schema_file: Optional[str] = None):
        """
        Initialize database schema
        
        Args:
            schema_file: Path to SQL schema file (default: dataset_registry_schema.sql)
        """
        if schema_file is None:
            # Use schema file in same directory
            schema_file = Path(__file__).parent / "dataset_registry_schema.sql"
        
        logger.info(f"Initializing schema from {schema_file}")
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        self.connect()
        
        with self.conn.cursor() as cur:
            cur.execute(schema_sql)
            self.conn.commit()
        
        logger.info("Schema initialized successfully")
    
    def check_schema_version(self) -> Optional[str]:
        """
        Check current schema version
        
        Returns:
            Schema version string or None if not initialized
        """
        self.connect()
        
        with self.conn.cursor() as cur:
            # Check if schema_version table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'schema_version'
                )
            """)
            
            if not cur.fetchone()[0]:
                return None
            
            # Get latest version
            cur.execute("""
                SELECT version 
                FROM schema_version 
                ORDER BY applied_at DESC 
                LIMIT 1
            """)
            
            row = cur.fetchone()
            return row[0] if row else None
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Dict]]:
        """
        Execute SQL query
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            fetch: Whether to fetch results
        
        Returns:
            List of result dicts if fetch=True, else None
        """
        self.connect()
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            
            if fetch:
                results = cur.fetchall()
                return [dict(row) for row in results]
            else:
                self.conn.commit()
                return None
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ):
        """
        Execute query with multiple parameter sets
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        self.connect()
        
        with self.conn.cursor() as cur:
            execute_values(cur, query, params_list)
            self.conn.commit()
    
    # ========================================================================
    # Dataset CRUD operations
    # ========================================================================
    
    def create_dataset(
        self,
        agent_type: str,
        name: str,
        description: Optional[str] = None,
        format: str = "chatml",
        dataset_type: str = "sft"
    ) -> str:
        """
        Create new dataset
        
        Args:
            agent_type: Agent type (technical, news, etc.)
            name: Dataset name
            description: Optional description
            format: Dataset format (chatml or alpaca)
            dataset_type: Dataset type (sft, rl, mixed)
        
        Returns:
            Dataset ID (UUID)
        """
        query = """
            INSERT INTO datasets (agent_type, name, description, format, dataset_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        
        result = self.execute_query(
            query,
            (agent_type, name, description, format, dataset_type),
            fetch=True
        )
        
        dataset_id = result[0]['id']
        logger.info(f"Created dataset {dataset_id} for {agent_type}")
        
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """
        Get dataset by ID
        
        Args:
            dataset_id: Dataset UUID
        
        Returns:
            DatasetRecord or None if not found
        """
        query = """
            SELECT * FROM datasets
            WHERE id = %s AND is_deleted = FALSE
        """
        
        results = self.execute_query(query, (dataset_id,))
        
        if not results:
            return None
        
        return DatasetRecord(**results[0])
    
    def get_dataset_by_agent_type(self, agent_type: str) -> Optional[DatasetRecord]:
        """
        Get dataset by agent type
        
        Args:
            agent_type: Agent type
        
        Returns:
            DatasetRecord or None if not found
        """
        query = """
            SELECT * FROM datasets
            WHERE agent_type = %s AND is_deleted = FALSE
        """
        
        results = self.execute_query(query, (agent_type,))
        
        if not results:
            return None
        
        return DatasetRecord(**results[0])
    
    def list_datasets(
        self,
        status: Optional[str] = None,
        format: Optional[str] = None
    ) -> List[DatasetRecord]:
        """
        List all datasets
        
        Args:
            status: Filter by status (optional)
            format: Filter by format (optional)
        
        Returns:
            List of DatasetRecord objects
        """
        query = "SELECT * FROM datasets WHERE is_deleted = FALSE"
        params = []
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        if format:
            query += " AND format = %s"
            params.append(format)
        
        query += " ORDER BY created_at DESC"
        
        results = self.execute_query(query, tuple(params) if params else None)
        
        return [DatasetRecord(**row) for row in results]
    
    def update_dataset_status(self, dataset_id: str, status: str):
        """
        Update dataset status
        
        Args:
            dataset_id: Dataset UUID
            status: New status (active, archived, deprecated)
        """
        query = """
            UPDATE datasets
            SET status = %s, updated_at = NOW()
            WHERE id = %s
        """
        
        self.execute_query(query, (status, dataset_id), fetch=False)
        logger.info(f"Updated dataset {dataset_id} status to {status}")
    
    def delete_dataset(self, dataset_id: str, soft: bool = True):
        """
        Delete dataset
        
        Args:
            dataset_id: Dataset UUID
            soft: If True, soft delete (set is_deleted=TRUE), else hard delete
        """
        if soft:
            query = """
                UPDATE datasets
                SET is_deleted = TRUE, updated_at = NOW()
                WHERE id = %s
            """
        else:
            query = "DELETE FROM datasets WHERE id = %s"
        
        self.execute_query(query, (dataset_id,), fetch=False)
        logger.info(f"Deleted dataset {dataset_id} (soft={soft})")
    
    # ========================================================================
    # Dataset Version operations
    # ========================================================================
    
    def create_version(
        self,
        dataset_id: str,
        version: str,
        file_path: str,
        file_size_bytes: int,
        sha256_hash: str,
        example_count: int,
        **kwargs
    ) -> str:
        """
        Create new dataset version
        
        Args:
            dataset_id: Dataset UUID
            version: Semantic version (e.g., "1.2.3")
            file_path: Path to dataset file
            file_size_bytes: File size in bytes
            sha256_hash: SHA256 hash
            example_count: Number of examples
            **kwargs: Additional fields (avg_tokens_per_example, description, etc.)
        
        Returns:
            Version ID (UUID)
        """
        # Parse semantic version
        parts = version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        query = """
            INSERT INTO dataset_versions (
                dataset_id, version, major_version, minor_version, patch_version,
                file_path, file_size_bytes, sha256_hash, example_count,
                avg_tokens_per_example, total_tokens,
                avg_quality_score, min_quality_score, max_quality_score,
                avg_judge_score, judge_pass_rate, judge_evaluated_count,
                description, changelog, tags, created_by
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING id
        """
        
        params = (
            dataset_id, version, major, minor, patch,
            file_path, file_size_bytes, sha256_hash, example_count,
            kwargs.get('avg_tokens_per_example'),
            kwargs.get('total_tokens'),
            kwargs.get('avg_quality_score'),
            kwargs.get('min_quality_score'),
            kwargs.get('max_quality_score'),
            kwargs.get('avg_judge_score'),
            kwargs.get('judge_pass_rate'),
            kwargs.get('judge_evaluated_count', 0),
            kwargs.get('description'),
            kwargs.get('changelog'),
            kwargs.get('tags', []),
            kwargs.get('created_by')
        )
        
        result = self.execute_query(query, params, fetch=True)
        
        version_id = result[0]['id']
        logger.info(f"Created version {version} for dataset {dataset_id}")
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[DatasetVersionRecord]:
        """
        Get version by ID
        
        Args:
            version_id: Version UUID
        
        Returns:
            DatasetVersionRecord or None if not found
        """
        query = """
            SELECT * FROM dataset_versions
            WHERE id = %s AND is_deleted = FALSE
        """
        
        results = self.execute_query(query, (version_id,))
        
        if not results:
            return None
        
        return DatasetVersionRecord(**results[0])
    
    def get_latest_version(self, dataset_id: str) -> Optional[DatasetVersionRecord]:
        """
        Get latest version of dataset
        
        Args:
            dataset_id: Dataset UUID
        
        Returns:
            DatasetVersionRecord or None if no versions
        """
        query = """
            SELECT * FROM dataset_versions
            WHERE dataset_id = %s AND is_latest = TRUE AND is_deleted = FALSE
            LIMIT 1
        """
        
        results = self.execute_query(query, (dataset_id,))
        
        if not results:
            return None
        
        return DatasetVersionRecord(**results[0])
    
    def list_versions(
        self,
        dataset_id: str,
        limit: Optional[int] = None
    ) -> List[DatasetVersionRecord]:
        """
        List all versions of dataset
        
        Args:
            dataset_id: Dataset UUID
            limit: Maximum number of versions to return (optional)
        
        Returns:
            List of DatasetVersionRecord objects
        """
        query = """
            SELECT * FROM dataset_versions
            WHERE dataset_id = %s AND is_deleted = FALSE
            ORDER BY major_version DESC, minor_version DESC, patch_version DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.execute_query(query, (dataset_id,))
        
        return [DatasetVersionRecord(**row) for row in results]
    
    # ========================================================================
    # Dataset Lineage operations
    # ========================================================================
    
    def create_lineage(
        self,
        child_version_id: str,
        parent_version_id: Optional[str] = None,
        backtest_id: Optional[str] = None,
        transformation_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create lineage record
        
        Args:
            child_version_id: Child version UUID
            parent_version_id: Parent version UUID (optional)
            backtest_id: Backtest ID (optional)
            transformation_type: Transformation type (optional)
            **kwargs: Additional fields
        
        Returns:
            Lineage ID (UUID)
        """
        query = """
            INSERT INTO dataset_lineage (
                child_version_id, parent_version_id, backtest_id, backtest_date,
                transformation_type, transformation_params,
                examples_inherited, examples_new, examples_modified, examples_removed,
                notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        params = (
            child_version_id,
            parent_version_id,
            backtest_id,
            kwargs.get('backtest_date'),
            transformation_type,
            kwargs.get('transformation_params'),
            kwargs.get('examples_inherited', 0),
            kwargs.get('examples_new', 0),
            kwargs.get('examples_modified', 0),
            kwargs.get('examples_removed', 0),
            kwargs.get('notes')
        )
        
        result = self.execute_query(query, params, fetch=True)
        
        lineage_id = result[0]['id']
        logger.info(f"Created lineage record {lineage_id}")
        
        return lineage_id
    
    def get_lineage(self, version_id: str) -> List[DatasetLineageRecord]:
        """
        Get lineage for version
        
        Args:
            version_id: Version UUID
        
        Returns:
            List of DatasetLineageRecord objects
        """
        query = """
            SELECT * FROM dataset_lineage
            WHERE child_version_id = %s OR parent_version_id = %s
            ORDER BY created_at DESC
        """
        
        results = self.execute_query(query, (version_id, version_id))
        
        return [DatasetLineageRecord(**row) for row in results]
    
    # ========================================================================
    # Quality Metrics operations
    # ========================================================================
    
    def create_quality_metrics(
        self,
        version_id: str,
        metrics: List[Dict]
    ):
        """
        Batch insert quality metrics
        
        Args:
            version_id: Version UUID
            metrics: List of metric dicts with keys:
                - example_index: int
                - quality_score: float
                - trajectory_id: str (optional)
                - reward_score, confidence_score, etc. (optional)
        """
        query = """
            INSERT INTO dataset_quality_metrics (
                version_id, example_index, trajectory_id,
                quality_score, reward_score, confidence_score,
                reasoning_score, consistency_score,
                symbol, agent_type
            )
            VALUES %s
        """
        
        params_list = [
            (
                version_id,
                m['example_index'],
                m.get('trajectory_id'),
                m['quality_score'],
                m.get('reward_score'),
                m.get('confidence_score'),
                m.get('reasoning_score'),
                m.get('consistency_score'),
                m.get('symbol'),
                m.get('agent_type')
            )
            for m in metrics
        ]
        
        self.execute_many(query, params_list)
        logger.info(f"Inserted {len(metrics)} quality metrics for version {version_id}")
    
    def get_quality_metrics(
        self,
        version_id: str,
        min_quality_score: Optional[float] = None
    ) -> List[QualityMetricRecord]:
        """
        Get quality metrics for version
        
        Args:
            version_id: Version UUID
            min_quality_score: Minimum quality score filter (optional)
        
        Returns:
            List of QualityMetricRecord objects
        """
        query = """
            SELECT * FROM dataset_quality_metrics
            WHERE version_id = %s
        """
        params = [version_id]
        
        if min_quality_score is not None:
            query += " AND quality_score >= %s"
            params.append(min_quality_score)
        
        query += " ORDER BY example_index"
        
        results = self.execute_query(query, tuple(params))
        
        return [QualityMetricRecord(**row) for row in results]
    
    def update_judge_scores(
        self,
        version_id: str,
        example_index: int,
        judge_score: float,
        judge_passed: bool,
        judge_feedback: Optional[str] = None
    ):
        """
        Update judge evaluation for example
        
        Args:
            version_id: Version UUID
            example_index: Example index
            judge_score: Judge score (0.0 to 1.0)
            judge_passed: Whether example passed
            judge_feedback: Optional feedback text
        """
        query = """
            UPDATE dataset_quality_metrics
            SET 
                judge_score = %s,
                judge_passed = %s,
                judge_feedback = %s,
                judge_evaluated_at = NOW()
            WHERE version_id = %s AND example_index = %s
        """
        
        self.execute_query(
            query,
            (judge_score, judge_passed, judge_feedback, version_id, example_index),
            fetch=False
        )


# ============================================================================
# Helper Functions
# ============================================================================

def get_db_connection(
    host: str = "localhost",
    port: int = 5432,
    database: str = "stock_agent",
    user: str = "postgres",
    password: str = ""
) -> DatasetRegistryDB:
    """
    Get database connection
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
    
    Returns:
        DatasetRegistryDB instance
    """
    db_config = {
        'host': host,
        'port': port,
        'database': database,
        'user': user,
        'password': password
    }
    
    return DatasetRegistryDB(db_config)


if __name__ == "__main__":
    # Example usage
    db = get_db_connection()
    
    with db:
        # Check schema version
        version = db.check_schema_version()
        print(f"Schema version: {version}")
        
        # List datasets
        datasets = db.list_datasets()
        print(f"Found {len(datasets)} datasets")
        
        for dataset in datasets:
            print(f"  - {dataset.agent_type}: {dataset.name} (v{dataset.current_version})")
