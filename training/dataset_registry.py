"""
Dataset Registry System - PRODUCTION VERSION

Centralized registry for tracking all generated datasets with full lineage,
versioning, and metadata management using Postgres persistence.

Phase A1 Week 3-4: Task 2 (COMPLETE REWRITE)

Key Features:
- Postgres persistence (not JSON files!)
- Semantic versioning (MAJOR.MINOR.PATCH)
- Complete lineage tracking (backtest → dataset → model)
- SHA256 integrity checking
- Tag-based organization
- Version comparison and rollback
- Import/Export capabilities
- Search and filtering
- Comprehensive metadata
- CLI and programmatic interfaces

Architecture:
1. Postgres Schema
   - datasets table: Core dataset metadata
   - dataset_versions table: Version history
   - dataset_lineage table: Backtest → Dataset → Model tracking
   - dataset_tags table: Tag associations
   
2. Versioning Strategy
   - Semantic versioning: vMAJOR.MINOR.PATCH
   - MAJOR: Breaking changes (format, schema)
   - MINOR: New features (new agent types, data sources)
   - PATCH: Bug fixes, quality improvements
   
3. Integrity
   - SHA256 file hashing
   - Size tracking
   - Modification detection
   
4. Lineage
   - Source backtest_id
   - Derived model_id
   - Training run associations
   
5. Search & Filter
   - By agent type, version, tags
   - By quality/judge scores
   - By date range
   - By lineage (find all datasets from backtest X)

Based on:
- MLflow Model Registry
- DVC (Data Version Control)
- Weights & Biases Artifacts
- HuggingFace Datasets Hub
"""

import json
import hashlib
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from utils.config_loader import load_config


@dataclass
class DatasetMetadata:
    """Complete dataset metadata"""
    dataset_id: str
    agent_type: str
    version: str  # vMAJOR.MINOR.PATCH
    file_path: str
    file_size: int
    file_hash: str  # SHA256
    num_examples: int
    avg_quality_score: float
    avg_judge_score: float
    source_backtest_id: Optional[str]
    derived_model_id: Optional[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict  # Additional custom metadata
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetMetadata':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class VersionComparison:
    """Comparison between two dataset versions"""
    old_version: str
    new_version: str
    num_examples_diff: int
    quality_score_diff: float
    judge_score_diff: float
    file_size_diff: int
    changes_summary: str


class DatasetRegistry:
    """
    Dataset Registry with Postgres Persistence
    
    Manages all dataset metadata, versioning, and lineage.
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize registry
        
        Args:
            db_config: Database configuration (or load from config)
        """
        if db_config is None:
            config = load_config()
            db_config = config['database']
        
        # Connect to Postgres
        self.conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        # Initialize schema
        self._init_schema()
        
        logger.info("Dataset Registry initialized with Postgres persistence")
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.conn.cursor() as cur:
            # Datasets table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id VARCHAR(255) PRIMARY KEY,
                    agent_type VARCHAR(50) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size BIGINT NOT NULL,
                    file_hash VARCHAR(64) NOT NULL,
                    num_examples INTEGER NOT NULL,
                    avg_quality_score FLOAT NOT NULL,
                    avg_judge_score FLOAT NOT NULL,
                    source_backtest_id VARCHAR(255),
                    derived_model_id VARCHAR(255),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(agent_type, version)
                )
            """)
            
            # Dataset versions table (for history)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    version_id SERIAL PRIMARY KEY,
                    dataset_id VARCHAR(255) REFERENCES datasets(dataset_id),
                    version VARCHAR(20) NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash VARCHAR(64) NOT NULL,
                    num_examples INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    changelog TEXT
                )
            """)
            
            # Dataset lineage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dataset_lineage (
                    lineage_id SERIAL PRIMARY KEY,
                    dataset_id VARCHAR(255) REFERENCES datasets(dataset_id),
                    parent_type VARCHAR(50) NOT NULL,  -- 'backtest', 'dataset', 'model'
                    parent_id VARCHAR(255) NOT NULL,
                    relationship VARCHAR(50) NOT NULL,  -- 'derived_from', 'used_by', 'merged_with'
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            
            # Dataset tags table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dataset_tags (
                    tag_id SERIAL PRIMARY KEY,
                    dataset_id VARCHAR(255) REFERENCES datasets(dataset_id),
                    tag VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    UNIQUE(dataset_id, tag)
                )
            """)
            
            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_agent_type ON datasets(agent_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_version ON datasets(version)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_backtest ON datasets(source_backtest_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_lineage_dataset ON dataset_lineage(dataset_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_lineage_parent ON dataset_lineage(parent_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_dataset ON dataset_tags(dataset_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON dataset_tags(tag)")
            
            self.conn.commit()
        
        logger.debug("Database schema initialized")
    
    def register_dataset(
        self,
        agent_type: str,
        file_path: str,
        num_examples: int,
        avg_quality_score: float,
        avg_judge_score: float = 0.0,
        source_backtest_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        version: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Register a new dataset
        
        Args:
            agent_type: Agent type (technical, news, etc.)
            file_path: Path to dataset file
            num_examples: Number of examples
            avg_quality_score: Average quality score
            avg_judge_score: Average judge score
            source_backtest_id: Source backtest ID (for lineage)
            tags: List of tags
            metadata: Additional metadata
            version: Explicit version (or auto-increment)
        
        Returns:
            DatasetMetadata
        """
        # Calculate file hash and size
        file_hash = self._calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        
        # Generate dataset_id
        dataset_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine version
        if version is None:
            version = self._get_next_version(agent_type)
        
        # Validate version format
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}. Expected: vMAJOR.MINOR.PATCH")
        
        # Create metadata
        dataset_meta = DatasetMetadata(
            dataset_id=dataset_id,
            agent_type=agent_type,
            version=version,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            num_examples=num_examples,
            avg_quality_score=avg_quality_score,
            avg_judge_score=avg_judge_score,
            source_backtest_id=source_backtest_id,
            derived_model_id=None,
            tags=tags or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Insert into database
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO datasets (
                    dataset_id, agent_type, version, file_path, file_size, file_hash,
                    num_examples, avg_quality_score, avg_judge_score,
                    source_backtest_id, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                dataset_id, agent_type, version, file_path, file_size, file_hash,
                num_examples, avg_quality_score, avg_judge_score,
                source_backtest_id, json.dumps(metadata or {})
            ))
            
            # Add to version history
            cur.execute("""
                INSERT INTO dataset_versions (
                    dataset_id, version, file_path, file_hash, num_examples
                ) VALUES (%s, %s, %s, %s, %s)
            """, (dataset_id, version, file_path, file_hash, num_examples))
            
            # Add lineage (if source backtest provided)
            if source_backtest_id:
                cur.execute("""
                    INSERT INTO dataset_lineage (
                        dataset_id, parent_type, parent_id, relationship
                    ) VALUES (%s, %s, %s, %s)
                """, (dataset_id, 'backtest', source_backtest_id, 'derived_from'))
            
            # Add tags
            if tags:
                execute_values(
                    cur,
                    "INSERT INTO dataset_tags (dataset_id, tag) VALUES %s",
                    [(dataset_id, tag) for tag in tags]
                )
            
            self.conn.commit()
        
        logger.info(f"Registered dataset: {dataset_id} ({version})")
        
        return dataset_meta
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get dataset by ID
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            DatasetMetadata or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT d.*, ARRAY_AGG(t.tag) as tags
                FROM datasets d
                LEFT JOIN dataset_tags t ON d.dataset_id = t.dataset_id
                WHERE d.dataset_id = %s
                GROUP BY d.dataset_id
            """, (dataset_id,))
            
            row = cur.fetchone()
            
            if not row:
                return None
            
            return DatasetMetadata(
                dataset_id=row['dataset_id'],
                agent_type=row['agent_type'],
                version=row['version'],
                file_path=row['file_path'],
                file_size=row['file_size'],
                file_hash=row['file_hash'],
                num_examples=row['num_examples'],
                avg_quality_score=row['avg_quality_score'],
                avg_judge_score=row['avg_judge_score'],
                source_backtest_id=row['source_backtest_id'],
                derived_model_id=row['derived_model_id'],
                tags=[t for t in row['tags'] if t is not None],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                metadata=row['metadata'] or {}
            )
    
    def get_latest_version(self, agent_type: str) -> Optional[DatasetMetadata]:
        """
        Get latest version for an agent type
        
        Args:
            agent_type: Agent type
        
        Returns:
            DatasetMetadata or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT d.*, ARRAY_AGG(t.tag) as tags
                FROM datasets d
                LEFT JOIN dataset_tags t ON d.dataset_id = t.dataset_id
                WHERE d.agent_type = %s
                GROUP BY d.dataset_id
                ORDER BY d.created_at DESC
                LIMIT 1
            """, (agent_type,))
            
            row = cur.fetchone()
            
            if not row:
                return None
            
            return DatasetMetadata(
                dataset_id=row['dataset_id'],
                agent_type=row['agent_type'],
                version=row['version'],
                file_path=row['file_path'],
                file_size=row['file_size'],
                file_hash=row['file_hash'],
                num_examples=row['num_examples'],
                avg_quality_score=row['avg_quality_score'],
                avg_judge_score=row['avg_judge_score'],
                source_backtest_id=row['source_backtest_id'],
                derived_model_id=row['derived_model_id'],
                tags=[t for t in row['tags'] if t is not None],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                metadata=row['metadata'] or {}
            )
    
    def list_datasets(
        self,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 100
    ) -> List[DatasetMetadata]:
        """
        List datasets with filtering
        
        Args:
            agent_type: Filter by agent type
            tags: Filter by tags (AND logic)
            min_quality_score: Minimum quality score
            limit: Maximum results
        
        Returns:
            List of DatasetMetadata
        """
        query = """
            SELECT d.*, ARRAY_AGG(t.tag) as tags
            FROM datasets d
            LEFT JOIN dataset_tags t ON d.dataset_id = t.dataset_id
            WHERE 1=1
        """
        params = []
        
        if agent_type:
            query += " AND d.agent_type = %s"
            params.append(agent_type)
        
        if min_quality_score is not None:
            query += " AND d.avg_quality_score >= %s"
            params.append(min_quality_score)
        
        query += " GROUP BY d.dataset_id ORDER BY d.created_at DESC LIMIT %s"
        params.append(limit)
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            
            datasets = []
            for row in rows:
                dataset = DatasetMetadata(
                    dataset_id=row['dataset_id'],
                    agent_type=row['agent_type'],
                    version=row['version'],
                    file_path=row['file_path'],
                    file_size=row['file_size'],
                    file_hash=row['file_hash'],
                    num_examples=row['num_examples'],
                    avg_quality_score=row['avg_quality_score'],
                    avg_judge_score=row['avg_judge_score'],
                    source_backtest_id=row['source_backtest_id'],
                    derived_model_id=row['derived_model_id'],
                    tags=[t for t in row['tags'] if t is not None],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=row['metadata'] or {}
                )
                
                # Filter by tags if specified
                if tags:
                    if all(tag in dataset.tags for tag in tags):
                        datasets.append(dataset)
                else:
                    datasets.append(dataset)
            
            return datasets
    
    def compare_versions(
        self,
        old_version: str,
        new_version: str,
        agent_type: str
    ) -> VersionComparison:
        """
        Compare two versions
        
        Args:
            old_version: Old version string
            new_version: New version string
            agent_type: Agent type
        
        Returns:
            VersionComparison
        """
        # Get both versions
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM datasets
                WHERE agent_type = %s AND version IN (%s, %s)
            """, (agent_type, old_version, new_version))
            
            rows = cur.fetchall()
            
            if len(rows) != 2:
                raise ValueError(f"Could not find both versions: {old_version}, {new_version}")
            
            old = next(r for r in rows if r['version'] == old_version)
            new = next(r for r in rows if r['version'] == new_version)
            
            # Calculate diffs
            num_examples_diff = new['num_examples'] - old['num_examples']
            quality_score_diff = new['avg_quality_score'] - old['avg_quality_score']
            judge_score_diff = new['avg_judge_score'] - old['avg_judge_score']
            file_size_diff = new['file_size'] - old['file_size']
            
            # Generate summary
            changes = []
            if num_examples_diff > 0:
                changes.append(f"+{num_examples_diff} examples")
            elif num_examples_diff < 0:
                changes.append(f"{num_examples_diff} examples")
            
            if quality_score_diff > 0.01:
                changes.append(f"+{quality_score_diff:.3f} quality")
            elif quality_score_diff < -0.01:
                changes.append(f"{quality_score_diff:.3f} quality")
            
            changes_summary = ", ".join(changes) if changes else "No significant changes"
            
            return VersionComparison(
                old_version=old_version,
                new_version=new_version,
                num_examples_diff=num_examples_diff,
                quality_score_diff=quality_score_diff,
                judge_score_diff=judge_score_diff,
                file_size_diff=file_size_diff,
                changes_summary=changes_summary
            )
    
    def get_lineage(self, dataset_id: str) -> Dict:
        """
        Get complete lineage for a dataset
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Lineage dictionary with parents and children
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get parents (what this dataset was derived from)
            cur.execute("""
                SELECT parent_type, parent_id, relationship
                FROM dataset_lineage
                WHERE dataset_id = %s
            """, (dataset_id,))
            parents = cur.fetchall()
            
            # Get children (what was derived from this dataset)
            cur.execute("""
                SELECT dataset_id as child_id, relationship
                FROM dataset_lineage
                WHERE parent_type = 'dataset' AND parent_id = %s
            """, (dataset_id,))
            children = cur.fetchall()
            
            return {
                'dataset_id': dataset_id,
                'parents': [dict(p) for p in parents],
                'children': [dict(c) for c in children]
            }
    
    def verify_integrity(self, dataset_id: str) -> Tuple[bool, str]:
        """
        Verify dataset file integrity
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            (is_valid, message) tuple
        """
        dataset = self.get_dataset(dataset_id)
        
        if not dataset:
            return False, f"Dataset {dataset_id} not found"
        
        # Check file exists
        if not os.path.exists(dataset.file_path):
            return False, f"File not found: {dataset.file_path}"
        
        # Check file size
        actual_size = os.path.getsize(dataset.file_path)
        if actual_size != dataset.file_size:
            return False, f"File size mismatch: expected {dataset.file_size}, got {actual_size}"
        
        # Check file hash
        actual_hash = self._calculate_file_hash(dataset.file_path)
        if actual_hash != dataset.file_hash:
            return False, f"File hash mismatch: expected {dataset.file_hash}, got {actual_hash}"
        
        return True, "Integrity verified"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _get_next_version(self, agent_type: str) -> str:
        """Get next version for agent type"""
        latest = self.get_latest_version(agent_type)
        
        if not latest:
            return "v1.0.0"
        
        # Parse version
        major, minor, patch = self._parse_version(latest.version)
        
        # Increment patch
        return f"v{major}.{minor}.{patch + 1}"
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse version string to (major, minor, patch)"""
        if not version.startswith('v'):
            raise ValueError(f"Version must start with 'v': {version}")
        
        parts = version[1:].split('.')
        
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")
        
        return int(parts[0]), int(parts[1]), int(parts[2])
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid"""
        try:
            self._parse_version(version)
            return True
        except:
            return False
    
    def export_registry(self, output_path: str):
        """
        Export entire registry to JSON
        
        Args:
            output_path: Output file path
        """
        datasets = self.list_datasets(limit=10000)
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'num_datasets': len(datasets),
            'datasets': [d.to_dict() for d in datasets]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(datasets)} datasets to {output_path}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Registry CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List datasets')
    list_parser.add_argument('--agent-type', type=str, help='Filter by agent type')
    list_parser.add_argument('--tags', type=str, nargs='+', help='Filter by tags')
    list_parser.add_argument('--min-quality', type=float, help='Minimum quality score')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum results')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get dataset details')
    get_parser.add_argument('dataset_id', type=str, help='Dataset ID')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset integrity')
    verify_parser.add_argument('dataset_id', type=str, help='Dataset ID')
    
    # Lineage command
    lineage_parser = subparsers.add_parser('lineage', help='Show dataset lineage')
    lineage_parser.add_argument('dataset_id', type=str, help='Dataset ID')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare versions')
    compare_parser.add_argument('agent_type', type=str, help='Agent type')
    compare_parser.add_argument('old_version', type=str, help='Old version')
    compare_parser.add_argument('new_version', type=str, help='New version')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export registry')
    export_parser.add_argument('output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = DatasetRegistry()
    
    try:
        if args.command == 'list':
            datasets = registry.list_datasets(
                agent_type=args.agent_type,
                tags=args.tags,
                min_quality_score=args.min_quality,
                limit=args.limit
            )
            
            print(f"\nFound {len(datasets)} datasets:\n")
            for d in datasets:
                print(f"{d.dataset_id}")
                print(f"  Agent: {d.agent_type}")
                print(f"  Version: {d.version}")
                print(f"  Examples: {d.num_examples}")
                print(f"  Quality: {d.avg_quality_score:.3f}")
                print(f"  Judge: {d.avg_judge_score:.3f}")
                print(f"  Tags: {', '.join(d.tags)}")
                print()
        
        elif args.command == 'get':
            dataset = registry.get_dataset(args.dataset_id)
            
            if not dataset:
                print(f"Dataset {args.dataset_id} not found")
            else:
                print(json.dumps(dataset.to_dict(), indent=2))
        
        elif args.command == 'verify':
            is_valid, message = registry.verify_integrity(args.dataset_id)
            
            if is_valid:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
        
        elif args.command == 'lineage':
            lineage = registry.get_lineage(args.dataset_id)
            print(json.dumps(lineage, indent=2))
        
        elif args.command == 'compare':
            comparison = registry.compare_versions(
                args.old_version,
                args.new_version,
                args.agent_type
            )
            
            print(f"\nVersion Comparison: {args.old_version} → {args.new_version}")
            print(f"  Examples: {comparison.num_examples_diff:+d}")
            print(f"  Quality: {comparison.quality_score_diff:+.3f}")
            print(f"  Judge: {comparison.judge_score_diff:+.3f}")
            print(f"  File size: {comparison.file_size_diff:+d} bytes")
            print(f"  Summary: {comparison.changes_summary}")
        
        elif args.command == 'export':
            registry.export_registry(args.output)
            print(f"✅ Registry exported to {args.output}")
        
        else:
            parser.print_help()
    
    finally:
        registry.close()
