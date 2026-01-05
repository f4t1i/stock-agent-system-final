"""
Dataset Registry System

Centralized registry for managing SFT training datasets with versioning,
metadata tracking, and quality metrics.

Phase A1 Week 3-4: Task 2
- Semantic versioning (major.minor.patch)
- Dataset metadata (size, quality, source)
- Version comparison and rollback
- Dataset lineage tracking
- Quality metrics aggregation

Based on:
- MLflow Model Registry
- Hugging Face Datasets versioning
- DVC (Data Version Control)
"""

import json
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger
import pandas as pd
import hashlib


@dataclass
class DatasetVersion:
    """Dataset version metadata"""
    version: str  # e.g., "v1.2.3"
    agent_type: str
    num_examples: int
    avg_quality_score: float
    avg_judge_score: float
    source_backtest_id: Optional[str]
    file_path: str
    file_hash: str  # SHA256 hash for integrity
    created_at: str
    created_by: str  # pipeline or user
    tags: List[str]  # e.g., ["high-quality", "bull-market"]
    metadata: Dict  # Additional metadata
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetVersion':
        return cls(**data)


@dataclass
class DatasetStats:
    """Aggregated dataset statistics"""
    total_versions: int
    total_examples: int
    latest_version: str
    avg_quality_score: float
    avg_judge_score: float
    version_history: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DatasetRegistry:
    """
    Dataset Registry System
    
    Manages SFT training datasets with versioning, metadata, and quality tracking.
    """
    
    def __init__(self, registry_dir: str = "datasets/registry"):
        """
        Initialize Dataset Registry
        
        Args:
            registry_dir: Registry directory path
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry database (JSON file)
        self.db_file = self.registry_dir / "registry.json"
        self.db = self._load_registry()
        
        logger.info(f"Dataset Registry initialized at {self.registry_dir}")
    
    def _load_registry(self) -> Dict:
        """Load registry database"""
        if self.db_file.exists():
            with open(self.db_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'datasets': {},  # agent_type -> List[DatasetVersion]
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            }
    
    def _save_registry(self):
        """Save registry database"""
        self.db['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.db_file, 'w') as f:
            json.dump(self.db, f, indent=2)
        logger.debug("Registry saved")
    
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
        auto_version: bool = True
    ) -> DatasetVersion:
        """
        Register a new dataset version
        
        Args:
            agent_type: Agent type (news, technical, fundamental, strategist)
            file_path: Path to dataset file
            num_examples: Number of examples
            avg_quality_score: Average quality score
            avg_judge_score: Average judge score
            source_backtest_id: Source backtest ID
            tags: Dataset tags
            metadata: Additional metadata
            auto_version: Auto-increment version
        
        Returns:
            DatasetVersion object
        """
        logger.info(f"Registering dataset for {agent_type}...")
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Generate version
        if auto_version:
            version = self._generate_next_version(agent_type)
        else:
            version = "v1.0.0"
        
        # Create dataset version
        dataset_version = DatasetVersion(
            version=version,
            agent_type=agent_type,
            num_examples=num_examples,
            avg_quality_score=avg_quality_score,
            avg_judge_score=avg_judge_score,
            source_backtest_id=source_backtest_id,
            file_path=file_path,
            file_hash=file_hash,
            created_at=datetime.now().isoformat(),
            created_by="auto_synthesis_pipeline",
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Add to registry
        if agent_type not in self.db['datasets']:
            self.db['datasets'][agent_type] = []
        
        self.db['datasets'][agent_type].append(dataset_version.to_dict())
        
        # Save registry
        self._save_registry()
        
        logger.info(
            f"✅ Registered {agent_type} dataset {version} "
            f"({num_examples} examples, quality={avg_quality_score:.3f})"
        )
        
        return dataset_version
    
    def get_latest_version(self, agent_type: str) -> Optional[DatasetVersion]:
        """
        Get latest dataset version for an agent
        
        Args:
            agent_type: Agent type
        
        Returns:
            Latest DatasetVersion or None
        """
        if agent_type not in self.db['datasets']:
            return None
        
        versions = self.db['datasets'][agent_type]
        if not versions:
            return None
        
        # Sort by version (descending)
        versions_sorted = sorted(
            versions,
            key=lambda v: self._version_to_tuple(v['version']),
            reverse=True
        )
        
        return DatasetVersion.from_dict(versions_sorted[0])
    
    def get_version(self, agent_type: str, version: str) -> Optional[DatasetVersion]:
        """
        Get specific dataset version
        
        Args:
            agent_type: Agent type
            version: Version string (e.g., "v1.2.3")
        
        Returns:
            DatasetVersion or None
        """
        if agent_type not in self.db['datasets']:
            return None
        
        for v in self.db['datasets'][agent_type]:
            if v['version'] == version:
                return DatasetVersion.from_dict(v)
        
        return None
    
    def list_versions(
        self,
        agent_type: str,
        tags: Optional[List[str]] = None,
        min_quality: Optional[float] = None
    ) -> List[DatasetVersion]:
        """
        List all dataset versions for an agent
        
        Args:
            agent_type: Agent type
            tags: Filter by tags
            min_quality: Minimum quality score
        
        Returns:
            List of DatasetVersion objects
        """
        if agent_type not in self.db['datasets']:
            return []
        
        versions = [DatasetVersion.from_dict(v) for v in self.db['datasets'][agent_type]]
        
        # Filter by tags
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        # Filter by quality
        if min_quality is not None:
            versions = [v for v in versions if v.avg_quality_score >= min_quality]
        
        # Sort by version (descending)
        versions.sort(key=lambda v: self._version_to_tuple(v.version), reverse=True)
        
        return versions
    
    def get_stats(self, agent_type: str) -> Optional[DatasetStats]:
        """
        Get aggregated statistics for an agent's datasets
        
        Args:
            agent_type: Agent type
        
        Returns:
            DatasetStats or None
        """
        versions = self.list_versions(agent_type)
        
        if not versions:
            return None
        
        return DatasetStats(
            total_versions=len(versions),
            total_examples=sum(v.num_examples for v in versions),
            latest_version=versions[0].version,
            avg_quality_score=sum(v.avg_quality_score for v in versions) / len(versions),
            avg_judge_score=sum(v.avg_judge_score for v in versions) / len(versions),
            version_history=[v.version for v in versions]
        )
    
    def compare_versions(
        self,
        agent_type: str,
        version1: str,
        version2: str
    ) -> Dict:
        """
        Compare two dataset versions
        
        Args:
            agent_type: Agent type
            version1: First version
            version2: Second version
        
        Returns:
            Comparison dict
        """
        v1 = self.get_version(agent_type, version1)
        v2 = self.get_version(agent_type, version2)
        
        if not v1 or not v2:
            raise ValueError(f"Version not found: {version1} or {version2}")
        
        return {
            'version1': version1,
            'version2': version2,
            'num_examples_diff': v2.num_examples - v1.num_examples,
            'quality_score_diff': v2.avg_quality_score - v1.avg_quality_score,
            'judge_score_diff': v2.avg_judge_score - v1.avg_judge_score,
            'created_at_diff': (
                datetime.fromisoformat(v2.created_at) - 
                datetime.fromisoformat(v1.created_at)
            ).total_seconds(),
            'file_hash_changed': v1.file_hash != v2.file_hash
        }
    
    def rollback_to_version(
        self,
        agent_type: str,
        version: str,
        target_path: str
    ) -> str:
        """
        Rollback to a specific dataset version
        
        Args:
            agent_type: Agent type
            version: Version to rollback to
            target_path: Target file path
        
        Returns:
            Path to rolled-back dataset
        """
        dataset_version = self.get_version(agent_type, version)
        
        if not dataset_version:
            raise ValueError(f"Version {version} not found for {agent_type}")
        
        # Copy file
        source_path = Path(dataset_version.file_path)
        target_path = Path(target_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {source_path}")
        
        shutil.copy(source_path, target_path)
        
        logger.info(f"✅ Rolled back {agent_type} to {version} → {target_path}")
        
        return str(target_path)
    
    def delete_version(
        self,
        agent_type: str,
        version: str,
        delete_file: bool = False
    ):
        """
        Delete a dataset version from registry
        
        Args:
            agent_type: Agent type
            version: Version to delete
            delete_file: Also delete the dataset file
        """
        if agent_type not in self.db['datasets']:
            raise ValueError(f"Agent type {agent_type} not found")
        
        # Find and remove version
        versions = self.db['datasets'][agent_type]
        version_to_delete = None
        
        for i, v in enumerate(versions):
            if v['version'] == version:
                version_to_delete = versions.pop(i)
                break
        
        if not version_to_delete:
            raise ValueError(f"Version {version} not found")
        
        # Delete file if requested
        if delete_file:
            file_path = Path(version_to_delete['file_path'])
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
        
        # Save registry
        self._save_registry()
        
        logger.info(f"✅ Deleted {agent_type} version {version}")
    
    def export_registry(self, output_path: str):
        """
        Export registry to JSON file
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.db, f, indent=2)
        logger.info(f"Registry exported to {output_path}")
    
    def import_registry(self, input_path: str, merge: bool = True):
        """
        Import registry from JSON file
        
        Args:
            input_path: Input file path
            merge: Merge with existing registry (True) or replace (False)
        """
        with open(input_path, 'r') as f:
            imported_db = json.load(f)
        
        if merge:
            # Merge datasets
            for agent_type, versions in imported_db['datasets'].items():
                if agent_type not in self.db['datasets']:
                    self.db['datasets'][agent_type] = []
                self.db['datasets'][agent_type].extend(versions)
        else:
            # Replace
            self.db = imported_db
        
        self._save_registry()
        logger.info(f"Registry imported from {input_path}")
    
    def _generate_next_version(self, agent_type: str) -> str:
        """
        Generate next semantic version
        
        Args:
            agent_type: Agent type
        
        Returns:
            Next version string (e.g., "v1.2.4")
        """
        latest = self.get_latest_version(agent_type)
        
        if not latest:
            return "v1.0.0"
        
        # Parse version
        major, minor, patch = self._version_to_tuple(latest.version)
        
        # Increment patch
        patch += 1
        
        return f"v{major}.{minor}.{patch}"
    
    def _version_to_tuple(self, version: str) -> Tuple[int, int, int]:
        """
        Convert version string to tuple
        
        Args:
            version: Version string (e.g., "v1.2.3")
        
        Returns:
            (major, minor, patch) tuple
        """
        # Remove 'v' prefix and any suffix
        version_clean = version.lstrip('v').split('_')[0]
        parts = version_clean.split('.')
        return tuple(int(p) for p in parts[:3])
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of file
        
        Args:
            file_path: File path
        
        Returns:
            SHA256 hash string
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def get_summary(self) -> Dict:
        """
        Get registry summary
        
        Returns:
            Summary dict with stats for all agents
        """
        summary = {
            'total_agents': len(self.db['datasets']),
            'agents': {}
        }
        
        for agent_type in self.db['datasets'].keys():
            stats = self.get_stats(agent_type)
            if stats:
                summary['agents'][agent_type] = stats.to_dict()
        
        return summary


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Registry CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List dataset versions')
    list_parser.add_argument('--agent-type', type=str, required=True)
    list_parser.add_argument('--tags', nargs='+', default=None)
    list_parser.add_argument('--min-quality', type=float, default=None)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--agent-type', type=str, required=True)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('--agent-type', type=str, required=True)
    compare_parser.add_argument('--version1', type=str, required=True)
    compare_parser.add_argument('--version2', type=str, required=True)
    
    # Summary command
    subparsers.add_parser('summary', help='Show registry summary')
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = DatasetRegistry()
    
    # Execute command
    if args.command == 'list':
        versions = registry.list_versions(
            agent_type=args.agent_type,
            tags=args.tags,
            min_quality=args.min_quality
        )
        print(f"\n{len(versions)} versions found for {args.agent_type}:\n")
        for v in versions:
            print(f"  {v.version}: {v.num_examples} examples, quality={v.avg_quality_score:.3f}, judge={v.avg_judge_score:.3f}")
    
    elif args.command == 'stats':
        stats = registry.get_stats(args.agent_type)
        if stats:
            print(f"\nStats for {args.agent_type}:\n")
            print(f"  Total versions: {stats.total_versions}")
            print(f"  Total examples: {stats.total_examples}")
            print(f"  Latest version: {stats.latest_version}")
            print(f"  Avg quality: {stats.avg_quality_score:.3f}")
            print(f"  Avg judge: {stats.avg_judge_score:.3f}")
        else:
            print(f"No datasets found for {args.agent_type}")
    
    elif args.command == 'compare':
        comparison = registry.compare_versions(
            agent_type=args.agent_type,
            version1=args.version1,
            version2=args.version2
        )
        print(f"\nComparison: {args.version1} vs {args.version2}\n")
        print(f"  Examples diff: {comparison['num_examples_diff']:+d}")
        print(f"  Quality diff: {comparison['quality_score_diff']:+.3f}")
        print(f"  Judge diff: {comparison['judge_score_diff']:+.3f}")
        print(f"  Time diff: {comparison['created_at_diff']:.0f}s")
        print(f"  File changed: {comparison['file_hash_changed']}")
    
    elif args.command == 'summary':
        summary = registry.get_summary()
        print(f"\nRegistry Summary:\n")
        print(f"  Total agents: {summary['total_agents']}")
        for agent_type, stats in summary['agents'].items():
            print(f"\n  {agent_type}:")
            print(f"    Versions: {stats['total_versions']}")
            print(f"    Examples: {stats['total_examples']}")
            print(f"    Latest: {stats['latest_version']}")
            print(f"    Avg quality: {stats['avg_quality_score']:.3f}")
