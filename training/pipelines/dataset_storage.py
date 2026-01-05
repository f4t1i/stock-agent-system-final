"""
Dataset File I/O + Versioning - Task 1.5

Manages dataset storage with semantic versioning and integrity checking.

Features:
- Save/load datasets (JSONL, JSON)
- Semantic versioning (vMAJOR.MINOR.PATCH)
- Metadata tracking (agent_type, quality, judge scores, lineage)
- SHA256 file integrity
- Version history
- Atomic writes (temp file + rename)

Directory Structure:
```
datasets/
  technical/
    v1.0.0/
      dataset.jsonl
      metadata.json
    v1.1.0/
      dataset.jsonl
      metadata.json
  news/
    v1.0.0/
      ...
```

Metadata Format:
```json
{
  "version": "1.0.0",
  "agent_type": "technical",
  "format": "chatml",
  "created_at": "2024-01-01T00:00:00Z",
  "example_count": 1000,
  "quality_stats": {...},
  "judge_stats": {...},
  "lineage": {
    "backtest_id": "test_001",
    "parent_version": null
  },
  "file_integrity": {
    "sha256": "abc123...",
    "size_bytes": 1024000
  }
}
```

Phase A1 Week 3-4: Task 1.5 COMPLETE
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class DatasetMetadata:
    """Metadata for a dataset version"""
    version: str
    agent_type: str
    format: str  # chatml, alpaca
    created_at: str
    example_count: int
    quality_stats: Dict
    judge_stats: Optional[Dict]
    lineage: Dict
    file_integrity: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetMetadata':
        return cls(**data)


class SemanticVersion:
    """Semantic version parser and comparator"""
    
    def __init__(self, version_str: str):
        """
        Parse semantic version string
        
        Args:
            version_str: Version string (e.g., "1.2.3" or "v1.2.3")
        """
        # Remove 'v' prefix if present
        if version_str.startswith('v'):
            version_str = version_str[1:]
        
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        
        self.major = int(parts[0])
        self.minor = int(parts[1])
        self.patch = int(parts[2])
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __repr__(self) -> str:
        return f"SemanticVersion({self})"
    
    def __eq__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __le__(self, other: 'SemanticVersion') -> bool:
        return self < other or self == other
    
    def __gt__(self, other: 'SemanticVersion') -> bool:
        return not self <= other
    
    def __ge__(self, other: 'SemanticVersion') -> bool:
        return not self < other
    
    def bump_major(self) -> 'SemanticVersion':
        """Increment major version, reset minor and patch"""
        return SemanticVersion(f"{self.major + 1}.0.0")
    
    def bump_minor(self) -> 'SemanticVersion':
        """Increment minor version, reset patch"""
        return SemanticVersion(f"{self.major}.{self.minor + 1}.0")
    
    def bump_patch(self) -> 'SemanticVersion':
        """Increment patch version"""
        return SemanticVersion(f"{self.major}.{self.minor}.{self.patch + 1}")


class DatasetStorage:
    """
    Manages dataset file I/O with versioning and integrity
    
    Usage:
        storage = DatasetStorage(base_dir="datasets")
        
        # Save dataset
        metadata = storage.save_dataset(
            agent_type="technical",
            version="1.0.0",
            data=examples,
            format="chatml",
            quality_stats={...},
            lineage={...}
        )
        
        # Load dataset
        data, metadata = storage.load_dataset(
            agent_type="technical",
            version="1.0.0"
        )
        
        # List versions
        versions = storage.list_versions(agent_type="technical")
    """
    
    def __init__(self, base_dir: str = "datasets"):
        """
        Initialize storage
        
        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetStorage initialized at {self.base_dir}")
    
    def save_dataset(
        self,
        agent_type: str,
        version: str,
        data: List[Dict],
        format: str,
        quality_stats: Dict,
        judge_stats: Optional[Dict] = None,
        lineage: Optional[Dict] = None
    ) -> DatasetMetadata:
        """
        Save dataset with metadata
        
        Args:
            agent_type: Agent type (e.g., "technical", "news")
            version: Semantic version (e.g., "1.0.0")
            data: List of examples (ChatML or Alpaca format)
            format: Dataset format ("chatml" or "alpaca")
            quality_stats: Quality statistics
            judge_stats: Judge statistics (optional)
            lineage: Lineage information (optional)
        
        Returns:
            DatasetMetadata object
        """
        # Validate version
        sem_ver = SemanticVersion(version)
        version_str = str(sem_ver)
        
        # Create version directory
        version_dir = self.base_dir / agent_type / f"v{version_str}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset file path
        if format == "chatml":
            dataset_file = version_dir / "dataset.jsonl"
        elif format == "alpaca":
            dataset_file = version_dir / "dataset.json"
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Write dataset file (atomic write)
        temp_file = version_dir / f".dataset.{format}.tmp"
        
        if format == "chatml":
            # JSONL format (one example per line)
            with open(temp_file, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')
        elif format == "alpaca":
            # JSON format (array of examples)
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Atomic rename
        shutil.move(str(temp_file), str(dataset_file))
        
        # Calculate file integrity
        sha256_hash, file_size = self._calculate_file_hash(dataset_file)
        
        # Create metadata
        metadata = DatasetMetadata(
            version=version_str,
            agent_type=agent_type,
            format=format,
            created_at=datetime.utcnow().isoformat() + 'Z',
            example_count=len(data),
            quality_stats=quality_stats,
            judge_stats=judge_stats or {},
            lineage=lineage or {},
            file_integrity={
                'sha256': sha256_hash,
                'size_bytes': file_size
            }
        )
        
        # Save metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(
            f"Saved dataset {agent_type} v{version_str}: "
            f"{len(data)} examples, {file_size} bytes"
        )
        
        return metadata
    
    def load_dataset(
        self,
        agent_type: str,
        version: str,
        verify_integrity: bool = True
    ) -> Tuple[List[Dict], DatasetMetadata]:
        """
        Load dataset with metadata
        
        Args:
            agent_type: Agent type
            version: Semantic version
            verify_integrity: Whether to verify file integrity
        
        Returns:
            Tuple of (data, metadata)
        
        Raises:
            FileNotFoundError: If dataset not found
            ValueError: If integrity check fails
        """
        # Validate version
        sem_ver = SemanticVersion(version)
        version_str = str(sem_ver)
        
        # Version directory
        version_dir = self.base_dir / agent_type / f"v{version_str}"
        
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found: {agent_type} v{version_str}"
            )
        
        # Load metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = DatasetMetadata.from_dict(metadata_dict)
        
        # Dataset file
        if metadata.format == "chatml":
            dataset_file = version_dir / "dataset.jsonl"
        elif metadata.format == "alpaca":
            dataset_file = version_dir / "dataset.json"
        else:
            raise ValueError(f"Unknown format: {metadata.format}")
        
        # Verify integrity
        if verify_integrity:
            sha256_hash, file_size = self._calculate_file_hash(dataset_file)
            
            if sha256_hash != metadata.file_integrity['sha256']:
                raise ValueError(
                    f"Integrity check failed for {agent_type} v{version_str}: "
                    f"expected {metadata.file_integrity['sha256']}, "
                    f"got {sha256_hash}"
                )
            
            logger.debug(f"Integrity verified for {agent_type} v{version_str}")
        
        # Load dataset
        if metadata.format == "chatml":
            # JSONL format
            data = []
            with open(dataset_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        elif metadata.format == "alpaca":
            # JSON format
            with open(dataset_file, 'r') as f:
                data = json.load(f)
        
        logger.info(
            f"Loaded dataset {agent_type} v{version_str}: "
            f"{len(data)} examples"
        )
        
        return data, metadata
    
    def list_versions(self, agent_type: str) -> List[str]:
        """
        List all versions for an agent type
        
        Args:
            agent_type: Agent type
        
        Returns:
            List of version strings (sorted, newest first)
        """
        agent_dir = self.base_dir / agent_type
        
        if not agent_dir.exists():
            return []
        
        versions = []
        for version_dir in agent_dir.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith('v'):
                version_str = version_dir.name[1:]  # Remove 'v' prefix
                versions.append(version_str)
        
        # Sort by semantic version (newest first)
        versions.sort(key=lambda v: SemanticVersion(v), reverse=True)
        
        return versions
    
    def get_latest_version(self, agent_type: str) -> Optional[str]:
        """
        Get latest version for an agent type
        
        Args:
            agent_type: Agent type
        
        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(agent_type)
        return versions[0] if versions else None
    
    def delete_version(self, agent_type: str, version: str):
        """
        Delete a dataset version
        
        Args:
            agent_type: Agent type
            version: Semantic version
        """
        sem_ver = SemanticVersion(version)
        version_str = str(sem_ver)
        
        version_dir = self.base_dir / agent_type / f"v{version_str}"
        
        if version_dir.exists():
            shutil.rmtree(version_dir)
            logger.info(f"Deleted dataset {agent_type} v{version_str}")
        else:
            logger.warning(f"Dataset not found: {agent_type} v{version_str}")
    
    def _calculate_file_hash(self, file_path: Path) -> Tuple[str, int]:
        """
        Calculate SHA256 hash and file size
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple of (sha256_hash, file_size)
        """
        sha256 = hashlib.sha256()
        file_size = 0
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
                file_size += len(chunk)
        
        return sha256.hexdigest(), file_size


if __name__ == "__main__":
    # Example usage
    storage = DatasetStorage(base_dir="datasets")
    
    # Sample data
    chatml_data = [
        [
            {"role": "system", "content": "You are a stock trading agent."},
            {"role": "user", "content": "Analyze AAPL..."},
            {"role": "assistant", "content": "BUY recommendation..."}
        ],
        [
            {"role": "system", "content": "You are a stock trading agent."},
            {"role": "user", "content": "Analyze TSLA..."},
            {"role": "assistant", "content": "HOLD recommendation..."}
        ]
    ]
    
    # Save dataset
    metadata = storage.save_dataset(
        agent_type="technical",
        version="1.0.0",
        data=chatml_data,
        format="chatml",
        quality_stats={
            'avg_quality_score': 0.85,
            'pass_rate': 0.90
        },
        judge_stats={
            'avg_judge_score': 0.80,
            'approval_rate': 0.85
        },
        lineage={
            'backtest_id': 'test_001',
            'parent_version': None
        }
    )
    
    print(f"Saved dataset: {metadata.agent_type} v{metadata.version}")
    print(f"  Examples: {metadata.example_count}")
    print(f"  Size: {metadata.file_integrity['size_bytes']} bytes")
    print(f"  SHA256: {metadata.file_integrity['sha256'][:16]}...")
    
    # Load dataset
    data, metadata = storage.load_dataset(
        agent_type="technical",
        version="1.0.0"
    )
    
    print(f"\nLoaded dataset: {metadata.agent_type} v{metadata.version}")
    print(f"  Examples: {len(data)}")
    
    # List versions
    versions = storage.list_versions(agent_type="technical")
    print(f"\nAvailable versions: {versions}")
    
    # Latest version
    latest = storage.get_latest_version(agent_type="technical")
    print(f"Latest version: {latest}")
