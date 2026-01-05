# Dataset Registry System - Task 2.1

Complete Postgres-based dataset registry for managing SFT training datasets.

## Overview

The Dataset Registry System provides:
- **Semantic Versioning** - MAJOR.MINOR.PATCH version management
- **SHA256 Integrity** - File integrity verification
- **Lineage Tracking** - Parent-child relationships and provenance
- **Quality Metrics** - Per-example quality and judge scores
- **Full-text Search** - Search datasets by name/description
- **Transaction Support** - ACID compliance for data consistency

## Database Schema

### Tables

1. **`datasets`** - Main dataset metadata
   - Agent type (technical, news, fundamental, etc.)
   - Current version tracking
   - Status (active, archived, deprecated)
   - Soft delete support

2. **`dataset_versions`** - Version history
   - Semantic versioning (MAJOR.MINOR.PATCH)
   - File path and SHA256 hash
   - Example count and token statistics
   - Quality score summaries
   - Judge evaluation summaries
   - Tags for categorization

3. **`dataset_lineage`** - Provenance tracking
   - Parent-child relationships
   - Backtest source tracking
   - Transformation type (filter, merge, augment, retrain)
   - Example count changes (inherited, new, modified, removed)

4. **`dataset_quality_metrics`** - Per-example metrics
   - Quality scores (reward, confidence, reasoning, consistency)
   - Judge evaluation (score, passed, feedback)
   - Trajectory ID linking
   - Symbol and agent type

### Views

- **`v_latest_datasets`** - Latest version of all active datasets
- **`v_dataset_lineage_tree`** - Complete lineage tree with version info
- **`v_quality_metrics_summary`** - Aggregated quality metrics by version

### Triggers

- **`trigger_datasets_updated_at`** - Auto-update `updated_at` timestamp
- **`trigger_update_dataset_stats`** - Auto-update dataset stats when new version added

### Functions

- **`update_updated_at_column()`** - Update timestamp trigger function
- **`update_dataset_stats()`** - Update dataset statistics trigger function
- **`calculate_quality_summary(version_id)`** - Calculate quality metrics summary

## Files

- **`dataset_registry_schema.sql`** (450 lines) - Complete SQL schema
- **`dataset_registry_db.py`** (814 lines) - Python database interface

## Python Interface

### Connection Management

```python
from training.registry.dataset_registry_db import get_db_connection

# Get connection
db = get_db_connection(
    host="localhost",
    port=5432,
    database="stock_agent",
    user="postgres",
    password="your_password"
)

# Use context manager
with db:
    # Check schema version
    version = db.check_schema_version()
    print(f"Schema version: {version}")
```

### Dataset Operations

```python
# Create dataset
dataset_id = db.create_dataset(
    agent_type="technical",
    name="Technical Analysis Dataset",
    description="Dataset for technical analysis agent",
    format="chatml"
)

# Get dataset
dataset = db.get_dataset(dataset_id)
dataset = db.get_dataset_by_agent_type("technical")

# List datasets
datasets = db.list_datasets(status="active", format="chatml")

# Update status
db.update_dataset_status(dataset_id, "archived")

# Delete dataset
db.delete_dataset(dataset_id, soft=True)
```

### Version Operations

```python
# Create version
version_id = db.create_version(
    dataset_id=dataset_id,
    version="1.0.0",
    file_path="/path/to/dataset.jsonl",
    file_size_bytes=1024000,
    sha256_hash="abc123...",
    example_count=1000,
    avg_quality_score=0.75,
    description="Initial release",
    tags=["production", "validated"]
)

# Get version
version = db.get_version(version_id)
latest = db.get_latest_version(dataset_id)

# List versions
versions = db.list_versions(dataset_id, limit=10)
```

### Lineage Operations

```python
# Create lineage
lineage_id = db.create_lineage(
    child_version_id=new_version_id,
    parent_version_id=old_version_id,
    backtest_id="test_001",
    transformation_type="judge_filter",
    examples_inherited=800,
    examples_new=200,
    notes="Filtered by judge threshold 0.7"
)

# Get lineage
lineage = db.get_lineage(version_id)
```

### Quality Metrics Operations

```python
# Batch insert metrics
metrics = [
    {
        'example_index': 0,
        'quality_score': 0.85,
        'reward_score': 0.9,
        'confidence_score': 0.8,
        'trajectory_id': 'traj_001',
        'symbol': 'AAPL',
        'agent_type': 'technical'
    },
    # ... more metrics
]

db.create_quality_metrics(version_id, metrics)

# Get metrics
metrics = db.get_quality_metrics(version_id, min_quality_score=0.7)

# Update judge scores
db.update_judge_scores(
    version_id=version_id,
    example_index=0,
    judge_score=0.9,
    judge_passed=True,
    judge_feedback="Excellent reasoning"
)
```

## Schema Initialization

```bash
# Using Python
python3 -c "
from training.registry.dataset_registry_db import get_db_connection

db = get_db_connection(host='localhost', database='stock_agent')
with db:
    db.init_schema()
    print('Schema initialized')
"

# Or directly with psql
psql -U postgres -d stock_agent -f training/registry/dataset_registry_schema.sql
```

## Features

### Semantic Versioning
- Enforced MAJOR.MINOR.PATCH format
- Automatic parsing into separate fields
- Version ordering support

### SHA256 Integrity
- File hash stored with each version
- Integrity verification support
- Indexed for fast lookups

### Lineage Tracking
- Parent-child relationships
- Backtest source tracking
- Transformation type categorization
- Example count changes tracked

### Quality Metrics
- Per-example quality scores
- Judge evaluation integration
- Aggregated summaries
- Filtering by score thresholds

### Full-text Search
- Trigram-based search on dataset names
- Fast partial matching
- Case-insensitive

### Soft Delete
- Datasets marked as deleted, not removed
- Preserves referential integrity
- Can be restored if needed

## Performance

- **Indexes** on all foreign keys and frequently queried columns
- **Triggers** for automatic updates (minimal overhead)
- **Views** for common complex queries
- **Batch operations** for quality metrics insertion

## Next Steps

- Task 2.2: Semantic Versioning Logic
- Task 2.3: SHA256 Integrity Checking
- Task 2.4: Lineage Tracking Implementation
- Task 2.5: Search/Filter Queries
- Task 2.6: Registry CLI Interface

## Phase A1 Week 3-4: Task 2.1 COMPLETE
