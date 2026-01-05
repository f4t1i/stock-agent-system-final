"""
Model Registry - Version tracking and management for trained SFT models

Purpose:
    Centralized registry for tracking all trained models with versioning,
    metadata, and performance metrics. Enables model comparison and regression testing.

Features:
    - Semantic versioning (1.0.0) with auto-increment
    - SQLite database for persistence
    - Model metadata tracking (hyperparams, dataset, git commit)
    - Performance metrics storage
    - Model comparison and regression testing
    - Model promotion (dev → staging → production)

Usage:
    registry = ModelRegistry()

    # Register new model
    model_id = registry.register_model(
        agent_name="news_agent",
        model_path="models/sft/news_agent_v1.0.0",
        version="1.0.0",
        metrics={"eval_loss": 0.45, "eval_accuracy": 0.85},
        metadata={"base_model": "mistral_7b", "lora_r": 16}
    )

    # Get best model for agent
    best_model = registry.get_best_model("news_agent", metric="eval_accuracy")

    # Promote to production
    registry.promote_model(model_id, stage="production")
"""

import sqlite3
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class ModelRecord:
    """Model registry record"""
    model_id: str
    agent_name: str
    version: str
    model_path: str

    # Metrics
    eval_loss: float
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None

    # Training info
    training_time_seconds: float = 0.0
    num_train_examples: int = 0
    num_eval_examples: int = 0

    # Model configuration
    base_model: str = ""
    lora_r: int = 0
    lora_alpha: int = 0

    # Dataset info
    dataset_version: str = ""
    dataset_path: str = ""

    # Status
    stage: str = "dev"  # dev, staging, production
    is_active: bool = True

    # Metadata
    git_commit: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    # Notes
    notes: str = ""


class ModelRegistry:
    """
    Model registry for version tracking and management

    Database Schema:
    - models: All registered models with metadata
    - metrics: Detailed metrics for each model
    - comparisons: Model comparison results
    """

    def __init__(self, registry_db: Optional[Path] = None):
        """
        Initialize registry

        Args:
            registry_db: Path to SQLite database (default: models/registry.db)
        """
        if registry_db is None:
            registry_db = Path("models/registry.db")

        self.registry_db = Path(registry_db)
        self.registry_db.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.registry_db))
        self._init_schema()

        logger.info(f"ModelRegistry initialized: {self.registry_db}")

    def _init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_path TEXT NOT NULL,

                -- Metrics
                eval_loss REAL NOT NULL,
                eval_accuracy REAL,
                eval_f1 REAL,

                -- Training info
                training_time_seconds REAL DEFAULT 0,
                num_train_examples INTEGER DEFAULT 0,
                num_eval_examples INTEGER DEFAULT 0,

                -- Model config
                base_model TEXT DEFAULT '',
                lora_r INTEGER DEFAULT 0,
                lora_alpha INTEGER DEFAULT 0,

                -- Dataset
                dataset_version TEXT DEFAULT '',
                dataset_path TEXT DEFAULT '',

                -- Status
                stage TEXT DEFAULT 'dev',
                is_active INTEGER DEFAULT 1,

                -- Metadata
                git_commit TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                -- Notes
                notes TEXT DEFAULT '',

                UNIQUE(agent_name, version)
            )
        """)

        # Metrics table (for additional metrics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                split TEXT DEFAULT 'eval',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Comparisons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_model_id TEXT NOT NULL,
                candidate_model_id TEXT NOT NULL,
                comparison_type TEXT DEFAULT 'regression_test',
                passed INTEGER DEFAULT 0,
                metrics_compared TEXT,
                results TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (baseline_model_id) REFERENCES models(model_id),
                FOREIGN KEY (candidate_model_id) REFERENCES models(model_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_name ON models(agent_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stage ON models(stage)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_active ON models(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_loss ON models(eval_loss)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)")

        self.conn.commit()
        logger.debug("Database schema initialized")

    def register_model(
        self,
        agent_name: str,
        model_path: Path,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new model

        Args:
            agent_name: Agent name (news_agent, technical_agent, fundamental_agent)
            model_path: Path to saved model
            version: Model version (e.g., "1.0.0")
            metrics: Performance metrics dict
            metadata: Model metadata (base_model, lora_r, etc.)
            dataset_info: Dataset information

        Returns:
            model_id: Unique model identifier
        """
        metadata = metadata or {}
        dataset_info = dataset_info or {}

        # Generate model ID
        model_id = f"{agent_name}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Get git commit
        git_commit = self._get_git_commit()

        # Create record
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO models (
                model_id, agent_name, version, model_path,
                eval_loss, eval_accuracy, eval_f1,
                training_time_seconds, num_train_examples, num_eval_examples,
                base_model, lora_r, lora_alpha,
                dataset_version, dataset_path,
                stage, git_commit, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            agent_name,
            version,
            str(model_path),
            metrics.get("eval_loss", 0.0),
            metrics.get("eval_accuracy"),
            metrics.get("eval_f1"),
            metadata.get("training_time_seconds", 0.0),
            metadata.get("num_train_examples", 0),
            metadata.get("num_eval_examples", 0),
            metadata.get("base_model", ""),
            metadata.get("lora_r", 0),
            metadata.get("lora_alpha", 0),
            dataset_info.get("dataset_version", ""),
            dataset_info.get("dataset_path", ""),
            "dev",  # Default stage
            git_commit,
            metadata.get("notes", "")
        ))

        # Store additional metrics
        for metric_name, metric_value in metrics.items():
            if metric_name not in ["eval_loss", "eval_accuracy", "eval_f1"]:
                cursor.execute("""
                    INSERT INTO metrics (model_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (model_id, metric_name, float(metric_value)))

        self.conn.commit()

        logger.info(f"Model registered: {model_id}")
        logger.info(f"  Agent: {agent_name}")
        logger.info(f"  Version: {version}")
        logger.info(f"  Eval Loss: {metrics.get('eval_loss', 0):.4f}")
        logger.info(f"  Path: {model_path}")

        return model_id

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Get model by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_models(
        self,
        agent_name: Optional[str] = None,
        stage: Optional[str] = None,
        is_active: bool = True,
        limit: Optional[int] = None
    ) -> List[ModelRecord]:
        """
        List models with filters

        Args:
            agent_name: Filter by agent
            stage: Filter by stage (dev, staging, production)
            is_active: Filter by active status
            limit: Maximum number of results

        Returns:
            List of ModelRecord
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)

        if stage:
            query += " AND stage = ?"
            params.append(stage)

        if is_active is not None:
            query += " AND is_active = ?"
            params.append(1 if is_active else 0)

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_best_model(
        self,
        agent_name: str,
        metric: str = "eval_loss",
        stage: Optional[str] = None,
        minimize: bool = True
    ) -> Optional[ModelRecord]:
        """
        Get best model for agent based on metric

        Args:
            agent_name: Agent name
            metric: Metric to optimize (eval_loss, eval_accuracy, eval_f1)
            stage: Filter by stage
            minimize: True to minimize metric, False to maximize

        Returns:
            Best ModelRecord or None
        """
        models = self.list_models(agent_name=agent_name, stage=stage, is_active=True)

        if not models:
            return None

        # Sort by metric
        if metric == "eval_loss":
            sorted_models = sorted(models, key=lambda m: m.eval_loss, reverse=not minimize)
        elif metric == "eval_accuracy":
            sorted_models = sorted(models, key=lambda m: m.eval_accuracy or 0, reverse=not minimize)
        elif metric == "eval_f1":
            sorted_models = sorted(models, key=lambda m: m.eval_f1 or 0, reverse=not minimize)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted_models[0] if sorted_models else None

    def promote_model(self, model_id: str, stage: str, notes: str = ""):
        """
        Promote model to new stage

        Args:
            model_id: Model ID
            stage: Target stage (staging, production)
            notes: Promotion notes
        """
        if stage not in ["staging", "production"]:
            raise ValueError(f"Invalid stage: {stage}. Must be 'staging' or 'production'")

        # Get model
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # If promoting to production, demote existing production models
        if stage == "production":
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE models
                SET stage = 'staging', updated_at = CURRENT_TIMESTAMP
                WHERE agent_name = ? AND stage = 'production'
            """, (model.agent_name,))

        # Promote model
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE models
            SET stage = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE model_id = ?
        """, (stage, notes, model_id))

        self.conn.commit()

        logger.info(f"Model promoted: {model_id} → {stage}")

    def compare_models(
        self,
        baseline_model_id: str,
        candidate_model_id: str,
        metrics: List[str],
        tolerance: float = 0.02
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare two models (regression testing)

        Args:
            baseline_model_id: Baseline model
            candidate_model_id: Candidate model
            metrics: Metrics to compare
            tolerance: Allowed degradation (e.g., 0.02 = 2%)

        Returns:
            (passed, comparison_results)
        """
        baseline = self.get_model(baseline_model_id)
        candidate = self.get_model(candidate_model_id)

        if not baseline or not candidate:
            raise ValueError("One or both models not found")

        results = {}
        passed = True

        for metric in metrics:
            baseline_value = getattr(baseline, metric, None)
            candidate_value = getattr(candidate, metric, None)

            if baseline_value is None or candidate_value is None:
                continue

            # For loss, lower is better
            if "loss" in metric:
                degradation = (candidate_value - baseline_value) / baseline_value
                metric_passed = degradation <= tolerance
            # For accuracy/f1, higher is better
            else:
                degradation = (baseline_value - candidate_value) / baseline_value
                metric_passed = degradation <= tolerance

            results[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "degradation": degradation,
                "passed": metric_passed
            }

            if not metric_passed:
                passed = False

        # Store comparison
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO comparisons (
                baseline_model_id, candidate_model_id, comparison_type,
                passed, metrics_compared, results
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            baseline_model_id,
            candidate_model_id,
            "regression_test",
            1 if passed else 0,
            json.dumps(metrics),
            json.dumps(results)
        ))
        self.conn.commit()

        logger.info(f"Model comparison: {baseline_model_id} vs {candidate_model_id}")
        logger.info(f"  Result: {'✅ PASSED' if passed else '❌ FAILED'}")

        return passed, results

    def get_version_history(self, agent_name: str) -> List[ModelRecord]:
        """Get version history for agent"""
        return self.list_models(agent_name=agent_name)

    def _row_to_record(self, row) -> ModelRecord:
        """Convert database row to ModelRecord"""
        return ModelRecord(
            model_id=row[0],
            agent_name=row[1],
            version=row[2],
            model_path=row[3],
            eval_loss=row[4],
            eval_accuracy=row[5],
            eval_f1=row[6],
            training_time_seconds=row[7],
            num_train_examples=row[8],
            num_eval_examples=row[9],
            base_model=row[10],
            lora_r=row[11],
            lora_alpha=row[12],
            dataset_version=row[13],
            dataset_path=row[14],
            stage=row[15],
            is_active=bool(row[16]),
            git_commit=row[17],
            created_at=row[18],
            updated_at=row[19],
            notes=row[20]
        )

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None

    def close(self):
        """Close database connection"""
        self.conn.close()


# CLI Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Registry - List and manage trained models")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--agent", type=str, help="Filter by agent")
    parser.add_argument("--stage", choices=["dev", "staging", "production"], help="Filter by stage")
    parser.add_argument("--best", action="store_true", help="Show best model for agent")
    parser.add_argument("--metric", default="eval_loss", help="Metric for best model")
    parser.add_argument("--promote", type=str, help="Model ID to promote")
    parser.add_argument("--to-stage", choices=["staging", "production"], help="Target stage for promotion")

    args = parser.parse_args()

    registry = ModelRegistry()

    if args.list:
        models = registry.list_models(agent_name=args.agent, stage=args.stage)
        print(f"\n{'='*80}")
        print("MODEL REGISTRY")
        print(f"{'='*80}")
        print(f"Total models: {len(models)}")

        for model in models:
            print(f"\n{model.model_id}")
            print(f"  Agent: {model.agent_name}")
            print(f"  Version: {model.version}")
            print(f"  Stage: {model.stage}")
            print(f"  Eval Loss: {model.eval_loss:.4f}")
            if model.eval_accuracy:
                print(f"  Eval Accuracy: {model.eval_accuracy:.4f}")
            print(f"  Created: {model.created_at}")

    elif args.best and args.agent:
        model = registry.get_best_model(args.agent, metric=args.metric)
        if model:
            print(f"\n{'='*80}")
            print(f"BEST MODEL FOR {args.agent.upper()} (by {args.metric})")
            print(f"{'='*80}")
            print(f"Model ID: {model.model_id}")
            print(f"Version: {model.version}")
            print(f"Stage: {model.stage}")
            print(f"Eval Loss: {model.eval_loss:.4f}")
            if model.eval_accuracy:
                print(f"Eval Accuracy: {model.eval_accuracy:.4f}")
            print(f"Path: {model.model_path}")
            print(f"Created: {model.created_at}")
        else:
            print(f"No models found for {args.agent}")

    elif args.promote and args.to_stage:
        registry.promote_model(args.promote, args.to_stage)
        print(f"✅ Model {args.promote} promoted to {args.to_stage}")

    else:
        parser.print_help()

    registry.close()
