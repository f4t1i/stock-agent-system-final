"""
Experience Store - Captures backtest results for training dataset synthesis

Purpose:
    Store (signal, action, outcome, reward) tuples from backtests for later conversion
    into SFT/RL training datasets.

Features:
    - Multi-format storage (JSON, SQLite, Parquet)
    - Metadata tracking (timestamp, symbol, backtest_id)
    - Query and filter capabilities
    - Dataset versioning
    - Judge-approved filtering

Usage:
    store = ExperienceStore(storage_dir="data/experiences")

    # Store experience from backtest
    store.add_experience(
        signal=signal_dict,
        action={"decision": "buy", "position_size": 0.10},
        outcome={"pnl": 1250.50, "return_pct": 0.05},
        reward=0.85,
        metadata={"symbol": "AAPL", "backtest_id": "bt_001"}
    )

    # Query experiences
    experiences = store.query(symbol="AAPL", min_reward=0.5)

    # Export for dataset synthesis
    store.export("output.jsonl", format="jsonl")
"""

import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from loguru import logger
import hashlib


StorageFormat = Literal["json", "jsonl", "sqlite", "parquet"]


@dataclass
class Experience:
    """Single experience tuple from backtest"""

    # Core components
    signal: Dict[str, Any]              # Full signal contract
    action: Dict[str, Any]              # Strategist decision
    outcome: Dict[str, Any]             # Trade outcome (P&L, return, duration)
    reward: float                       # Normalized reward [-1, 1]

    # Metadata
    experience_id: str                  # Unique ID (hash of signal + timestamp)
    symbol: str
    timestamp: str                      # ISO 8601
    backtest_id: Optional[str] = None
    judge_approved: bool = False        # Passed judge validation
    judge_score: Optional[float] = None # LLM judge score (0-10)

    # Contextual data
    market_regime: Optional[str] = None # "bull", "bear", "neutral"
    volatility_regime: Optional[str] = None  # "low", "medium", "high"

    # Versioning
    schema_version: str = "1.0.0"

    @staticmethod
    def generate_id(signal: Dict, timestamp: str) -> str:
        """Generate unique experience ID from signal + timestamp"""
        content = json.dumps(signal, sort_keys=True) + timestamp
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ExperienceStoreConfig:
    """Configuration for experience store"""
    storage_dir: Path = Path("data/experiences")
    storage_format: StorageFormat = "jsonl"
    auto_backup: bool = True
    backup_interval: int = 100  # Backup every N experiences
    enable_sqlite_index: bool = True
    max_experiences_per_file: int = 10000


class ExperienceStore:
    """
    Storage system for backtest experiences

    Supports multiple storage backends:
    - JSON/JSONL: Human-readable, good for small datasets
    - SQLite: Queryable, good for medium datasets (up to 1M records)
    - Parquet: Efficient, good for large datasets (1M+ records)
    """

    def __init__(self, config: Optional[ExperienceStoreConfig] = None):
        self.config = config or ExperienceStoreConfig()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)

        self.experiences: List[Experience] = []
        self.db_conn: Optional[sqlite3.Connection] = None

        # Initialize storage backend
        self._init_storage()

        logger.info(f"ExperienceStore initialized: {self.config.storage_dir}")
        logger.info(f"Storage format: {self.config.storage_format}")

    def _init_storage(self):
        """Initialize storage backend"""
        if self.config.storage_format == "sqlite":
            db_path = self.config.storage_dir / "experiences.db"
            self.db_conn = sqlite3.connect(str(db_path))
            self._create_sqlite_schema()
            logger.info(f"SQLite database initialized: {db_path}")

    def _create_sqlite_schema(self):
        """Create SQLite schema with indexes"""
        if not self.db_conn:
            return

        cursor = self.db_conn.cursor()

        # Create experiences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                backtest_id TEXT,
                reward REAL NOT NULL,
                judge_approved INTEGER DEFAULT 0,
                judge_score REAL,
                market_regime TEXT,
                volatility_regime TEXT,
                signal TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                schema_version TEXT DEFAULT '1.0.0',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        if self.config.enable_sqlite_index:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON experiences(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reward ON experiences(reward)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_judge_approved ON experiences(judge_approved)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_id ON experiences(backtest_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp)")

        self.db_conn.commit()
        logger.debug("SQLite schema created with indexes")

    def add_experience(
        self,
        signal: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add single experience to store

        Args:
            signal: Full signal contract (analysis, signal, sizing, risk, rationale, evidence)
            action: Strategist action (decision, position_size, entry, stop_loss, take_profit)
            outcome: Trade outcome (pnl, return_pct, duration_days, exit_reason)
            reward: Normalized reward score [-1, 1]
            metadata: Optional metadata (symbol, backtest_id, judge_approved, etc.)

        Returns:
            experience_id: Unique identifier for this experience
        """
        metadata = metadata or {}

        # Generate experience ID
        timestamp = datetime.now().isoformat()
        experience_id = Experience.generate_id(signal, timestamp)

        # Create experience object
        experience = Experience(
            experience_id=experience_id,
            signal=signal,
            action=action,
            outcome=outcome,
            reward=reward,
            symbol=metadata.get("symbol", "UNKNOWN"),
            timestamp=timestamp,
            backtest_id=metadata.get("backtest_id"),
            judge_approved=metadata.get("judge_approved", False),
            judge_score=metadata.get("judge_score"),
            market_regime=metadata.get("market_regime"),
            volatility_regime=metadata.get("volatility_regime")
        )

        # Store based on format
        if self.config.storage_format == "sqlite":
            self._store_sqlite(experience)
        else:
            self.experiences.append(experience)

            # Auto-backup if threshold reached
            if self.config.auto_backup and len(self.experiences) % self.config.backup_interval == 0:
                self._backup()

        logger.debug(f"Experience added: {experience_id} (symbol={experience.symbol}, reward={reward:.3f})")
        return experience_id

    def _store_sqlite(self, experience: Experience):
        """Store experience in SQLite"""
        if not self.db_conn:
            raise RuntimeError("SQLite connection not initialized")

        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO experiences (
                experience_id, symbol, timestamp, backtest_id, reward,
                judge_approved, judge_score, market_regime, volatility_regime,
                signal, action, outcome, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experience.experience_id,
            experience.symbol,
            experience.timestamp,
            experience.backtest_id,
            experience.reward,
            1 if experience.judge_approved else 0,
            experience.judge_score,
            experience.market_regime,
            experience.volatility_regime,
            json.dumps(experience.signal),
            json.dumps(experience.action),
            json.dumps(experience.outcome),
            experience.schema_version
        ))
        self.db_conn.commit()

    def add_batch(self, experiences: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple experiences in batch

        Args:
            experiences: List of experience dicts with keys: signal, action, outcome, reward, metadata

        Returns:
            List of experience IDs
        """
        ids = []
        for exp in experiences:
            exp_id = self.add_experience(
                signal=exp["signal"],
                action=exp["action"],
                outcome=exp["outcome"],
                reward=exp["reward"],
                metadata=exp.get("metadata", {})
            )
            ids.append(exp_id)

        logger.info(f"Batch added: {len(ids)} experiences")
        return ids

    def query(
        self,
        symbol: Optional[str] = None,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
        judge_approved_only: bool = False,
        min_judge_score: Optional[float] = None,
        backtest_id: Optional[str] = None,
        market_regime: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Experience]:
        """
        Query experiences with filters

        Args:
            symbol: Filter by symbol (e.g., "AAPL")
            min_reward: Minimum reward threshold
            max_reward: Maximum reward threshold
            judge_approved_only: Only return judge-approved experiences
            min_judge_score: Minimum judge score (0-10)
            backtest_id: Filter by backtest ID
            market_regime: Filter by market regime
            limit: Maximum number of results

        Returns:
            List of matching experiences
        """
        if self.config.storage_format == "sqlite":
            return self._query_sqlite(
                symbol=symbol,
                min_reward=min_reward,
                max_reward=max_reward,
                judge_approved_only=judge_approved_only,
                min_judge_score=min_judge_score,
                backtest_id=backtest_id,
                market_regime=market_regime,
                limit=limit
            )
        else:
            return self._query_memory(
                symbol=symbol,
                min_reward=min_reward,
                max_reward=max_reward,
                judge_approved_only=judge_approved_only,
                min_judge_score=min_judge_score,
                backtest_id=backtest_id,
                market_regime=market_regime,
                limit=limit
            )

    def _query_sqlite(self, **filters) -> List[Experience]:
        """Query SQLite database"""
        if not self.db_conn:
            raise RuntimeError("SQLite connection not initialized")

        # Build query
        conditions = []
        params = []

        if filters.get("symbol"):
            conditions.append("symbol = ?")
            params.append(filters["symbol"])

        if filters.get("min_reward") is not None:
            conditions.append("reward >= ?")
            params.append(filters["min_reward"])

        if filters.get("max_reward") is not None:
            conditions.append("reward <= ?")
            params.append(filters["max_reward"])

        if filters.get("judge_approved_only"):
            conditions.append("judge_approved = 1")

        if filters.get("min_judge_score") is not None:
            conditions.append("judge_score >= ?")
            params.append(filters["min_judge_score"])

        if filters.get("backtest_id"):
            conditions.append("backtest_id = ?")
            params.append(filters["backtest_id"])

        if filters.get("market_regime"):
            conditions.append("market_regime = ?")
            params.append(filters["market_regime"])

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM experiences WHERE {where_clause} ORDER BY timestamp DESC"

        if filters.get("limit"):
            query += f" LIMIT {filters['limit']}"

        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert rows to Experience objects
        experiences = []
        for row in rows:
            exp = Experience(
                experience_id=row[0],
                symbol=row[1],
                timestamp=row[2],
                backtest_id=row[3],
                reward=row[4],
                judge_approved=bool(row[5]),
                judge_score=row[6],
                market_regime=row[7],
                volatility_regime=row[8],
                signal=json.loads(row[9]),
                action=json.loads(row[10]),
                outcome=json.loads(row[11]),
                schema_version=row[12]
            )
            experiences.append(exp)

        return experiences

    def _query_memory(self, **filters) -> List[Experience]:
        """Query in-memory experiences"""
        results = self.experiences

        if filters.get("symbol"):
            results = [e for e in results if e.symbol == filters["symbol"]]

        if filters.get("min_reward") is not None:
            results = [e for e in results if e.reward >= filters["min_reward"]]

        if filters.get("max_reward") is not None:
            results = [e for e in results if e.reward <= filters["max_reward"]]

        if filters.get("judge_approved_only"):
            results = [e for e in results if e.judge_approved]

        if filters.get("min_judge_score") is not None:
            results = [e for e in results if e.judge_score and e.judge_score >= filters["min_judge_score"]]

        if filters.get("backtest_id"):
            results = [e for e in results if e.backtest_id == filters["backtest_id"]]

        if filters.get("market_regime"):
            results = [e for e in results if e.market_regime == filters["market_regime"]]

        if filters.get("limit"):
            results = results[:filters["limit"]]

        return results

    def export(self, output_path: Path, format: Literal["json", "jsonl"] = "jsonl"):
        """
        Export experiences to file

        Args:
            output_path: Output file path
            format: Export format ("json" or "jsonl")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all experiences
        if self.config.storage_format == "sqlite":
            experiences = self.query()
        else:
            experiences = self.experiences

        # Export
        if format == "jsonl":
            with open(output_path, "w") as f:
                for exp in experiences:
                    f.write(json.dumps(asdict(exp)) + "\n")
        else:  # json
            with open(output_path, "w") as f:
                json.dump([asdict(exp) for exp in experiences], f, indent=2)

        logger.info(f"Exported {len(experiences)} experiences to {output_path}")

    def _backup(self):
        """Backup in-memory experiences to disk"""
        if not self.experiences:
            return

        backup_path = self.config.storage_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.export(backup_path, format="jsonl")
        logger.debug(f"Backup created: {backup_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        if self.config.storage_format == "sqlite":
            if not self.db_conn:
                return {}

            cursor = self.db_conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM experiences")
            total = cursor.fetchone()[0]

            # Judge approved count
            cursor.execute("SELECT COUNT(*) FROM experiences WHERE judge_approved = 1")
            approved = cursor.fetchone()[0]

            # Average reward
            cursor.execute("SELECT AVG(reward) FROM experiences")
            avg_reward = cursor.fetchone()[0] or 0.0

            # Symbol distribution
            cursor.execute("SELECT symbol, COUNT(*) FROM experiences GROUP BY symbol ORDER BY COUNT(*) DESC LIMIT 10")
            symbol_dist = dict(cursor.fetchall())

            return {
                "total_experiences": total,
                "judge_approved": approved,
                "approval_rate": approved / total if total > 0 else 0.0,
                "avg_reward": avg_reward,
                "top_symbols": symbol_dist
            }
        else:
            total = len(self.experiences)
            approved = sum(1 for e in self.experiences if e.judge_approved)
            avg_reward = sum(e.reward for e in self.experiences) / total if total > 0 else 0.0

            # Symbol distribution
            from collections import Counter
            symbol_counts = Counter(e.symbol for e in self.experiences)

            return {
                "total_experiences": total,
                "judge_approved": approved,
                "approval_rate": approved / total if total > 0 else 0.0,
                "avg_reward": avg_reward,
                "top_symbols": dict(symbol_counts.most_common(10))
            }

    def close(self):
        """Close storage connections"""
        if self.db_conn:
            self.db_conn.close()
            logger.debug("SQLite connection closed")


# CLI Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experience Store - Query and export backtest experiences")
    parser.add_argument("--storage-dir", type=Path, default=Path("data/experiences"), help="Storage directory")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--query", action="store_true", help="Query experiences")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--min-reward", type=float, help="Minimum reward")
    parser.add_argument("--judge-approved-only", action="store_true", help="Only judge-approved")
    parser.add_argument("--export", type=Path, help="Export to file")
    parser.add_argument("--format", choices=["json", "jsonl"], default="jsonl", help="Export format")

    args = parser.parse_args()

    # Initialize store
    config = ExperienceStoreConfig(storage_dir=args.storage_dir)
    store = ExperienceStore(config)

    # Show statistics
    if args.stats:
        stats = store.get_statistics()
        print("\n=== Experience Store Statistics ===")
        print(f"Total experiences: {stats['total_experiences']}")
        print(f"Judge approved: {stats['judge_approved']} ({stats['approval_rate']:.1%})")
        print(f"Average reward: {stats['avg_reward']:.3f}")
        print("\nTop symbols:")
        for symbol, count in stats['top_symbols'].items():
            print(f"  {symbol}: {count}")

    # Query experiences
    if args.query:
        results = store.query(
            symbol=args.symbol,
            min_reward=args.min_reward,
            judge_approved_only=args.judge_approved_only
        )
        print(f"\n=== Query Results: {len(results)} experiences ===")
        for exp in results[:5]:  # Show first 5
            print(f"\n{exp.experience_id}")
            print(f"  Symbol: {exp.symbol}")
            print(f"  Reward: {exp.reward:.3f}")
            print(f"  Judge approved: {exp.judge_approved}")
            print(f"  Action: {exp.action.get('decision', 'N/A')}")

    # Export
    if args.export:
        store.export(args.export, format=args.format)
        print(f"Exported to {args.export}")

    store.close()
