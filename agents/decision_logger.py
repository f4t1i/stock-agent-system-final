#!/usr/bin/env python3
"""
Decision Logger - Log and Retrieve Agent Decisions

Logs all agent decisions with metadata for explainability and audit trail.
Uses SQLite for persistent storage.

Usage:
    logger = DecisionLogger()
    decision_id = logger.log_decision(symbol, agent_name, agent_output)
    decision = logger.get_decision(decision_id)
    recent = logger.list_recent(limit=10)
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
from contextlib import contextmanager


class DecisionLogger:
    """Log and retrieve agent decisions"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize decision logger
        
        Args:
            db_path: Path to SQLite database (default: data/decisions.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "decisions.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create database schema if not exists"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    agent_output TEXT NOT NULL,
                    context TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    factor_name TEXT NOT NULL,
                    importance REAL NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol 
                ON decisions(symbol)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_agent 
                ON decisions(agent_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_timestamp 
                ON decisions(timestamp DESC)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_decision(
        self,
        symbol: str,
        agent_name: str,
        agent_output: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an agent decision
        
        Args:
            symbol: Stock symbol
            agent_name: Name of the agent
            agent_output: Complete agent output dictionary
            context: Additional context (optional)
            
        Returns:
            decision_id: Unique decision identifier
        """
        try:
            # Generate decision ID
            decision_id = str(uuid.uuid4())
            
            # Extract key fields
            recommendation = agent_output.get("recommendation", "HOLD")
            confidence = agent_output.get("confidence", 0.5)
            reasoning = agent_output.get("reasoning", "")
            
            # Serialize complex fields
            agent_output_json = json.dumps(agent_output)
            context_json = json.dumps(context) if context else None
            metadata_json = json.dumps({
                "logged_at": datetime.now().isoformat(),
                "version": "1.0"
            })
            
            # Insert decision
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO decisions (
                        decision_id, symbol, agent_name, recommendation,
                        confidence, reasoning, agent_output, context,
                        timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision_id,
                    symbol,
                    agent_name,
                    recommendation,
                    confidence,
                    reasoning,
                    agent_output_json,
                    context_json,
                    datetime.now().isoformat(),
                    metadata_json
                ))
                conn.commit()
            
            logger.info(f"Logged decision {decision_id} for {symbol} by {agent_name}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            raise
    
    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Get decision by ID
        
        Args:
            decision_id: Decision identifier
            
        Returns:
            Decision dictionary or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM decisions WHERE decision_id = ?
                """, (decision_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Parse JSON fields
                decision = dict(row)
                decision["agent_output"] = json.loads(decision["agent_output"])
                decision["context"] = json.loads(decision["context"]) if decision["context"] else None
                decision["metadata"] = json.loads(decision["metadata"]) if decision["metadata"] else {}
                decision["timestamp"] = datetime.fromisoformat(decision["timestamp"])
                
                return decision
                
        except Exception as e:
            logger.error(f"Error fetching decision {decision_id}: {e}")
            return None
    
    def list_recent(
        self,
        limit: int = 10,
        agent_name: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent decisions with optional filters
        
        Args:
            limit: Maximum number of decisions to return
            agent_name: Filter by agent name (optional)
            symbol: Filter by symbol (optional)
            
        Returns:
            List of decision dictionaries
        """
        try:
            query = "SELECT * FROM decisions WHERE 1=1"
            params = []
            
            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                decisions = []
                for row in rows:
                    decision = dict(row)
                    decision["agent_output"] = json.loads(decision["agent_output"])
                    decision["context"] = json.loads(decision["context"]) if decision["context"] else None
                    decision["metadata"] = json.loads(decision["metadata"]) if decision["metadata"] else {}
                    decision["timestamp"] = datetime.fromisoformat(decision["timestamp"])
                    decisions.append(decision)
                
                return decisions
                
        except Exception as e:
            logger.error(f"Error listing decisions: {e}")
            return []
    
    def log_factors(
        self,
        decision_id: str,
        factors: List[Dict[str, Any]]
    ):
        """
        Log decision factors
        
        Args:
            decision_id: Decision identifier
            factors: List of factor dictionaries
        """
        try:
            with self._get_connection() as conn:
                for factor in factors:
                    conn.execute("""
                        INSERT INTO decision_factors (
                            decision_id, factor_name, importance, value, description
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        decision_id,
                        factor["name"],
                        factor["importance"],
                        json.dumps(factor["value"]),
                        factor.get("description", "")
                    ))
                conn.commit()
                
            logger.info(f"Logged {len(factors)} factors for decision {decision_id}")
            
        except Exception as e:
            logger.error(f"Error logging factors: {e}")
            raise
    
    def get_factors(self, decision_id: str) -> List[Dict[str, Any]]:
        """
        Get factors for a decision
        
        Args:
            decision_id: Decision identifier
            
        Returns:
            List of factor dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM decision_factors 
                    WHERE decision_id = ?
                    ORDER BY importance DESC
                """, (decision_id,))
                
                rows = cursor.fetchall()
                
                factors = []
                for row in rows:
                    factor = dict(row)
                    factor["value"] = json.loads(factor["value"])
                    factors.append(factor)
                
                return factors
                
        except Exception as e:
            logger.error(f"Error fetching factors: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get decision statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            with self._get_connection() as conn:
                # Total decisions
                cursor = conn.execute("SELECT COUNT(*) FROM decisions")
                total = cursor.fetchone()[0]
                
                # By agent
                cursor = conn.execute("""
                    SELECT agent_name, COUNT(*) as count
                    FROM decisions
                    GROUP BY agent_name
                """)
                by_agent = {row[0]: row[1] for row in cursor.fetchall()}
                
                # By recommendation
                cursor = conn.execute("""
                    SELECT recommendation, COUNT(*) as count
                    FROM decisions
                    GROUP BY recommendation
                """)
                by_recommendation = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Average confidence
                cursor = conn.execute("SELECT AVG(confidence) FROM decisions")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                return {
                    "total_decisions": total,
                    "by_agent": by_agent,
                    "by_recommendation": by_recommendation,
                    "average_confidence": avg_confidence
                }
                
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print(f"Testing DecisionLogger with database: {db_path}")
    
    logger_instance = DecisionLogger(db_path)
    
    # Test logging decision
    print("\n=== Test 1: Log Decision ===")
    agent_output = {
        "recommendation": "BUY",
        "confidence": 0.85,
        "reasoning": "Strong technical signals and positive news sentiment",
        "sentiment_score": 1.5,
        "news_count": 12
    }
    
    decision_id = logger_instance.log_decision(
        symbol="AAPL",
        agent_name="news_agent",
        agent_output=agent_output,
        context={"market_regime": "bull"}
    )
    
    print(f"✓ Logged decision: {decision_id}")
    
    # Test fetching decision
    print("\n=== Test 2: Get Decision ===")
    decision = logger_instance.get_decision(decision_id)
    print(f"✓ Fetched decision:")
    print(f"  Symbol: {decision['symbol']}")
    print(f"  Agent: {decision['agent_name']}")
    print(f"  Recommendation: {decision['recommendation']}")
    print(f"  Confidence: {decision['confidence']:.2f}")
    
    # Test listing recent
    print("\n=== Test 3: List Recent ===")
    recent = logger_instance.list_recent(limit=5)
    print(f"✓ Found {len(recent)} recent decisions")
    
    # Test statistics
    print("\n=== Test 4: Statistics ===")
    stats = logger_instance.get_statistics()
    print(f"✓ Statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    
    print("\n✅ All tests passed!")
    print(f"Database: {db_path}")
