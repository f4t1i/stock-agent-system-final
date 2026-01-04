"""
Experience Library with PostgreSQL and Vector Extension

Advanced experience storage system using PostgreSQL with pgvector extension
for semantic similarity search of trading trajectories.

Key Features:
1. Stores complete analysis trajectories (input → reasoning → decision → outcome)
2. Vector embeddings for semantic similarity search
3. Success filtering and retrieval
4. Market regime tagging
5. Performance tracking

Based on:
- TradingGroup experience replay
- SIRIUS trajectory storage
- PrimoAgent memory system
"""

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from typing import Dict, List, Optional, Any, Tuple
import json
import time
import numpy as np
from loguru import logger
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class Trajectory:
    """
    A complete analysis trajectory
    
    Captures: Input → Reasoning → Decision → Outcome
    """
    trajectory_id: str
    timestamp: float
    symbol: str
    agent_type: str  # news, technical, fundamental, strategist
    
    # Input
    market_state: Dict
    agent_inputs: Dict
    
    # Reasoning
    reasoning: str
    confidence: float
    
    # Decision
    recommendation: Optional[str] = None
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Outcome (filled later)
    actual_return: Optional[float] = None
    success: Optional[bool] = None
    reward: Optional[float] = None
    
    # Metadata
    market_regime: Optional[str] = None  # bull, bear, sideways, volatile
    embedding: Optional[List[float]] = None  # Vector embedding for similarity search
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trajectory':
        """Create from dictionary"""
        return cls(**data)


class ExperienceLibraryPostgres:
    """
    Experience Library using PostgreSQL with pgvector
    
    Stores and retrieves trading trajectories with semantic search capability.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'trading_experience',
        user: str = 'postgres',
        password: str = 'postgres'
    ):
        """
        Initialize Experience Library
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
        self.conn = None
        self._connect()
        self._initialize_schema()
    
    def _connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            logger.info(f"Connected to PostgreSQL at {self.connection_params['host']}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _initialize_schema(self):
        """Initialize database schema with pgvector extension"""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create trajectories table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id VARCHAR(64) PRIMARY KEY,
                    timestamp DOUBLE PRECISION NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    
                    -- Input
                    market_state JSONB NOT NULL,
                    agent_inputs JSONB NOT NULL,
                    
                    -- Reasoning
                    reasoning TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    
                    -- Decision
                    recommendation VARCHAR(50),
                    position_size REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    
                    -- Outcome
                    actual_return REAL,
                    success BOOLEAN,
                    reward REAL,
                    
                    -- Metadata
                    market_regime VARCHAR(50),
                    embedding vector(384),  -- 384-dim for sentence-transformers
                    
                    -- Indexes
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_symbol 
                ON trajectories(symbol);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_agent_type 
                ON trajectories(agent_type);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_success 
                ON trajectories(success);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_market_regime 
                ON trajectories(market_regime);
            """)
            
            # Vector similarity index (HNSW for fast approximate search)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_embedding 
                ON trajectories USING hnsw (embedding vector_cosine_ops);
            """)
            
            # Create statistics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_statistics (
                    stat_date DATE PRIMARY KEY,
                    total_trajectories INTEGER,
                    successful_trajectories INTEGER,
                    avg_reward REAL,
                    by_agent_type JSONB,
                    by_market_regime JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            self.conn.commit()
            logger.info("Database schema initialized")
    
    def store_trajectory(self, trajectory: Trajectory) -> bool:
        """
        Store a trajectory in the database
        
        Args:
            trajectory: Trajectory to store
        
        Returns:
            bool: Success status
        """
        try:
            with self.conn.cursor() as cur:
                # Convert embedding to pgvector format
                embedding_str = None
                if trajectory.embedding:
                    embedding_str = f"[{','.join(map(str, trajectory.embedding))}]"
                
                cur.execute("""
                    INSERT INTO trajectories (
                        trajectory_id, timestamp, symbol, agent_type,
                        market_state, agent_inputs,
                        reasoning, confidence,
                        recommendation, position_size, stop_loss, take_profit,
                        actual_return, success, reward,
                        market_regime, embedding
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    ON CONFLICT (trajectory_id) DO UPDATE SET
                        actual_return = EXCLUDED.actual_return,
                        success = EXCLUDED.success,
                        reward = EXCLUDED.reward;
                """, (
                    trajectory.trajectory_id,
                    trajectory.timestamp,
                    trajectory.symbol,
                    trajectory.agent_type,
                    Json(trajectory.market_state),
                    Json(trajectory.agent_inputs),
                    trajectory.reasoning,
                    trajectory.confidence,
                    trajectory.recommendation,
                    trajectory.position_size,
                    trajectory.stop_loss,
                    trajectory.take_profit,
                    trajectory.actual_return,
                    trajectory.success,
                    trajectory.reward,
                    trajectory.market_regime,
                    embedding_str
                ))
                
                self.conn.commit()
                logger.debug(f"Stored trajectory {trajectory.trajectory_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to store trajectory: {e}")
            self.conn.rollback()
            return False
    
    def update_outcome(
        self,
        trajectory_id: str,
        actual_return: float,
        success: bool,
        reward: float
    ) -> bool:
        """
        Update trajectory outcome after actual results are known
        
        Args:
            trajectory_id: Trajectory ID
            actual_return: Actual return achieved
            success: Whether trajectory was successful
            reward: Reward value
        
        Returns:
            bool: Success status
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE trajectories
                    SET actual_return = %s,
                        success = %s,
                        reward = %s
                    WHERE trajectory_id = %s;
                """, (actual_return, success, reward, trajectory_id))
                
                self.conn.commit()
                logger.debug(f"Updated outcome for trajectory {trajectory_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
            self.conn.rollback()
            return False
    
    def get_successful_trajectories(
        self,
        agent_type: Optional[str] = None,
        market_regime: Optional[str] = None,
        min_reward: float = 0.0,
        limit: int = 100
    ) -> List[Trajectory]:
        """
        Retrieve successful trajectories for SFT training
        
        Args:
            agent_type: Filter by agent type
            market_regime: Filter by market regime
            min_reward: Minimum reward threshold
            limit: Maximum number of trajectories
        
        Returns:
            List of successful trajectories
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT * FROM trajectories
                    WHERE success = TRUE
                    AND reward >= %s
                """
                params = [min_reward]
                
                if agent_type:
                    query += " AND agent_type = %s"
                    params.append(agent_type)
                
                if market_regime:
                    query += " AND market_regime = %s"
                    params.append(market_regime)
                
                query += " ORDER BY reward DESC LIMIT %s;"
                params.append(limit)
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                trajectories = []
                for row in rows:
                    # Convert JSONB to dict
                    row['market_state'] = dict(row['market_state'])
                    row['agent_inputs'] = dict(row['agent_inputs'])
                    
                    # Remove created_at (not in Trajectory dataclass)
                    row.pop('created_at', None)
                    
                    trajectory = Trajectory.from_dict(row)
                    trajectories.append(trajectory)
                
                logger.info(f"Retrieved {len(trajectories)} successful trajectories")
                return trajectories
        
        except Exception as e:
            logger.error(f"Failed to retrieve successful trajectories: {e}")
            return []
    
    def find_similar_trajectories(
        self,
        query_embedding: List[float],
        agent_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[Trajectory, float]]:
        """
        Find similar trajectories using vector similarity search
        
        Args:
            query_embedding: Query embedding vector
            agent_type: Filter by agent type
            limit: Maximum number of results
        
        Returns:
            List of (trajectory, similarity_score) tuples
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Convert embedding to pgvector format
                embedding_str = f"[{','.join(map(str, query_embedding))}]"
                
                query = """
                    SELECT *, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM trajectories
                    WHERE embedding IS NOT NULL
                """
                params = [embedding_str]
                
                if agent_type:
                    query += " AND agent_type = %s"
                    params.append(agent_type)
                
                query += " ORDER BY embedding <=> %s::vector LIMIT %s;"
                params.extend([embedding_str, limit])
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    similarity = row.pop('similarity')
                    
                    # Convert JSONB to dict
                    row['market_state'] = dict(row['market_state'])
                    row['agent_inputs'] = dict(row['agent_inputs'])
                    
                    # Remove created_at
                    row.pop('created_at', None)
                    
                    trajectory = Trajectory.from_dict(row)
                    results.append((trajectory, similarity))
                
                logger.info(f"Found {len(results)} similar trajectories")
                return results
        
        except Exception as e:
            logger.error(f"Failed to find similar trajectories: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get trajectory statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Overall statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful,
                        AVG(reward) FILTER (WHERE reward IS NOT NULL) as avg_reward,
                        AVG(confidence) as avg_confidence
                    FROM trajectories;
                """)
                overall = cur.fetchone()
                
                # By agent type
                cur.execute("""
                    SELECT 
                        agent_type,
                        COUNT(*) as count,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful,
                        AVG(reward) FILTER (WHERE reward IS NOT NULL) as avg_reward
                    FROM trajectories
                    GROUP BY agent_type;
                """)
                by_agent = cur.fetchall()
                
                # By market regime
                cur.execute("""
                    SELECT 
                        market_regime,
                        COUNT(*) as count,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful,
                        AVG(reward) FILTER (WHERE reward IS NOT NULL) as avg_reward
                    FROM trajectories
                    WHERE market_regime IS NOT NULL
                    GROUP BY market_regime;
                """)
                by_regime = cur.fetchall()
                
                return {
                    'overall': dict(overall),
                    'by_agent_type': [dict(row) for row in by_agent],
                    'by_market_regime': [dict(row) for row in by_regime]
                }
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def generate_trajectory_id(
    symbol: str,
    agent_type: str,
    timestamp: float
) -> str:
    """Generate unique trajectory ID"""
    data = f"{symbol}_{agent_type}_{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


if __name__ == '__main__':
    # Test
    library = ExperienceLibraryPostgres(
        host='localhost',
        database='trading_experience',
        user='postgres',
        password='postgres'
    )
    
    # Create test trajectory
    trajectory = Trajectory(
        trajectory_id=generate_trajectory_id('AAPL', 'news', time.time()),
        timestamp=time.time(),
        symbol='AAPL',
        agent_type='news',
        market_state={'price': 150.0, 'volume': 1000000},
        agent_inputs={'news': ['Positive earnings']},
        reasoning='Strong earnings beat expectations',
        confidence=0.85,
        recommendation='buy',
        market_regime='bull',
        embedding=[0.1] * 384  # Dummy embedding
    )
    
    # Store
    library.store_trajectory(trajectory)
    
    # Update outcome
    library.update_outcome(
        trajectory.trajectory_id,
        actual_return=0.05,
        success=True,
        reward=1.0
    )
    
    # Retrieve successful
    successful = library.get_successful_trajectories(agent_type='news', limit=10)
    print(f"Found {len(successful)} successful trajectories")
    
    # Statistics
    stats = library.get_statistics()
    print(f"Statistics: {stats}")
    
    library.close()
