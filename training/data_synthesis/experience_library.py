"""
Experience Library - SQLite-based storage for agent trajectories

This module implements a persistent storage system for agent execution trajectories,
enabling:
- Trajectory storage and retrieval
- Quality-based filtering
- Regime detection and adaptation
- Experience replay for training
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from loguru import logger


class ExperienceLibrary:
    """
    SQLite-based experience library for storing and retrieving agent trajectories.
    
    Features:
    - Persistent storage of execution trajectories
    - Quality-based filtering (reward threshold)
    - Regime detection (market condition changes)
    - Top-k trajectory retrieval
    - Statistics and analytics
    """
    
    def __init__(self, db_path: str = "data/experience_library.db"):
        """
        Initialize the experience library.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Experience Library initialized at {db_path}")
    
    def _init_database(self):
        """Create database schema if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trajectories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                trajectory_data TEXT NOT NULL,
                reward REAL,
                final_decision TEXT,
                market_regime TEXT,
                volatility REAL,
                success INTEGER,
                error_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                INDEX idx_symbol (symbol),
                INDEX idx_reward (reward),
                INDEX idx_timestamp (timestamp),
                INDEX idx_agent_type (agent_type),
                INDEX idx_market_regime (market_regime)
            )
        """)
        
        # Market regime tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                regime_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                characteristics TEXT,
                avg_volatility REAL,
                trend TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                actual_return REAL,
                predicted_direction TEXT,
                correct_prediction INTEGER,
                sharpe_ratio REAL,
                max_drawdown REAL,
                created_at TEXT NOT NULL,
                INDEX idx_symbol_date (symbol, date)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema initialized")
    
    def add_trajectory(
        self,
        symbol: str,
        agent_type: str,
        trajectory_data: Dict,
        reward: Optional[float] = None,
        final_decision: Optional[str] = None,
        market_data: Optional[Dict] = None
    ) -> int:
        """
        Add a trajectory to the experience library.
        
        Args:
            symbol: Stock symbol
            agent_type: Type of agent (news, technical, fundamental, strategist)
            trajectory_data: Complete trajectory data
            reward: Reward score (if available)
            final_decision: Final decision (buy/sell/hold)
            market_data: Market data for regime detection
        
        Returns:
            Trajectory ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Detect market regime
        market_regime = self._detect_market_regime(market_data) if market_data else "unknown"
        
        # Calculate volatility
        volatility = market_data.get('volatility', 0.0) if market_data else 0.0
        
        # Determine success
        success = 1 if reward and reward > 0.5 else 0
        
        # Count errors
        error_count = len(trajectory_data.get('errors', []))
        
        # Insert trajectory
        cursor.execute("""
            INSERT INTO trajectories (
                symbol, timestamp, agent_type, trajectory_data, reward,
                final_decision, market_regime, volatility, success,
                error_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            trajectory_data.get('timestamp', datetime.now().isoformat()),
            agent_type,
            json.dumps(trajectory_data),
            reward,
            final_decision,
            market_regime,
            volatility,
            success,
            error_count,
            datetime.now().isoformat()
        ))
        
        trajectory_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trajectory added: ID={trajectory_id}, symbol={symbol}, reward={reward}")
        
        return trajectory_id
    
    def get_top_trajectories(
        self,
        n: int = 100,
        agent_type: Optional[str] = None,
        min_reward: float = 0.7,
        market_regime: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve top-performing trajectories.
        
        Args:
            n: Number of trajectories to retrieve
            agent_type: Filter by agent type
            min_reward: Minimum reward threshold
            market_regime: Filter by market regime
            symbol: Filter by symbol
        
        Returns:
            List of trajectory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT id, symbol, timestamp, agent_type, trajectory_data, reward,
                   final_decision, market_regime, volatility, success
            FROM trajectories
            WHERE reward >= ?
        """
        params = [min_reward]
        
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)
        
        if market_regime:
            query += " AND market_regime = ?"
            params.append(market_regime)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY reward DESC LIMIT ?"
        params.append(n)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # Parse results
        trajectories = []
        for row in rows:
            trajectories.append({
                'id': row[0],
                'symbol': row[1],
                'timestamp': row[2],
                'agent_type': row[3],
                'trajectory_data': json.loads(row[4]),
                'reward': row[5],
                'final_decision': row[6],
                'market_regime': row[7],
                'volatility': row[8],
                'success': row[9]
            })
        
        logger.info(f"Retrieved {len(trajectories)} top trajectories")
        
        return trajectories
    
    def get_failed_trajectories(
        self,
        n: int = 50,
        max_reward: float = 0.3,
        agent_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve failed trajectories for error healing.
        
        Args:
            n: Number of trajectories to retrieve
            max_reward: Maximum reward threshold
            agent_type: Filter by agent type
        
        Returns:
            List of failed trajectory dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, symbol, timestamp, agent_type, trajectory_data, reward,
                   error_count
            FROM trajectories
            WHERE reward <= ? AND error_count > 0
        """
        params = [max_reward]
        
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)
        
        query += " ORDER BY error_count DESC, reward ASC LIMIT ?"
        params.append(n)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # Parse results
        trajectories = []
        for row in rows:
            trajectories.append({
                'id': row[0],
                'symbol': row[1],
                'timestamp': row[2],
                'agent_type': row[3],
                'trajectory_data': json.loads(row[4]),
                'reward': row[5],
                'error_count': row[6]
            })
        
        logger.info(f"Retrieved {len(trajectories)} failed trajectories")
        
        return trajectories
    
    def detect_regime_shift(
        self,
        lookback_days: int = 30,
        volatility_threshold: float = 0.3
    ) -> Dict:
        """
        Detect if market regime has shifted recently.
        
        Args:
            lookback_days: Days to look back
            volatility_threshold: Threshold for regime change detection
        
        Returns:
            Dict with regime shift information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent trajectories
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        cursor.execute("""
            SELECT market_regime, volatility, COUNT(*) as count
            FROM trajectories
            WHERE timestamp >= ?
            GROUP BY market_regime
            ORDER BY count DESC
        """, (cutoff_date,))
        
        regimes = cursor.fetchall()
        
        # Get historical average volatility
        cursor.execute("""
            SELECT AVG(volatility) as avg_vol
            FROM trajectories
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        historical_vol = cursor.fetchone()[0] or 0.0
        
        # Get recent average volatility
        cursor.execute("""
            SELECT AVG(volatility) as avg_vol
            FROM trajectories
            WHERE timestamp >= ?
        """, (cutoff_date,))
        
        recent_vol = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        # Detect shift
        volatility_change = abs(recent_vol - historical_vol)
        regime_shifted = volatility_change > volatility_threshold
        
        current_regime = regimes[0][0] if regimes else "unknown"
        
        result = {
            'regime_shifted': regime_shifted,
            'current_regime': current_regime,
            'recent_volatility': recent_vol,
            'historical_volatility': historical_vol,
            'volatility_change': volatility_change,
            'regime_distribution': [
                {'regime': r[0], 'volatility': r[1], 'count': r[2]}
                for r in regimes
            ]
        }
        
        logger.info(f"Regime shift detection: shifted={regime_shifted}, regime={current_regime}")
        
        return result
    
    def _detect_market_regime(self, market_data: Dict) -> str:
        """
        Detect current market regime based on market data.
        
        Args:
            market_data: Market data dictionary
        
        Returns:
            Regime name (bull, bear, sideways, high_volatility)
        """
        volatility = market_data.get('volatility', 0.0)
        
        # Simple regime classification
        if volatility > 0.5:
            return "high_volatility"
        elif volatility > 0.3:
            return "moderate_volatility"
        else:
            # Check trend
            price_change = market_data.get('regularMarketChangePercent', 0.0)
            if price_change > 2.0:
                return "bull"
            elif price_change < -2.0:
                return "bear"
            else:
                return "sideways"
    
    def add_performance_metric(
        self,
        symbol: str,
        date: str,
        actual_return: float,
        predicted_direction: str,
        correct_prediction: bool,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None
    ):
        """
        Add performance metrics for backtesting evaluation.
        
        Args:
            symbol: Stock symbol
            date: Date of prediction
            actual_return: Actual return achieved
            predicted_direction: Predicted direction (buy/sell/hold)
            correct_prediction: Whether prediction was correct
            sharpe_ratio: Sharpe ratio (optional)
            max_drawdown: Maximum drawdown (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics (
                symbol, date, actual_return, predicted_direction,
                correct_prediction, sharpe_ratio, max_drawdown, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            date,
            actual_return,
            predicted_direction,
            1 if correct_prediction else 0,
            sharpe_ratio,
            max_drawdown,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Performance metric added: {symbol} on {date}")
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics from the experience library.
        
        Returns:
            Dict with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total trajectories
        cursor.execute("SELECT COUNT(*) FROM trajectories")
        total_trajectories = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute("SELECT AVG(success) FROM trajectories WHERE reward IS NOT NULL")
        success_rate = cursor.fetchone()[0] or 0.0
        
        # Average reward
        cursor.execute("SELECT AVG(reward) FROM trajectories WHERE reward IS NOT NULL")
        avg_reward = cursor.fetchone()[0] or 0.0
        
        # Agent type distribution
        cursor.execute("""
            SELECT agent_type, COUNT(*) as count
            FROM trajectories
            GROUP BY agent_type
        """)
        agent_distribution = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Regime distribution
        cursor.execute("""
            SELECT market_regime, COUNT(*) as count
            FROM trajectories
            GROUP BY market_regime
        """)
        regime_distribution = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_trajectories': total_trajectories,
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'agent_distribution': agent_distribution,
            'regime_distribution': regime_distribution
        }
    
    def clear_old_trajectories(self, days: int = 90):
        """
        Clear trajectories older than specified days.
        
        Args:
            days: Number of days to keep
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            DELETE FROM trajectories
            WHERE created_at < ?
        """, (cutoff_date,))
        
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared {deleted_count} old trajectories")
        
        return deleted_count


# Convenience functions
def get_library(db_path: str = "data/experience_library.db") -> ExperienceLibrary:
    """Get or create experience library instance"""
    return ExperienceLibrary(db_path)
