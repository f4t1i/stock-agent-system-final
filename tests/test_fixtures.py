#!/usr/bin/env python3
"""
Test Fixtures for Acceptance Tests - Task 4.1

Provides test fixtures for all acceptance tests.

Features:
- Postgres test database setup
- Mock backtest data generation
- Mock trajectory data
- Mock dataset data
- Test data cleanup
- Fixture management utilities

Phase A1 Week 3-4: Task 4.1 COMPLETE
"""

import os
import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
from loguru import logger


@dataclass
class TestConfig:
    """Test configuration"""
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "test_stock_agent"
    db_user: str = "postgres"
    db_password: str = "postgres"
    
    @classmethod
    def from_env(cls) -> 'TestConfig':
        """Create from environment variables"""
        return cls(
            db_host=os.getenv('TEST_DB_HOST', 'localhost'),
            db_port=int(os.getenv('TEST_DB_PORT', '5432')),
            db_name=os.getenv('TEST_DB_NAME', 'test_stock_agent'),
            db_user=os.getenv('TEST_DB_USER', 'postgres'),
            db_password=os.getenv('TEST_DB_PASSWORD', 'postgres')
        )


class TestDatabase:
    """
    Test database management
    """
    
    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize test database
        
        Args:
            config: Test configuration (optional, uses defaults)
        """
        self.config = config or TestConfig.from_env()
        self.conn = None
        
        logger.info(f"TestDatabase initialized: {self.config.db_name}")
    
    def connect(self):
        """Connect to test database"""
        try:
            self.conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to test database")
        except Exception as e:
            logger.warning(f"Could not connect to test database: {e}")
            logger.info("Tests will run in mock mode")
            self.conn = None
    
    def disconnect(self):
        """Disconnect from test database"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Disconnected from test database")
    
    def create_tables(self):
        """Create test tables"""
        if not self.conn:
            logger.warning("No database connection, skipping table creation")
            return
        
        with self.conn.cursor() as cur:
            # Backtests table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id UUID PRIMARY KEY,
                    agent_type VARCHAR(50),
                    start_date DATE,
                    end_date DATE,
                    status VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Trajectories table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id UUID PRIMARY KEY,
                    backtest_id UUID REFERENCES backtests(id),
                    agent_type VARCHAR(50),
                    state JSONB,
                    action JSONB,
                    reasoning TEXT,
                    reward FLOAT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.commit()
            logger.info("Test tables created")
    
    def drop_tables(self):
        """Drop test tables"""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS trajectories CASCADE")
            cur.execute("DROP TABLE IF EXISTS backtests CASCADE")
            self.conn.commit()
            logger.info("Test tables dropped")
    
    def clear_tables(self):
        """Clear all data from test tables"""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM trajectories")
            cur.execute("DELETE FROM backtests")
            self.conn.commit()
            logger.info("Test tables cleared")
    
    def insert_backtest(self, backtest: Dict) -> str:
        """Insert backtest"""
        if not self.conn:
            return str(uuid.uuid4())
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO backtests (id, agent_type, start_date, end_date, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                backtest['id'],
                backtest['agent_type'],
                backtest['start_date'],
                backtest['end_date'],
                backtest['status'],
                backtest['created_at']
            ))
            
            result = cur.fetchone()
            self.conn.commit()
            
            return str(result['id'])
    
    def insert_trajectory(self, trajectory: Dict) -> str:
        """Insert trajectory"""
        if not self.conn:
            return str(uuid.uuid4())
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trajectories (
                    id, backtest_id, agent_type, state, action, reasoning, reward, confidence, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                trajectory['id'],
                trajectory['backtest_id'],
                trajectory['agent_type'],
                json.dumps(trajectory['state']),
                json.dumps(trajectory['action']),
                trajectory['reasoning'],
                trajectory['reward'],
                trajectory['confidence'],
                trajectory['created_at']
            ))
            
            result = cur.fetchone()
            self.conn.commit()
            
            return str(result['id'])
    
    def get_backtest(self, backtest_id: str) -> Optional[Dict]:
        """Get backtest by ID"""
        if not self.conn:
            return None
        
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM backtests WHERE id = %s", (backtest_id,))
            return cur.fetchone()
    
    def get_trajectories(self, backtest_id: str) -> List[Dict]:
        """Get trajectories for backtest"""
        if not self.conn:
            return []
        
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM trajectories WHERE backtest_id = %s ORDER BY created_at",
                (backtest_id,)
            )
            return cur.fetchall()


class MockDataGenerator:
    """
    Generate mock test data
    """
    
    @staticmethod
    def generate_backtest(
        agent_type: str = "technical",
        status: str = "completed",
        days: int = 30
    ) -> Dict:
        """
        Generate mock backtest
        
        Args:
            agent_type: Agent type
            status: Backtest status
            days: Number of days
        
        Returns:
            Backtest dict
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return {
            'id': str(uuid.uuid4()),
            'agent_type': agent_type,
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'status': status,
            'created_at': datetime.now()
        }
    
    @staticmethod
    def generate_trajectory(
        backtest_id: str,
        agent_type: str = "technical",
        reward: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> Dict:
        """
        Generate mock trajectory
        
        Args:
            backtest_id: Backtest ID
            agent_type: Agent type
            reward: Reward (optional, random if not provided)
            confidence: Confidence (optional, random if not provided)
        
        Returns:
            Trajectory dict
        """
        if reward is None:
            reward = random.uniform(-1.0, 1.0)
        
        if confidence is None:
            confidence = random.uniform(0.5, 1.0)
        
        # Generate realistic state
        state = {
            'symbol': random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']),
            'price': round(random.uniform(100, 500), 2),
            'volume': random.randint(1000000, 10000000),
            'indicators': {
                'rsi': round(random.uniform(30, 70), 2),
                'macd': round(random.uniform(-5, 5), 2),
                'sma_20': round(random.uniform(100, 500), 2),
                'sma_50': round(random.uniform(100, 500), 2)
            }
        }
        
        # Generate realistic action
        action = {
            'type': random.choice(['buy', 'sell', 'hold']),
            'quantity': random.randint(10, 100),
            'price': state['price']
        }
        
        # Generate reasoning
        reasoning = f"Based on {agent_type} analysis, recommend {action['type']} due to favorable indicators."
        
        return {
            'id': str(uuid.uuid4()),
            'backtest_id': backtest_id,
            'agent_type': agent_type,
            'state': state,
            'action': action,
            'reasoning': reasoning,
            'reward': reward,
            'confidence': confidence,
            'created_at': datetime.now()
        }
    
    @staticmethod
    def generate_trajectories(
        backtest_id: str,
        count: int = 10,
        agent_type: str = "technical",
        min_reward: float = 0.0,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Generate multiple trajectories
        
        Args:
            backtest_id: Backtest ID
            count: Number of trajectories
            agent_type: Agent type
            min_reward: Minimum reward
            min_confidence: Minimum confidence
        
        Returns:
            List of trajectory dicts
        """
        trajectories = []
        
        for _ in range(count):
            reward = random.uniform(min_reward, 1.0)
            confidence = random.uniform(min_confidence, 1.0)
            
            trajectory = MockDataGenerator.generate_trajectory(
                backtest_id=backtest_id,
                agent_type=agent_type,
                reward=reward,
                confidence=confidence
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    @staticmethod
    def generate_dataset_metadata(
        agent_type: str = "technical",
        version: str = "1.0.0",
        format: str = "chatml",
        example_count: int = 100
    ) -> Dict:
        """
        Generate dataset metadata
        
        Args:
            agent_type: Agent type
            version: Version
            format: Format
            example_count: Example count
        
        Returns:
            Metadata dict
        """
        return {
            'dataset_id': str(uuid.uuid4()),
            'version_id': str(uuid.uuid4()),
            'agent_type': agent_type,
            'version': version,
            'format': format,
            'example_count': example_count,
            'quality_score': round(random.uniform(0.7, 0.95), 3),
            'created_at': datetime.now().isoformat(),
            'sha256': 'a' * 64  # Mock hash
        }


class TestFixtures:
    """
    Test fixtures manager
    """
    
    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize test fixtures
        
        Args:
            config: Test configuration (optional)
        """
        self.config = config or TestConfig.from_env()
        self.db = TestDatabase(self.config)
        self.generator = MockDataGenerator()
        
        logger.info("TestFixtures initialized")
    
    def setup(self):
        """Setup test environment"""
        self.db.connect()
        self.db.create_tables()
        logger.info("Test environment setup complete")
    
    def teardown(self):
        """Teardown test environment"""
        self.db.clear_tables()
        self.db.disconnect()
        logger.info("Test environment teardown complete")
    
    def create_backtest_with_trajectories(
        self,
        agent_type: str = "technical",
        trajectory_count: int = 10,
        min_reward: float = 0.0,
        min_confidence: float = 0.5
    ) -> tuple[Dict, List[Dict]]:
        """
        Create backtest with trajectories
        
        Args:
            agent_type: Agent type
            trajectory_count: Number of trajectories
            min_reward: Minimum reward
            min_confidence: Minimum confidence
        
        Returns:
            Tuple of (backtest, trajectories)
        """
        # Generate backtest
        backtest = self.generator.generate_backtest(agent_type=agent_type)
        
        # Insert backtest
        self.db.insert_backtest(backtest)
        
        # Generate trajectories
        trajectories = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=trajectory_count,
            agent_type=agent_type,
            min_reward=min_reward,
            min_confidence=min_confidence
        )
        
        # Insert trajectories
        for trajectory in trajectories:
            self.db.insert_trajectory(trajectory)
        
        logger.info(
            f"Created backtest {backtest['id']} with {len(trajectories)} trajectories"
        )
        
        return backtest, trajectories
    
    def create_test_dataset(
        self,
        output_dir: str,
        agent_type: str = "technical",
        example_count: int = 100,
        format: str = "chatml"
    ) -> str:
        """
        Create test dataset file
        
        Args:
            output_dir: Output directory
            agent_type: Agent type
            example_count: Number of examples
            format: Format (chatml or alpaca)
        
        Returns:
            Path to dataset file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate examples
        examples = []
        
        for i in range(example_count):
            if format == "chatml":
                example = {
                    'messages': [
                        {'role': 'system', 'content': f'{agent_type} analysis agent'},
                        {'role': 'user', 'content': f'Analyze stock {i}'},
                        {'role': 'assistant', 'content': f'Analysis result {i}'}
                    ]
                }
            else:  # alpaca
                example = {
                    'instruction': f'Analyze stock {i}',
                    'input': f'Stock data {i}',
                    'output': f'Analysis result {i}'
                }
            
            examples.append(example)
        
        # Write to file
        filename = f"{agent_type}_{format}_v1.0.0.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Created test dataset: {filepath}")
        
        return filepath
    
    def create_test_metadata(
        self,
        output_dir: str,
        agent_type: str = "technical",
        version: str = "1.0.0",
        format: str = "chatml",
        example_count: int = 100
    ) -> str:
        """
        Create test metadata file
        
        Args:
            output_dir: Output directory
            agent_type: Agent type
            version: Version
            format: Format
            example_count: Example count
        
        Returns:
            Path to metadata file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate metadata
        metadata = self.generator.generate_dataset_metadata(
            agent_type=agent_type,
            version=version,
            format=format,
            example_count=example_count
        )
        
        # Write to file
        filename = f"{agent_type}_{format}_v{version}_metadata.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created test metadata: {filepath}")
        
        return filepath


# ============================================================================
# Pytest Fixtures
# ============================================================================

try:
    import pytest
    
    @pytest.fixture(scope="session")
    def test_config():
        """Test configuration fixture"""
        return TestConfig.from_env()
    
    @pytest.fixture(scope="session")
    def test_db(test_config):
        """Test database fixture"""
        db = TestDatabase(test_config)
        db.connect()
        db.create_tables()
        yield db
        db.clear_tables()
        db.disconnect()
    
    @pytest.fixture(scope="function")
    def test_fixtures(test_config):
        """Test fixtures fixture"""
        fixtures = TestFixtures(test_config)
        fixtures.setup()
        yield fixtures
        fixtures.teardown()
    
    @pytest.fixture
    def mock_backtest(test_fixtures):
        """Mock backtest fixture"""
        backtest, trajectories = test_fixtures.create_backtest_with_trajectories(
            agent_type="technical",
            trajectory_count=10
        )
        return backtest
    
    @pytest.fixture
    def mock_trajectories(test_fixtures, mock_backtest):
        """Mock trajectories fixture"""
        return test_fixtures.db.get_trajectories(mock_backtest['id'])

except ImportError:
    logger.warning("pytest not installed, fixtures will not be available")


if __name__ == "__main__":
    # Example usage
    print("=== Test Fixtures Example ===\n")
    
    # Create fixtures
    fixtures = TestFixtures()
    
    print("Setting up test environment...")
    fixtures.setup()
    
    print("\nCreating backtest with trajectories...")
    backtest, trajectories = fixtures.create_backtest_with_trajectories(
        agent_type="technical",
        trajectory_count=5,
        min_reward=0.5,
        min_confidence=0.7
    )
    
    print(f"  Backtest ID: {backtest['id']}")
    print(f"  Agent Type: {backtest['agent_type']}")
    print(f"  Status: {backtest['status']}")
    print(f"  Trajectories: {len(trajectories)}")
    
    print("\nCreating test dataset...")
    dataset_path = fixtures.create_test_dataset(
        output_dir="/tmp/test_datasets",
        agent_type="technical",
        example_count=10,
        format="chatml"
    )
    print(f"  Dataset: {dataset_path}")
    
    print("\nCreating test metadata...")
    metadata_path = fixtures.create_test_metadata(
        output_dir="/tmp/test_datasets",
        agent_type="technical",
        version="1.0.0",
        format="chatml",
        example_count=10
    )
    print(f"  Metadata: {metadata_path}")
    
    print("\nTearing down test environment...")
    fixtures.teardown()
    
    print("\n✅ Example completed!")
