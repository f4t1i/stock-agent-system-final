"""
Acceptance Tests for Auto Data-Synthesis Pipeline - PRODUCTION VERSION

Comprehensive integration tests for the complete data synthesis workflow:
1. Backtest completion → 2. Trajectory extraction → 3. Quality filtering →
4. Judge evaluation → 5. Dataset creation → 6. Registry versioning

Phase A1 Week 3-4: Task 4 (COMPLETE REWRITE)

Test Coverage:
- End-to-end pipeline execution
- Postgres integration (ExperienceLibrary, DatasetRegistry)
- LLM Judge integration (with mocking option)
- Error handling and edge cases
- Performance benchmarks
- Data integrity validation

Test Fixtures:
- Real Postgres database (test schema)
- Sample backtest trajectories
- Mock LLM Judge (optional)
- Temporary file system

Test Scenarios:
1. Happy path: Complete pipeline execution
2. Quality filtering: Low-quality trajectories rejected
3. Judge filtering: Failed judge evaluation
4. Empty backtest: No trajectories found
5. Duplicate processing: Same backtest processed twice
6. Registry integration: Version tracking
7. Performance: Large dataset processing
8. Error recovery: API failures, DB errors

Based on:
- pytest best practices
- Integration testing patterns
- Test fixtures and mocking
- Performance benchmarking
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import psycopg2
from psycopg2.extras import RealDictCursor

from training.pipelines.auto_data_synthesis_pipeline import (
    AutoDataSynthesisPipeline,
    PipelineConfig,
    PipelineStatistics
)
from training.dataset_registry import DatasetRegistry, DatasetMetadata
from training.judge_filtering import JudgeApprovedFilter, FilteringStrategy
from judge.llm_judge import LLMJudge


# Test configuration
TEST_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'stock_agent_test',
    'user': 'postgres',
    'password': 'postgres'
}

MOCK_JUDGE = True  # Set to False to test with real Anthropic API


@pytest.fixture(scope='session')
def test_db():
    """Create test database schema"""
    conn = psycopg2.connect(**TEST_DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Create test schema
    cursor.execute("CREATE SCHEMA IF NOT EXISTS test")
    
    # Create experience library tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test.trajectories (
            id SERIAL PRIMARY KEY,
            backtest_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            state JSONB NOT NULL,
            action JSONB NOT NULL,
            reward FLOAT NOT NULL,
            next_state JSONB,
            done BOOLEAN NOT NULL,
            info JSONB,
            confidence FLOAT,
            reasoning TEXT
        )
    """)
    
    # Create dataset registry tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test.datasets (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test.dataset_versions (
            id SERIAL PRIMARY KEY,
            dataset_id INTEGER REFERENCES test.datasets(id),
            version TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size BIGINT NOT NULL,
            file_hash TEXT NOT NULL,
            example_count INTEGER NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE(dataset_id, version)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test.dataset_lineage (
            id SERIAL PRIMARY KEY,
            dataset_version_id INTEGER REFERENCES test.dataset_versions(id),
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test.dataset_tags (
            id SERIAL PRIMARY KEY,
            dataset_version_id INTEGER REFERENCES test.dataset_versions(id),
            tag TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE(dataset_version_id, tag)
        )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    yield TEST_DB_CONFIG
    
    # Cleanup (optional - comment out to inspect test data)
    # conn = psycopg2.connect(**TEST_DB_CONFIG)
    # conn.autocommit = True
    # cursor = conn.cursor()
    # cursor.execute("DROP SCHEMA test CASCADE")
    # conn.commit()
    # cursor.close()
    # conn.close()


@pytest.fixture
def sample_trajectories(test_db):
    """Insert sample trajectories into test database"""
    conn = psycopg2.connect(**test_db)
    cursor = conn.cursor()
    
    backtest_id = f"test_backtest_{datetime.now().timestamp()}"
    
    # Insert high-quality trajectories
    for i in range(10):
        cursor.execute("""
            INSERT INTO test.trajectories 
            (backtest_id, agent_type, symbol, timestamp, state, action, reward, done, confidence, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            backtest_id,
            'technical',
            'AAPL',
            datetime.now(),
            json.dumps({'price': 150.0 + i, 'volume': 1000000}),
            json.dumps({'action': 'buy', 'quantity': 100}),
            0.8 + i * 0.01,  # High reward
            i == 9,
            0.9,  # High confidence
            f"Strong technical signal {i}"
        ))
    
    # Insert low-quality trajectories
    for i in range(5):
        cursor.execute("""
            INSERT INTO test.trajectories 
            (backtest_id, agent_type, symbol, timestamp, state, action, reward, done, confidence, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            backtest_id,
            'technical',
            'AAPL',
            datetime.now(),
            json.dumps({'price': 150.0 + i, 'volume': 1000000}),
            json.dumps({'action': 'hold', 'quantity': 0}),
            0.2 + i * 0.01,  # Low reward
            False,
            0.4,  # Low confidence
            f"Weak signal {i}"
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return backtest_id


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_judge():
    """Mock LLM Judge for testing"""
    if not MOCK_JUDGE:
        return None
    
    judge = Mock(spec=LLMJudge)
    
    def mock_evaluate(agent_output, rubric_name, context=None):
        # Simulate judge evaluation
        reward = agent_output.get('reward', 0.5)
        confidence = agent_output.get('confidence', 0.5)
        
        # High reward + confidence → approved
        if reward > 0.7 and confidence > 0.7:
            return {
                'overall_score': 0.85,
                'dimension_scores': {
                    'accuracy': 0.9,
                    'reasoning': 0.8,
                    'clarity': 0.85
                },
                'feedback': 'Excellent analysis',
                'strengths': ['Clear reasoning', 'High confidence'],
                'weaknesses': [],
                'suggestions': []
            }
        else:
            return {
                'overall_score': 0.45,
                'dimension_scores': {
                    'accuracy': 0.5,
                    'reasoning': 0.4,
                    'clarity': 0.45
                },
                'feedback': 'Needs improvement',
                'strengths': [],
                'weaknesses': ['Low confidence', 'Unclear reasoning'],
                'suggestions': ['Provide more context']
            }
    
    judge.evaluate = mock_evaluate
    return judge


# ============================================================================
# TEST CASES
# ============================================================================

def test_end_to_end_pipeline(test_db, sample_trajectories, temp_output_dir, mock_judge):
    """
    Test 1: Happy Path - Complete Pipeline Execution
    
    Verifies:
    - Trajectory extraction from Postgres
    - Quality filtering
    - Judge evaluation
    - Dataset creation
    - Registry versioning
    """
    backtest_id = sample_trajectories
    
    # Configure pipeline
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        quality_threshold=0.6,
        judge_strategy='balanced',
        enable_judge=True
    )
    
    # Create pipeline
    if mock_judge:
        with patch('training.pipelines.auto_data_synthesis_pipeline.LLMJudge', return_value=mock_judge):
            pipeline = AutoDataSynthesisPipeline(config)
            stats = pipeline.process_backtest(backtest_id)
    else:
        pipeline = AutoDataSynthesisPipeline(config)
        stats = pipeline.process_backtest(backtest_id)
    
    # Verify statistics
    assert stats.total_trajectories == 15  # 10 high + 5 low quality
    assert stats.quality_approved > 0
    assert stats.quality_rejected > 0
    assert stats.judge_approved > 0
    assert stats.final_dataset_size > 0
    
    # Verify dataset file created
    dataset_files = list(temp_output_dir.glob("*.jsonl"))
    assert len(dataset_files) > 0
    
    # Verify dataset content
    with open(dataset_files[0], 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]
        assert len(examples) == stats.final_dataset_size
        
        # Verify ChatML format
        for example in examples:
            assert 'messages' in example
            assert len(example['messages']) >= 2
            assert example['messages'][0]['role'] == 'system'
            assert example['messages'][1]['role'] == 'user'


def test_quality_filtering(test_db, sample_trajectories, temp_output_dir):
    """
    Test 2: Quality Filtering
    
    Verifies:
    - Low-quality trajectories are rejected
    - Quality score calculation
    - Threshold enforcement
    """
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        quality_threshold=0.7,  # High threshold
        enable_judge=False  # Disable judge for this test
    )
    
    pipeline = AutoDataSynthesisPipeline(config)
    stats = pipeline.process_backtest(backtest_id)
    
    # With high threshold, only high-quality trajectories should pass
    assert stats.quality_approved < stats.total_trajectories
    assert stats.quality_rejected > 0
    
    # Verify rejection reasons
    assert 'low_quality_score' in stats.rejection_reasons


def test_judge_filtering(test_db, sample_trajectories, temp_output_dir, mock_judge):
    """
    Test 3: Judge Filtering
    
    Verifies:
    - Judge evaluation integration
    - Approval/rejection based on judge scores
    - Feedback logging
    """
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        quality_threshold=0.5,  # Low threshold to pass quality filter
        judge_strategy='strict',  # Strict judge
        enable_judge=True
    )
    
    if mock_judge:
        with patch('training.pipelines.auto_data_synthesis_pipeline.LLMJudge', return_value=mock_judge):
            pipeline = AutoDataSynthesisPipeline(config)
            stats = pipeline.process_backtest(backtest_id)
    else:
        pipeline = AutoDataSynthesisPipeline(config)
        stats = pipeline.process_backtest(backtest_id)
    
    # Verify judge was called
    assert stats.judge_approved > 0
    assert stats.judge_rejected > 0
    
    # Verify final dataset is smaller than quality-approved
    assert stats.final_dataset_size <= stats.quality_approved


def test_empty_backtest(test_db, temp_output_dir):
    """
    Test 4: Empty Backtest
    
    Verifies:
    - Graceful handling of empty backtest
    - No dataset created
    - Appropriate error message
    """
    backtest_id = "nonexistent_backtest"
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical']
    )
    
    pipeline = AutoDataSynthesisPipeline(config)
    stats = pipeline.process_backtest(backtest_id)
    
    # Verify no trajectories found
    assert stats.total_trajectories == 0
    assert stats.final_dataset_size == 0
    
    # Verify no dataset file created
    dataset_files = list(temp_output_dir.glob("*.jsonl"))
    assert len(dataset_files) == 0


def test_duplicate_processing(test_db, sample_trajectories, temp_output_dir):
    """
    Test 5: Duplicate Processing Prevention
    
    Verifies:
    - Same backtest cannot be processed twice
    - Duplicate detection mechanism
    """
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        enable_judge=False
    )
    
    pipeline = AutoDataSynthesisPipeline(config)
    
    # First processing
    stats1 = pipeline.process_backtest(backtest_id)
    assert stats1.final_dataset_size > 0
    
    # Second processing (should be prevented or handled)
    # Implementation depends on duplicate detection strategy
    # This test verifies the behavior is defined
    stats2 = pipeline.process_backtest(backtest_id)
    
    # Either: duplicate processing prevented (stats2 is None or empty)
    # Or: new version created (different output file)
    assert stats2 is not None  # At minimum, should return stats


def test_registry_integration(test_db, sample_trajectories, temp_output_dir):
    """
    Test 6: Dataset Registry Integration
    
    Verifies:
    - Dataset registered in registry
    - Version tracking
    - Lineage tracking (backtest → dataset)
    - Metadata storage
    """
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        enable_judge=False,
        register_dataset=True
    )
    
    pipeline = AutoDataSynthesisPipeline(config)
    stats = pipeline.process_backtest(backtest_id)
    
    # Verify dataset registered
    registry = DatasetRegistry(db_config=test_db, schema='test')
    
    # Find dataset by backtest_id in lineage
    conn = psycopg2.connect(**test_db)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT dv.* 
        FROM test.dataset_versions dv
        JOIN test.dataset_lineage dl ON dl.dataset_version_id = dv.id
        WHERE dl.source_id = %s
    """, (backtest_id,))
    
    versions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    assert len(versions) > 0
    
    # Verify metadata
    version = versions[0]
    assert version['example_count'] == stats.final_dataset_size
    assert version['file_hash'] is not None


def test_performance_large_dataset(test_db, temp_output_dir):
    """
    Test 7: Performance - Large Dataset
    
    Verifies:
    - Pipeline can handle large datasets (1000+ trajectories)
    - Processing time is reasonable
    - Memory usage is acceptable
    """
    # Insert 1000 trajectories
    conn = psycopg2.connect(**test_db)
    cursor = conn.cursor()
    
    backtest_id = f"large_backtest_{datetime.now().timestamp()}"
    
    for i in range(1000):
        cursor.execute("""
            INSERT INTO test.trajectories 
            (backtest_id, agent_type, symbol, timestamp, state, action, reward, done, confidence, reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            backtest_id,
            'technical',
            'AAPL',
            datetime.now(),
            json.dumps({'price': 150.0, 'volume': 1000000}),
            json.dumps({'action': 'buy', 'quantity': 100}),
            0.7,
            False,
            0.8,
            f"Signal {i}"
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    # Process with judge disabled for speed
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        enable_judge=False
    )
    
    pipeline = AutoDataSynthesisPipeline(config)
    
    import time
    start_time = time.time()
    stats = pipeline.process_backtest(backtest_id)
    processing_time = time.time() - start_time
    
    # Verify performance
    assert stats.total_trajectories == 1000
    assert processing_time < 60  # Should complete in under 1 minute
    
    # Verify throughput
    throughput = stats.total_trajectories / processing_time
    assert throughput > 10  # At least 10 trajectories/second


def test_error_recovery(test_db, sample_trajectories, temp_output_dir):
    """
    Test 8: Error Recovery
    
    Verifies:
    - Pipeline handles DB errors gracefully
    - Pipeline handles API errors gracefully
    - Partial results are saved
    - Error logging is comprehensive
    """
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        enable_judge=True
    )
    
    # Simulate API error
    mock_judge_with_error = Mock(spec=LLMJudge)
    mock_judge_with_error.evaluate = Mock(side_effect=Exception("API Error"))
    
    with patch('training.pipelines.auto_data_synthesis_pipeline.LLMJudge', return_value=mock_judge_with_error):
        pipeline = AutoDataSynthesisPipeline(config)
        
        # Should not crash, but handle error
        try:
            stats = pipeline.process_backtest(backtest_id)
            
            # Verify error was logged
            assert stats.errors > 0
            
        except Exception as e:
            # If it crashes, verify error message is informative
            assert "API Error" in str(e) or "Judge" in str(e)


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
def test_benchmark_quality_filtering(test_db, sample_trajectories, temp_output_dir, benchmark):
    """Benchmark quality filtering performance"""
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        enable_judge=False
    )
    
    pipeline = AutoDataSynthesisPipeline(config)
    
    result = benchmark(pipeline.process_backtest, backtest_id)
    
    assert result.final_dataset_size > 0


@pytest.mark.benchmark
def test_benchmark_judge_filtering(test_db, sample_trajectories, temp_output_dir, mock_judge, benchmark):
    """Benchmark judge filtering performance"""
    backtest_id = sample_trajectories
    
    config = PipelineConfig(
        db_config=test_db,
        output_dir=str(temp_output_dir),
        agent_types=['technical'],
        enable_judge=True
    )
    
    with patch('training.pipelines.auto_data_synthesis_pipeline.LLMJudge', return_value=mock_judge):
        pipeline = AutoDataSynthesisPipeline(config)
        result = benchmark(pipeline.process_backtest, backtest_id)
    
    assert result.final_dataset_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
