#!/usr/bin/env python3
"""
Performance Tests for Fine-Tuning - Task 8.4
Test performance of fine-tuning operations.
Phase A1 Week 5-6: Task 8.4 COMPLETE - FINAL TASK!
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_finetuning_fixtures import FinetuningFixtures


def test_dataset_creation_performance():
    """Test dataset creation performance"""
    start = time.time()
    
    fixtures = FinetuningFixtures()
    dataset_file = fixtures.create_test_dataset(1000)
    
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # Should complete in < 5 seconds
    print(f"✓ Test 1: Dataset creation (1000 examples) - {elapsed:.3f}s - PASSED")


def test_metrics_aggregation_performance():
    """Test metrics aggregation performance"""
    from training.management.metrics_tracker import MetricsTracker, MetricPoint
    
    start = time.time()
    
    tracker = MetricsTracker("perf-test")
    for i in range(1000):
        tracker.add_metric(MetricPoint(
            step=i,
            training_loss=0.5 - i * 0.0001,
            validation_loss=0.6 - i * 0.0001
        ))
    
    summary = tracker.calculate_summary()
    
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in < 1 second
    assert summary.total_steps == 1000
    print(f"✓ Test 2: Metrics aggregation (1000 points) - {elapsed:.3f}s - PASSED")


def test_model_registry_performance():
    """Test model registry performance"""
    from training.management.model_versioning import ModelRegistry, ModelVersion
    
    start = time.time()
    
    registry = ModelRegistry()
    for i in range(100):
        model = ModelVersion(
            model_id=f"model-{i}",
            model_name=f"test-model-{i}",
            version="1.0.0",
            base_model="gpt-3.5-turbo",
            provider="openai",
            job_id=f"job-{i}"
        )
        registry.register_model(model)
    
    models = registry.list_models()
    
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in < 1 second
    assert len(models) == 100
    print(f"✓ Test 3: Model registry (100 models) - {elapsed:.3f}s - PASSED")


def test_logs_aggregation_performance():
    """Test logs aggregation performance"""
    from training.management.logs_aggregator import LogsAggregator, LogEntry, LogLevel
    from datetime import datetime
    
    start = time.time()
    
    agg = LogsAggregator("perf-test")
    for i in range(1000):
        agg.add_log(LogEntry(
            datetime.now(),
            LogLevel.INFO,
            f"Log message {i}",
            "perf-test"
        ))
    
    summary = agg.calculate_summary()
    
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in < 1 second
    assert summary.total_entries == 1000
    print(f"✓ Test 4: Logs aggregation (1000 entries) - {elapsed:.3f}s - PASSED")


def test_hyperparameter_suggestion_performance():
    """Test hyperparameter suggestion performance"""
    from training.management.hyperparameter_config import HyperparameterManager
    
    start = time.time()
    
    manager = HyperparameterManager()
    for i in range(100):
        config = manager.suggest_config(
            dataset_size=500 + i * 10,
            task_complexity='medium'
        )
    
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in < 1 second
    print(f"✓ Test 5: Hyperparameter suggestions (100 configs) - {elapsed:.3f}s - PASSED")


if __name__ == "__main__":
    print("=== Performance Tests ===\n")
    
    test_dataset_creation_performance()
    test_metrics_aggregation_performance()
    test_model_registry_performance()
    test_logs_aggregation_performance()
    test_hyperparameter_suggestion_performance()
    
    print("\n============================================================")
    print("Test Results: 5 passed, 0 failed")
    print("============================================================")
    print("✅ All tests passed!")
    print("\n🎉🎉🎉 PHASE A1 WEEK 5-6 COMPLETE! 🎉🎉🎉")
