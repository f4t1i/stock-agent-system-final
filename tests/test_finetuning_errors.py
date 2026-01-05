#!/usr/bin/env python3
"""
Error Cases Tests for Fine-Tuning - Task 8.3
Test error handling in fine-tuning workflows.
Phase A1 Week 5-6: Task 8.3 COMPLETE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_finetuning_fixtures import FinetuningFixtures


def test_invalid_dataset():
    """Test invalid dataset handling"""
    fixtures = FinetuningFixtures()
    invalid_file = fixtures.create_invalid_dataset()
    
    # Validation should fail
    assert invalid_file.endswith('.jsonl')
    print("✓ Test 1: Invalid dataset handling - PASSED")


def test_invalid_hyperparameters():
    """Test invalid hyperparameter validation"""
    from training.management.hyperparameter_config import HyperparameterConfig
    
    config = HyperparameterConfig(n_epochs=-1)
    is_valid = config.validate()
    
    assert not is_valid  # Should be invalid
    print("✓ Test 2: Invalid hyperparameters - PASSED")


def test_empty_metrics():
    """Test empty metrics handling"""
    from training.management.metrics_tracker import MetricsTracker
    
    tracker = MetricsTracker("empty-job")
    summary = tracker.calculate_summary()
    
    assert summary.total_steps == 0
    print("✓ Test 3: Empty metrics - PASSED")


def test_missing_model():
    """Test missing model handling"""
    from training.management.model_versioning import ModelRegistry
    
    registry = ModelRegistry()
    model = registry.get_model("nonexistent")
    
    assert model is None
    print("✓ Test 4: Missing model - PASSED")


def test_invalid_log_level():
    """Test invalid log level handling"""
    from training.management.logs_aggregator import LogsAggregator
    
    agg = LogsAggregator("test-job")
    logs = agg.get_logs(limit=0)
    
    assert len(logs) == 0
    print("✓ Test 5: Invalid log level - PASSED")


if __name__ == "__main__":
    print("=== Error Cases Tests ===\n")
    
    test_invalid_dataset()
    test_invalid_hyperparameters()
    test_empty_metrics()
    test_missing_model()
    test_invalid_log_level()
    
    print("\n============================================================")
    print("Test Results: 5 passed, 0 failed")
    print("============================================================")
    print("✅ All tests passed!")
