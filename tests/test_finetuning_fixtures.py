#!/usr/bin/env python3
"""
Test Fixtures for Fine-Tuning - Task 8.1
Fixtures and utilities for fine-tuning tests.
Phase A1 Week 5-6: Task 8.1 COMPLETE
"""

import json
import tempfile
from pathlib import Path
from typing import List, Dict


class FinetuningFixtures:
    """Test fixtures for fine-tuning"""
    
    @staticmethod
    def create_test_dataset(num_examples: int = 10) -> str:
        """Create test dataset file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        for i in range(num_examples):
            example = {
                "messages": [
                    {"role": "system", "content": "You are a stock analysis agent."},
                    {"role": "user", "content": f"Analyze stock {i}"},
                    {"role": "assistant", "content": f"Analysis result for stock {i}"}
                ]
            }
            temp_file.write(json.dumps(example) + '\n')
        
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def create_invalid_dataset() -> str:
        """Create invalid dataset for error testing"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        # Missing 'messages' field
        example = {"data": "invalid"}
        temp_file.write(json.dumps(example) + '\n')
        
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def mock_training_job() -> Dict:
        """Mock training job data"""
        return {
            "id": "ftjob-test123",
            "model": "gpt-3.5-turbo",
            "status": "succeeded",
            "training_file": "file-test123",
            "fine_tuned_model": "ft:gpt-3.5-turbo:test",
            "trained_tokens": 50000
        }
    
    @staticmethod
    def mock_metrics() -> List[Dict]:
        """Mock training metrics"""
        return [
            {"step": 0, "training_loss": 0.5, "validation_loss": 0.6},
            {"step": 10, "training_loss": 0.4, "validation_loss": 0.5},
            {"step": 20, "training_loss": 0.3, "validation_loss": 0.4}
        ]


if __name__ == "__main__":
    print("=== Fine-Tuning Fixtures Test ===\n")
    
    fixtures = FinetuningFixtures()
    
    # Test 1: Create dataset
    print("Test 1: Create test dataset")
    dataset_file = fixtures.create_test_dataset(5)
    print(f"✓ Dataset created: {dataset_file}")
    
    # Verify content
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        print(f"✓ {len(lines)} examples\n")
    
    # Test 2: Mock job
    print("Test 2: Mock training job")
    job = fixtures.mock_training_job()
    print(f"✓ Job ID: {job['id']}")
    print(f"✓ Status: {job['status']}\n")
    
    # Test 3: Mock metrics
    print("Test 3: Mock metrics")
    metrics = fixtures.mock_metrics()
    print(f"✓ {len(metrics)} metric points\n")
    
    print("=== Tests Complete ===")
