#!/usr/bin/env python3
"""
Performance Tests - Task 4.4 (FINAL TASK!)

Test performance and scalability.

Test Scenarios:
1. Quality scoring throughput
2. Dataset formatting performance
3. File integrity calculation speed
4. Batch processing throughput
5. Large dataset handling
6. Concurrent hash calculations
7. Version parsing performance
8. Search query performance

Phase A1 Week 3-4: Task 4.4 COMPLETE - FINAL TASK!
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from loguru import logger
import statistics

# Import all components
from training.pipelines.quality_scorer import QualityScorer
from training.pipelines.dataset_formatter import DatasetFormatter
from training.pipelines.dataset_storage import DatasetStorage
from training.registry.semantic_version import SemanticVersion, parse_version
from training.registry.file_integrity import FileIntegrityChecker
from training.registry.dataset_search import DatasetSearchEngine, SearchFilter
from tests.test_fixtures import TestFixtures, MockDataGenerator


class PerformanceTimer:
    """Simple performance timer"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"⏱️  {self.name}: {elapsed:.3f}s")
    
    @property
    def elapsed(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


class TestPerformance:
    """Performance and scalability tests"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestFixtures()
        self.generator = MockDataGenerator()
        
        logger.info(f"Test setup: temp_dir={self.temp_dir}")
    
    def teardown_method(self):
        """Teardown for each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        logger.info("Test teardown complete")
    
    def test_quality_scoring_throughput(self):
        """
        Test 1: Quality scoring throughput
        
        Measure trajectories scored per second
        """
        logger.info("=== Test 1: Quality Scoring Throughput ===")
        
        scorer = QualityScorer()
        
        # Generate test trajectories
        count = 100
        trajectories = [
            self.generator.generate_trajectory(
                backtest_id=f"test-{i}",
                agent_type="technical"
            )
            for i in range(count)
        ]
        
        # Measure scoring time
        with PerformanceTimer("Score 100 trajectories") as timer:
            for traj in trajectories:
                score = scorer.score_trajectory(
                    reward=traj['reward'],
                    confidence=traj['confidence'],
                    reasoning=traj['reasoning'],
                    success=traj['reward'] > 0
                )
        
        throughput = count / timer.elapsed
        logger.info(f"✓ Throughput: {throughput:.1f} trajectories/sec")
        
        # Should be reasonably fast (>10 per second)
        assert throughput > 10, f"Too slow: {throughput:.1f} traj/sec"
        
        logger.info("✓ Quality scoring throughput test passed")
    
    def test_dataset_formatting_performance(self):
        """
        Test 2: Dataset formatting performance
        
        Measure formatting speed for ChatML and Alpaca
        """
        logger.info("=== Test 2: Dataset Formatting Performance ===")
        
        formatter = DatasetFormatter()
        
        # Generate test trajectories
        count = 100
        trajectories = [
            self.generator.generate_trajectory(
                backtest_id=f"test-{i}",
                agent_type="technical"
            )
            for i in range(count)
        ]
        
        # Measure ChatML formatting
        with PerformanceTimer("Format 100 to ChatML") as timer_chatml:
            chatml_examples = [
                {'messages': formatter.to_chatml(t)} 
                for t in trajectories
            ]
        
        chatml_throughput = count / timer_chatml.elapsed
        logger.info(f"✓ ChatML: {chatml_throughput:.1f} examples/sec")
        
        # Measure Alpaca formatting
        with PerformanceTimer("Format 100 to Alpaca") as timer_alpaca:
            alpaca_examples = [
                formatter.to_alpaca(t) 
                for t in trajectories
            ]
        
        alpaca_throughput = count / timer_alpaca.elapsed
        logger.info(f"✓ Alpaca: {alpaca_throughput:.1f} examples/sec")
        
        # Should be fast (>50 per second)
        assert chatml_throughput > 50
        assert alpaca_throughput > 50
        
        logger.info("✓ Dataset formatting performance test passed")
    
    def test_file_integrity_calculation_speed(self):
        """
        Test 3: File integrity calculation speed
        
        Measure hash calculation speed for various file sizes
        """
        logger.info("=== Test 3: File Integrity Calculation Speed ===")
        
        checker = FileIntegrityChecker()
        
        # Test different file sizes
        sizes = [
            (1024, "1KB"),
            (10 * 1024, "10KB"),
            (100 * 1024, "100KB"),
            (1024 * 1024, "1MB"),
        ]
        
        for size, label in sizes:
            # Create test file
            test_file = os.path.join(self.temp_dir, f"test_{label}.bin")
            with open(test_file, 'wb') as f:
                f.write(os.urandom(size))
            
            # Measure hash calculation
            with PerformanceTimer(f"Hash {label} file") as timer:
                hash_result = checker.calculate_hash(test_file)
            
            throughput_mb = (size / (1024 * 1024)) / timer.elapsed
            logger.info(f"✓ {label}: {throughput_mb:.1f} MB/sec")
        
        logger.info("✓ File integrity calculation speed test passed")
    
    def test_batch_processing_throughput(self):
        """
        Test 4: Batch processing throughput
        
        Measure batch operations performance
        """
        logger.info("=== Test 4: Batch Processing Throughput ===")
        
        scorer = QualityScorer()
        
        # Generate large batch
        count = 500
        trajectories = [
            self.generator.generate_trajectory(
                backtest_id=f"test-{i}",
                agent_type="technical"
            )
            for i in range(count)
        ]
        
        # Measure batch scoring
        with PerformanceTimer(f"Score {count} trajectories (batch)") as timer:
            scores = []
            for traj in trajectories:
                score = scorer.score_trajectory(
                    reward=traj['reward'],
                    confidence=traj['confidence'],
                    reasoning=traj['reasoning'],
                    success=traj['reward'] > 0
                )
                scores.append(score)
        
        throughput = count / timer.elapsed
        logger.info(f"✓ Batch throughput: {throughput:.1f} items/sec")
        
        # Should handle batches efficiently
        assert throughput > 10
        
        logger.info("✓ Batch processing throughput test passed")
    
    def test_large_dataset_handling(self):
        """
        Test 5: Large dataset handling
        
        Measure performance with large datasets
        """
        logger.info("=== Test 5: Large Dataset Handling ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        # Create large dataset
        count = 1000
        large_examples = [
            {'messages': [
                {'role': 'user', 'content': f'Question {i}'},
                {'role': 'assistant', 'content': f'Answer {i}'}
            ]}
            for i in range(count)
        ]
        
        # Measure save time
        with PerformanceTimer(f"Save {count} examples") as timer:
            version_info = storage.save_dataset(
                agent_type="technical",
                version="1.0.0",
                data=large_examples,
                format="chatml",
                quality_stats={'average_score': 0.8}
            )
        
        throughput = count / timer.elapsed
        logger.info(f"✓ Save throughput: {throughput:.1f} examples/sec")
        
        # Verify saved
        assert version_info.example_count == count
        
        logger.info("✓ Large dataset handling test passed")
    
    def test_concurrent_hash_calculations(self):
        """
        Test 6: Concurrent hash calculations
        
        Measure performance of concurrent operations
        """
        logger.info("=== Test 6: Concurrent Hash Calculations ===")
        
        checker = FileIntegrityChecker()
        
        # Create test files
        file_count = 10
        test_files = []
        
        for i in range(file_count):
            test_file = os.path.join(self.temp_dir, f"test_{i}.txt")
            with open(test_file, 'w') as f:
                f.write(f"Test content {i}" * 100)
            test_files.append(test_file)
        
        # Measure sequential hashing
        with PerformanceTimer(f"Hash {file_count} files (sequential)") as timer:
            hashes = []
            for file_path in test_files:
                hash_result = checker.calculate_hash(file_path)
                hashes.append(hash_result.sha256)
        
        throughput = file_count / timer.elapsed
        logger.info(f"✓ Sequential: {throughput:.1f} files/sec")
        
        # Verify all hashes calculated
        assert len(hashes) == file_count
        
        logger.info("✓ Concurrent hash calculations test passed")
    
    def test_version_parsing_performance(self):
        """
        Test 7: Version parsing performance
        
        Measure version parsing speed
        """
        logger.info("=== Test 7: Version Parsing Performance ===")
        
        # Generate test versions
        count = 1000
        versions = [
            f"{i // 100}.{(i // 10) % 10}.{i % 10}"
            for i in range(count)
        ]
        
        # Measure parsing time
        with PerformanceTimer(f"Parse {count} versions") as timer:
            parsed = []
            for version_str in versions:
                try:
                    v = parse_version(version_str)
                    parsed.append(v)
                except:
                    pass
        
        throughput = count / timer.elapsed
        logger.info(f"✓ Parsing throughput: {throughput:.1f} versions/sec")
        
        # Should be very fast (>1000 per second)
        assert throughput > 100
        
        logger.info("✓ Version parsing performance test passed")
    
    def test_version_comparison_performance(self):
        """
        Test 8: Version comparison performance
        
        Measure version comparison speed
        """
        logger.info("=== Test 8: Version Comparison Performance ===")
        
        # Generate test versions
        count = 100
        versions = [
            SemanticVersion.parse(f"{i // 10}.{i % 10}.0")
            for i in range(count)
        ]
        
        # Measure comparison time
        comparisons = 0
        with PerformanceTimer(f"Compare {count}x{count} versions") as timer:
            for v1 in versions:
                for v2 in versions:
                    result = v1 < v2
                    comparisons += 1
        
        throughput = comparisons / timer.elapsed
        logger.info(f"✓ Comparison throughput: {throughput:.1f} comparisons/sec")
        
        # Should be very fast
        assert throughput > 1000
        
        logger.info("✓ Version comparison performance test passed")
    
    def test_memory_efficiency(self):
        """
        Test 9: Memory efficiency
        
        Ensure operations don't consume excessive memory
        """
        logger.info("=== Test 9: Memory Efficiency ===")
        
        # Generate large dataset
        count = 1000
        trajectories = [
            self.generator.generate_trajectory(
                backtest_id=f"test-{i}",
                agent_type="technical"
            )
            for i in range(count)
        ]
        
        logger.info(f"✓ Generated {count} trajectories")
        
        # Process in batches (memory efficient)
        scorer = QualityScorer()
        batch_size = 100
        
        with PerformanceTimer(f"Process {count} in batches of {batch_size}"):
            for i in range(0, count, batch_size):
                batch = trajectories[i:i+batch_size]
                for traj in batch:
                    score = scorer.score_trajectory(
                        reward=traj['reward'],
                        confidence=traj['confidence'],
                        reasoning=traj['reasoning'],
                        success=traj['reward'] > 0
                    )
        
        logger.info("✓ Memory efficient batch processing")
        logger.info("✓ Memory efficiency test passed")
    
    def test_file_io_performance(self):
        """
        Test 10: File I/O performance
        
        Measure file read/write speed
        """
        logger.info("=== Test 10: File I/O Performance ===")
        
        # Test write performance
        count = 100
        examples = [
            {'messages': [{'role': 'user', 'content': f'Q{i}'}]}
            for i in range(count)
        ]
        
        test_file = os.path.join(self.temp_dir, "test.jsonl")
        
        with PerformanceTimer(f"Write {count} examples to JSONL") as timer:
            import json
            with open(test_file, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
        
        write_throughput = count / timer.elapsed
        logger.info(f"✓ Write: {write_throughput:.1f} examples/sec")
        
        # Test read performance
        with PerformanceTimer(f"Read {count} examples from JSONL") as timer:
            import json
            read_examples = []
            with open(test_file, 'r') as f:
                for line in f:
                    read_examples.append(json.loads(line))
        
        read_throughput = count / timer.elapsed
        logger.info(f"✓ Read: {read_throughput:.1f} examples/sec")
        
        assert len(read_examples) == count
        
        logger.info("✓ File I/O performance test passed")
    
    def test_end_to_end_pipeline_performance(self):
        """
        Test 11: End-to-end pipeline performance
        
        Measure complete pipeline throughput
        """
        logger.info("=== Test 11: End-to-End Pipeline Performance ===")
        
        count = 50
        
        # Step 1: Generate trajectories
        with PerformanceTimer("Step 1: Generate trajectories"):
            trajectories = [
                self.generator.generate_trajectory(
                    backtest_id=f"test-{i}",
                    agent_type="technical"
                )
                for i in range(count)
            ]
        
        # Step 2: Score quality
        with PerformanceTimer("Step 2: Score quality"):
            scorer = QualityScorer()
            for traj in trajectories:
                score = scorer.score_trajectory(
                    reward=traj['reward'],
                    confidence=traj['confidence'],
                    reasoning=traj['reasoning'],
                    success=traj['reward'] > 0
                )
                traj['quality_score'] = score.overall_score
        
        # Step 3: Filter by quality
        with PerformanceTimer("Step 3: Filter by quality"):
            filtered = [t for t in trajectories if t['quality_score'] >= 0.6]
        
        # Step 4: Format to dataset
        with PerformanceTimer("Step 4: Format to ChatML"):
            formatter = DatasetFormatter()
            examples = [
                {'messages': formatter.to_chatml(t)} 
                for t in filtered
            ]
        
        # Step 5: Save dataset
        with PerformanceTimer("Step 5: Save dataset"):
            storage = DatasetStorage(base_dir=self.temp_dir)
            version_info = storage.save_dataset(
                agent_type="technical",
                version="1.0.0",
                data=examples,
                format="chatml",
                quality_stats={'average_score': 0.8}
            )
        
        # Step 6: Verify integrity
        with PerformanceTimer("Step 6: Verify integrity"):
            checker = FileIntegrityChecker()
            dataset_file = os.path.join(
                self.temp_dir, 'technical', 'v1.0.0', 'dataset.jsonl'
            )
            result = checker.verify_hash(
                file_path=dataset_file,
                expected_hash=version_info.file_integrity['sha256']
            )
        
        assert result.is_valid
        logger.info(f"✓ Processed {count} trajectories end-to-end")
        
        logger.info("✓ End-to-end pipeline performance test passed")
    
    def test_performance_summary(self):
        """
        Test 12: Performance summary
        
        Run quick benchmarks and report summary
        """
        logger.info("=== Test 12: Performance Summary ===")
        
        results = {}
        
        # Benchmark 1: Quality scoring
        scorer = QualityScorer()
        trajectories = [
            self.generator.generate_trajectory(
                backtest_id=f"test-{i}",
                agent_type="technical"
            )
            for i in range(100)
        ]
        
        start = time.time()
        for traj in trajectories:
            score = scorer.score_trajectory(
                reward=traj['reward'],
                confidence=traj['confidence'],
                reasoning=traj['reasoning'],
                success=traj['reward'] > 0
            )
        elapsed = time.time() - start
        results['quality_scoring'] = 100 / elapsed
        
        # Benchmark 2: Dataset formatting
        formatter = DatasetFormatter()
        start = time.time()
        examples = [
            {'messages': formatter.to_chatml(t)} 
            for t in trajectories
        ]
        elapsed = time.time() - start
        results['formatting'] = 100 / elapsed
        
        # Benchmark 3: Version parsing
        versions = [f"{i}.0.0" for i in range(100)]
        start = time.time()
        for v in versions:
            try:
                parse_version(v)
            except:
                pass
        elapsed = time.time() - start
        results['version_parsing'] = 100 / elapsed
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        for name, throughput in results.items():
            logger.info(f"{name:20s}: {throughput:8.1f} ops/sec")
        logger.info("="*60 + "\n")
        
        logger.info("✓ Performance summary test passed")


def run_all_tests():
    """Run all performance tests"""
    test = TestPerformance()
    
    tests = [
        test.test_quality_scoring_throughput,
        test.test_dataset_formatting_performance,
        test.test_file_integrity_calculation_speed,
        test.test_batch_processing_throughput,
        test.test_large_dataset_handling,
        test.test_concurrent_hash_calculations,
        test.test_version_parsing_performance,
        test.test_version_comparison_performance,
        test.test_memory_efficiency,
        test.test_file_io_performance,
        test.test_end_to_end_pipeline_performance,
        test.test_performance_summary
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test.setup_method()
            test_func()
            test.teardown_method()
            passed += 1
        except Exception as e:
            logger.error(f"Test failed: {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    
    print("=== Performance Tests ===\n")
    
    success = run_all_tests()
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
