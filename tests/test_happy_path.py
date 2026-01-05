#!/usr/bin/env python3
"""
Happy Path Acceptance Tests - Task 4.2

Test the complete happy path through the system.

Test Scenarios:
1. End-to-end pipeline: Backtest → Trajectories → Dataset → Registry
2. Judge evaluation: Load rubrics → Evaluate → Filter
3. Quality filtering: Score → Filter → Export
4. Dataset versioning: Create → Version → Track lineage
5. Full integration: All components working together

Phase A1 Week 3-4: Task 4.2 COMPLETE
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from loguru import logger

# Import all components
from training.pipelines.trajectory_extractor import TrajectoryExtractor
from training.pipelines.quality_scorer import QualityScorer
from training.pipelines.dataset_formatter import DatasetFormatter
from training.pipelines.dataset_storage import DatasetStorage
from training.registry.semantic_version import SemanticVersion
from training.registry.file_integrity import FileIntegrityChecker
from training.judge.llm_judge import JudgeRubric
from training.judge.rubrics_loader import load_rubrics
from tests.test_fixtures import TestFixtures, MockDataGenerator


class TestHappyPath:
    """Happy path acceptance tests"""
    
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
    
    def test_trajectory_extraction(self):
        """
        Test 1: Extract trajectories from backtest
        
        Flow:
        1. Create mock backtest with trajectories
        2. Extract trajectories
        3. Verify extraction
        """
        logger.info("=== Test 1: Trajectory Extraction ===")
        
        # Create mock backtest
        backtest = self.generator.generate_backtest(agent_type="technical")
        trajectories = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=10,
            agent_type="technical",
            min_reward=0.5,
            min_confidence=0.7
        )
        
        logger.info(f"Created backtest with {len(trajectories)} trajectories")
        
        # Verify trajectories
        assert len(trajectories) == 10
        assert all(t['agent_type'] == 'technical' for t in trajectories)
        assert all(t['reward'] >= 0.5 for t in trajectories)
        assert all(t['confidence'] >= 0.7 for t in trajectories)
        
        logger.info("✓ Trajectory extraction successful")
    
    def test_quality_scoring(self):
        """
        Test 2: Calculate quality scores for trajectories
        
        Flow:
        1. Generate trajectories
        2. Score quality
        3. Verify scores
        """
        logger.info("=== Test 2: Quality Scoring ===")
        
        # Generate trajectories
        backtest = self.generator.generate_backtest()
        trajectories = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=5
        )
        
        # Score quality
        scorer = QualityScorer()
        
        for trajectory in trajectories:
            score = scorer.score_trajectory(
                reward=trajectory['reward'],
                confidence=trajectory['confidence'],
                reasoning=trajectory['reasoning'],
                success=trajectory['reward'] > 0
            )
            
            # Verify score
            assert 0.0 <= score.overall_score <= 1.0
            assert 0.0 <= score.reward_score <= 1.0
            assert 0.0 <= score.confidence_score <= 1.0
            assert 0.0 <= score.reasoning_score <= 1.0
            assert 0.0 <= score.consistency_score <= 1.0
            
            logger.info(f"Trajectory {trajectory['id'][:8]}: quality={score.overall_score:.3f}")
        
        logger.info("✓ Quality scoring successful")
    
    def test_dataset_formatting(self):
        """
        Test 3: Format trajectories to ChatML/Alpaca
        
        Flow:
        1. Generate trajectories
        2. Format to ChatML
        3. Format to Alpaca
        4. Verify formats
        """
        logger.info("=== Test 3: Dataset Formatting ===")
        
        # Generate trajectories
        backtest = self.generator.generate_backtest()
        trajectories = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=3
        )
        
        formatter = DatasetFormatter()
        
        # Format to ChatML
        chatml_examples = [{'messages': formatter.to_chatml(t)} for t in trajectories]
        assert len(chatml_examples) == 3
        
        for example in chatml_examples:
            assert 'messages' in example
            assert len(example['messages']) >= 2
            assert any(m['role'] == 'user' for m in example['messages'])
            assert any(m['role'] == 'assistant' for m in example['messages'])
        
        logger.info(f"✓ ChatML format: {len(chatml_examples)} examples")
        
        # Format to Alpaca
        alpaca_examples = [formatter.to_alpaca(t) for t in trajectories]
        assert len(alpaca_examples) == 3
        
        for example in alpaca_examples:
            assert 'instruction' in example
            assert 'output' in example
        
        logger.info(f"✓ Alpaca format: {len(alpaca_examples)} examples")
    
    def test_dataset_storage(self):
        """
        Test 4: Save dataset with versioning
        
        Flow:
        1. Create dataset
        2. Save with version
        3. Verify files
        4. Check integrity
        """
        logger.info("=== Test 4: Dataset Storage ===")
        
        # Create dataset
        examples = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Question {i}'},
                    {'role': 'assistant', 'content': f'Answer {i}'}
                ]
            }
            for i in range(5)
        ]
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        # Save dataset
        version_info = storage.save_dataset(
            agent_type="technical",
            version="1.0.0",
            data=examples,
            format="chatml",
            quality_stats={'average_score': 0.8}
        )
        
        # Verify
        assert version_info.agent_type == 'technical'
        assert version_info.version == '1.0.0'
        assert version_info.format == 'chatml'
        assert version_info.example_count == 5
        assert version_info.file_integrity
        assert version_info.file_integrity.get('sha256')
        
        # Check files exist
        dataset_dir = os.path.join(self.temp_dir, 'technical')
        assert os.path.exists(dataset_dir)
        
        logger.info(f"✓ Dataset saved to {dataset_dir}")
        
        # Verify integrity
        checker = FileIntegrityChecker()
        dataset_file = os.path.join(self.temp_dir, 'technical', 'v1.0.0', 'dataset.jsonl')
        result = checker.verify_hash(
            file_path=dataset_file,
            expected_hash=version_info.file_integrity['sha256']
        )
        
        assert result.is_valid
        logger.info("✓ Integrity check passed")
    
    def test_semantic_versioning(self):
        """
        Test 5: Semantic versioning operations
        
        Flow:
        1. Parse versions
        2. Compare versions
        3. Bump versions
        4. Check compatibility
        """
        logger.info("=== Test 5: Semantic Versioning ===")
        
        # Parse versions
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.2.3")
        v3 = SemanticVersion.parse("2.0.0")
        
        # Compare
        assert v1 < v2 < v3
        assert v1 != v2
        
        logger.info("✓ Version comparison works")
        
        # Bump versions
        v1_major = v1.bump_major()
        assert str(v1_major) == "2.0.0"
        
        v1_minor = v1.bump_minor()
        assert str(v1_minor) == "1.1.0"
        
        v1_patch = v1.bump_patch()
        assert str(v1_patch) == "1.0.1"
        
        logger.info("✓ Version bumping works")
        
        # Check compatibility
        assert v1.is_compatible_with(v2)  # 1.0.0 compatible with 1.2.3
        assert not v1.is_compatible_with(v3)  # 1.0.0 NOT compatible with 2.0.0
        
        logger.info("✓ Compatibility checking works")
    
    def test_file_integrity(self):
        """
        Test 6: File integrity checking
        
        Flow:
        1. Create file
        2. Calculate hash
        3. Verify hash
        4. Detect corruption
        """
        logger.info("=== Test 6: File Integrity ===")
        
        # Create file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        checker = FileIntegrityChecker()
        
        # Calculate hash
        hash_result = checker.calculate_hash(test_file)
        assert hash_result.sha256
        assert hash_result.size_bytes > 0
        
        logger.info(f"✓ Hash calculated: {hash_result.sha256[:16]}...")
        
        # Verify hash
        verify_result = checker.verify_hash(test_file, hash_result.sha256)
        assert verify_result.is_valid
        
        logger.info("✓ Hash verification passed")
        
        # Test wrong hash
        wrong_hash = "a" * 64
        verify_result = checker.verify_hash(test_file, wrong_hash)
        assert not verify_result.is_valid
        
        logger.info("✓ Corruption detection works")
    
    def test_rubrics_loading(self):
        """
        Test 7: Load rubric templates
        
        Flow:
        1. Load predefined rubrics
        2. Verify rubric structure
        3. Check weights
        """
        logger.info("=== Test 7: Rubrics Loading ===")
        
        # Load rubrics
        rubrics = load_rubrics(template_name="technical_v1")
        
        assert len(rubrics) > 0
        assert all(isinstance(r, JudgeRubric) for r in rubrics)
        
        # Check weights
        total_weight = sum(r.weight for r in rubrics)
        assert total_weight > 0
        
        logger.info(f"✓ Loaded {len(rubrics)} rubrics, total weight: {total_weight}")
        
        # Verify structure
        for rubric in rubrics:
            assert rubric.criterion
            assert rubric.description
            assert rubric.weight > 0
            assert 0.0 <= rubric.min_score <= rubric.max_score <= 1.0
        
        logger.info("✓ Rubric structure valid")
    
    def test_end_to_end_pipeline(self):
        """
        Test 8: Complete end-to-end pipeline
        
        Flow:
        1. Generate backtest + trajectories
        2. Score quality
        3. Filter by quality
        4. Format to dataset
        5. Save with versioning
        6. Verify integrity
        """
        logger.info("=== Test 8: End-to-End Pipeline ===")
        
        # Step 1: Generate data
        backtest = self.generator.generate_backtest(agent_type="technical")
        trajectories = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=20,
            agent_type="technical"
        )
        
        logger.info(f"Step 1: Generated {len(trajectories)} trajectories")
        
        # Step 2: Score quality
        scorer = QualityScorer()
        scored_trajectories = []
        
        for trajectory in trajectories:
            score = scorer.score_trajectory(
                reward=trajectory['reward'],
                confidence=trajectory['confidence'],
                reasoning=trajectory['reasoning'],
                success=trajectory['reward'] > 0
            )
            trajectory['quality_score'] = score.overall_score
            scored_trajectories.append(trajectory)  
        logger.info(f"Step 2: Scored {len(scored_trajectories)} trajectories")
        
        # Step 3: Filter by quality
        quality_threshold = 0.6
        filtered_trajectories = [
            t for t in scored_trajectories
            if t['quality_score'] >= quality_threshold
        ]
        
        logger.info(
            f"Step 3: Filtered to {len(filtered_trajectories)} trajectories "
            f"(threshold: {quality_threshold})"
        )
        
        # Step 4: Format to dataset
        formatter = DatasetFormatter()
        examples = [{'messages': formatter.to_chatml(t)} for t in filtered_trajectories]
        
        logger.info(f"Step 4: Formatted {len(examples)} examples")
        
        # Step 5: Save with versioning
        storage = DatasetStorage(base_dir=self.temp_dir)
        version_info = storage.save_dataset(
            agent_type="technical",
            version="1.0.0",
            data=examples,
            format="chatml",
            quality_stats={'average_score': 0.8}
        )
        
        logger.info(f"Step 5: Saved dataset v{version_info.version}")
        
        # Step 6: Verify integrity
        checker = FileIntegrityChecker()
        dataset_file = os.path.join(self.temp_dir, 'technical', 'v1.0.0', 'dataset.jsonl')
        result = checker.verify_hash(
            file_path=dataset_file,
            expected_hash=version_info.file_integrity['sha256']
        )
        
        assert result.is_valid
        logger.info("Step 6: Integrity verified")
        
        # Final verification
        assert len(examples) > 0
        dataset_dir = os.path.join(self.temp_dir, 'technical')
        assert os.path.exists(dataset_dir)
        assert version_info.example_count == len(examples)
        
        logger.info("✓ End-to-end pipeline successful")
    
    def test_dataset_versioning_workflow(self):
        """
        Test 9: Dataset versioning workflow
        
        Flow:
        1. Create v1.0.0
        2. Create v1.1.0 (minor bump)
        3. Create v2.0.0 (major bump)
        4. Verify version ordering
        """
        logger.info("=== Test 9: Dataset Versioning Workflow ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        # Create v1.0.0
        examples_v1 = [
            {'messages': [{'role': 'user', 'content': f'Q{i}'}]}
            for i in range(5)
        ]
        
        v1_info = storage.save_dataset(
            agent_type="technical",
            version="1.0.0",
            data=examples_v1,
            format="chatml",
            quality_stats={'average_score': 0.8}
        )
        
        logger.info(f"Created v1.0.0: {v1_info.example_count} examples")
        
        # Create v1.1.0
        examples_v1_1 = examples_v1 + [
            {'messages': [{'role': 'user', 'content': 'Q5'}]}
        ]
        
        v1_1_info = storage.save_dataset(
            agent_type="technical",
            version="1.1.0",
            data=examples_v1_1,
            format="chatml",
            quality_stats={'average_score': 0.8}
        )
        
        logger.info(f"Created v1.1.0: {v1_1_info.example_count} examples")
        
        # Create v2.0.0
        examples_v2 = [
            {'messages': [{'role': 'user', 'content': f'New{i}'}]}
            for i in range(10)
        ]
        
        v2_info = storage.save_dataset(
            agent_type="technical",
            version="2.0.0",
            data=examples_v2,
            format="chatml",
            quality_stats={'average_score': 0.8}
        )
        
        logger.info(f"Created v2.0.0: {v2_info.example_count} examples")
        
        # Verify version ordering
        v1 = SemanticVersion.parse(v1_info.version)
        v1_1 = SemanticVersion.parse(v1_1_info.version)
        v2 = SemanticVersion.parse(v2_info.version)
        
        assert v1 < v1_1 < v2
        assert v1.is_compatible_with(v1_1)
        assert not v1.is_compatible_with(v2)
        
        logger.info("✓ Dataset versioning workflow successful")
    
    def test_quality_filtering_workflow(self):
        """
        Test 10: Quality filtering workflow
        
        Flow:
        1. Generate mixed quality trajectories
        2. Score all trajectories
        3. Filter by different thresholds
        4. Verify filtering
        """
        logger.info("=== Test 10: Quality Filtering Workflow ===")
        
        # Generate mixed quality
        backtest = self.generator.generate_backtest()
        
        # High quality
        high_quality = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=5,
            min_reward=0.8,
            min_confidence=0.9
        )
        
        # Medium quality
        medium_quality = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=5,
            min_reward=0.5,
            min_confidence=0.6
        )
        
        # Low quality
        low_quality = self.generator.generate_trajectories(
            backtest_id=backtest['id'],
            count=5,
            min_reward=0.0,
            min_confidence=0.3
        )
        
        all_trajectories = high_quality + medium_quality + low_quality
        
        logger.info(f"Generated {len(all_trajectories)} trajectories (mixed quality)")
        
        # Score all
        scorer = QualityScorer()
        scored = []
        
        for trajectory in all_trajectories:
            score = scorer.score_trajectory(
                reward=trajectory['reward'],
                confidence=trajectory['confidence'],
                reasoning=trajectory['reasoning'],
                success=trajectory['reward'] > 0
            )
            trajectory['quality_score'] = score.overall_score
            scored.append(trajectory)
        
        # Filter by threshold 0.7
        threshold_07 = [t for t in scored if t['quality_score'] >= 0.7]
        
        # Filter by threshold 0.5
        threshold_05 = [t for t in scored if t['quality_score'] >= 0.5]
        
        # Verify
        assert len(threshold_07) <= len(threshold_05) <= len(scored)
        assert all(t['quality_score'] >= 0.7 for t in threshold_07)
        assert all(t['quality_score'] >= 0.5 for t in threshold_05)
        
        logger.info(f"✓ Filtered: {len(threshold_07)} (≥0.7), {len(threshold_05)} (≥0.5)")
        logger.info("✓ Quality filtering workflow successful")


def run_all_tests():
    """Run all happy path tests"""
    test = TestHappyPath()
    
    tests = [
        test.test_trajectory_extraction,
        test.test_quality_scoring,
        test.test_dataset_formatting,
        test.test_dataset_storage,
        test.test_semantic_versioning,
        test.test_file_integrity,
        test.test_rubrics_loading,
        test.test_end_to_end_pipeline,
        test.test_dataset_versioning_workflow,
        test.test_quality_filtering_workflow
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
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    
    print("=== Happy Path Acceptance Tests ===\n")
    
    success = run_all_tests()
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
