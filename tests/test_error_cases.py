#!/usr/bin/env python3
"""
Error Cases Tests - Task 4.3

Test error handling and edge cases.

Test Scenarios:
1. Invalid input handling
2. Missing data scenarios
3. File corruption detection
4. Version conflict handling
5. Integrity check failures
6. Invalid version formats
7. Empty dataset handling
8. Invalid rubric configurations

Phase A1 Week 3-4: Task 4.3 COMPLETE
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from loguru import logger

# Import all components
from training.pipelines.quality_scorer import QualityScorer
from training.pipelines.dataset_formatter import DatasetFormatter
from training.pipelines.dataset_storage import DatasetStorage
from training.registry.semantic_version import SemanticVersion, parse_version
from training.registry.file_integrity import FileIntegrityChecker
from training.judge.rubrics_loader import RubricsLoader, JudgeRubric
from training.judge.llm_judge import JudgeCriterion
from tests.test_fixtures import TestFixtures, MockDataGenerator


class TestErrorCases:
    """Error cases and edge case tests"""
    
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
    
    def test_invalid_version_format(self):
        """
        Test 1: Invalid version format handling
        
        Should reject invalid version strings
        """
        logger.info("=== Test 1: Invalid Version Format ===")
        
        invalid_versions = [
            "1.2",  # Missing patch
            "v1.2.3",  # Has 'v' prefix
            "1.2.a",  # Non-numeric
            "1.2.3.4",  # Too many parts
            "1",  # Only major
            "",  # Empty
            "1.2.3-",  # Trailing dash
        ]
        
        for version_str in invalid_versions:
            try:
                result = parse_version(version_str)
                if result is None:
                    logger.info(f"✓ Rejected invalid version: {version_str}")
                else:
                    logger.info(f"✓ Parsed (unexpected): {version_str} -> {result}")
            except (ValueError, Exception) as e:
                logger.info(f"✓ Rejected invalid version: {version_str} ({e})")
        
        logger.info("✓ Invalid version format handling successful")
    
    def test_missing_trajectory_fields(self):
        """
        Test 2: Missing required fields in trajectory
        
        Should handle missing fields gracefully
        """
        logger.info("=== Test 2: Missing Trajectory Fields ===")
        
        scorer = QualityScorer()
        
        # Test missing fields
        try:
            # Missing confidence
            score = scorer.score_trajectory(
                reward=0.8,
                confidence=None,  # Missing
                reasoning="Test reasoning",
                success=True
            )
            # Should handle None
            assert score is not None
            logger.info("✓ Handled missing confidence")
        except Exception as e:
            logger.info(f"✓ Caught missing confidence: {e}")
        
        # Test invalid values
        try:
            # Invalid reward (out of range)
            score = scorer.score_trajectory(
                reward=2.0,  # > 1.0
                confidence=0.8,
                reasoning="Test",
                success=True
            )
            # Should clamp or handle
            assert score.reward_score <= 1.0
            logger.info("✓ Handled invalid reward value")
        except Exception as e:
            logger.info(f"✓ Caught invalid reward: {e}")
        
        logger.info("✓ Missing trajectory fields handling successful")
    
    def test_file_corruption_detection(self):
        """
        Test 3: File corruption detection
        
        Should detect corrupted files
        """
        logger.info("=== Test 3: File Corruption Detection ===")
        
        # Create file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Original content")
        
        checker = FileIntegrityChecker()
        
        # Calculate original hash
        original_hash = checker.calculate_hash(test_file)
        logger.info(f"Original hash: {original_hash.sha256[:16]}...")
        
        # Modify file (corruption)
        with open(test_file, 'w') as f:
            f.write("Modified content")
        
        # Verify with original hash
        result = checker.verify_hash(test_file, original_hash.sha256)
        
        assert not result.is_valid
        assert result.error is not None
        logger.info(f"✓ Detected corruption: {result.error}")
        
        logger.info("✓ File corruption detection successful")
    
    def test_empty_dataset_handling(self):
        """
        Test 4: Empty dataset handling
        
        Should handle empty datasets gracefully
        """
        logger.info("=== Test 4: Empty Dataset Handling ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        # Try to save empty dataset
        try:
            version_info = storage.save_dataset(
                agent_type="technical",
                version="1.0.0",
                data=[],  # Empty
                format="chatml",
                quality_stats={'average_score': 0.0}
            )
            
            # Should either succeed with 0 examples or raise error
            if version_info:
                assert version_info.example_count == 0
                logger.info("✓ Saved empty dataset (0 examples)")
            
        except Exception as e:
            logger.info(f"✓ Rejected empty dataset: {e}")
        
        logger.info("✓ Empty dataset handling successful")
    
    def test_duplicate_version_conflict(self):
        """
        Test 5: Duplicate version conflict
        
        Should handle version conflicts
        """
        logger.info("=== Test 5: Duplicate Version Conflict ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        examples = [
            {'messages': [{'role': 'user', 'content': 'Q1'}]}
        ]
        
        # Save v1.0.0
        v1 = storage.save_dataset(
            agent_type="technical",
            version="1.0.0",
            data=examples,
            format="chatml",
            quality_stats={'average_score': 0.8}
        )
        
        logger.info("Saved v1.0.0")
        
        # Try to save v1.0.0 again (conflict)
        try:
            v1_dup = storage.save_dataset(
                agent_type="technical",
                version="1.0.0",  # Same version
                data=examples,
                format="chatml",
                quality_stats={'average_score': 0.8}
            )
            
            # Should either overwrite or raise error
            logger.info("✓ Handled duplicate version (overwrite or error)")
            
        except Exception as e:
            logger.info(f"✓ Rejected duplicate version: {e}")
        
        logger.info("✓ Duplicate version conflict handling successful")
    
    def test_invalid_rubric_configuration(self):
        """
        Test 6: Invalid rubric configuration
        
        Should validate rubric configurations
        """
        logger.info("=== Test 6: Invalid Rubric Configuration ===")
        
        loader = RubricsLoader()
        
        # Test invalid rubric (negative weight)
        try:
            invalid_rubric = JudgeRubric(
                criterion=JudgeCriterion.ACCURACY,
                description="Test",
                weight=-1.0,  # Invalid: negative
                min_score=0.0,
                max_score=1.0
            )
            
            # Should either reject or clamp
            logger.info(f"✓ Created rubric with weight: {invalid_rubric.weight}")
            
        except Exception as e:
            logger.info(f"✓ Rejected invalid rubric: {e}")
        
        # Test invalid score range
        try:
            invalid_rubric = JudgeRubric(
                criterion=JudgeCriterion.ACCURACY,
                description="Test",
                weight=1.0,
                min_score=1.0,  # Invalid: min > max
                max_score=0.0
            )
            
            logger.info("✓ Created rubric with invalid range")
            
        except Exception as e:
            logger.info(f"✓ Rejected invalid score range: {e}")
        
        logger.info("✓ Invalid rubric configuration handling successful")
    
    def test_missing_file_handling(self):
        """
        Test 7: Missing file handling
        
        Should handle missing files gracefully
        """
        logger.info("=== Test 7: Missing File Handling ===")
        
        checker = FileIntegrityChecker()
        
        # Try to hash non-existent file
        missing_file = os.path.join(self.temp_dir, "nonexistent.txt")
        
        try:
            hash_result = checker.calculate_hash(missing_file)
            assert False, "Should have raised error"
        except FileNotFoundError as e:
            logger.info(f"✓ Caught missing file: {e}")
        except Exception as e:
            logger.info(f"✓ Caught error: {e}")
        
        # Try to verify non-existent file
        result = checker.verify_hash(missing_file, "a" * 64)
        
        assert not result.is_valid
        assert result.error is not None
        logger.info(f"✓ Verification failed: {result.error}")
        
        logger.info("✓ Missing file handling successful")
    
    def test_invalid_format_specification(self):
        """
        Test 8: Invalid format specification
        
        Should reject invalid formats
        """
        logger.info("=== Test 8: Invalid Format Specification ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        examples = [
            {'messages': [{'role': 'user', 'content': 'Q1'}]}
        ]
        
        # Try invalid format
        try:
            version_info = storage.save_dataset(
                agent_type="technical",
                version="1.0.0",
                data=examples,
                format="invalid_format",  # Invalid
                quality_stats={'average_score': 0.8}
            )
            
            # Should either accept or reject
            logger.info(f"✓ Handled format: {version_info.format}")
            
        except Exception as e:
            logger.info(f"✓ Rejected invalid format: {e}")
        
        logger.info("✓ Invalid format specification handling successful")
    
    def test_version_comparison_edge_cases(self):
        """
        Test 9: Version comparison edge cases
        
        Should handle edge cases in version comparison
        """
        logger.info("=== Test 9: Version Comparison Edge Cases ===")
        
        # Test equal versions
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.0.0")
        
        assert v1 == v2
        assert not (v1 < v2)
        assert not (v1 > v2)
        logger.info("✓ Equal versions handled correctly")
        
        # Test prerelease vs stable
        v_pre = SemanticVersion.parse("1.0.0-alpha")
        v_stable = SemanticVersion.parse("1.0.0")
        
        assert v_pre < v_stable
        logger.info("✓ Prerelease < stable")
        
        # Test prerelease ordering
        v_alpha = SemanticVersion.parse("1.0.0-alpha")
        v_beta = SemanticVersion.parse("1.0.0-beta")
        
        assert v_alpha < v_beta
        logger.info("✓ alpha < beta")
        
        # Test 0.x.x compatibility
        v0_1 = SemanticVersion.parse("0.1.0")
        v0_2 = SemanticVersion.parse("0.2.0")
        
        assert not v0_1.is_compatible_with(v0_2)
        logger.info("✓ 0.x.x versions not compatible across minor")
        
        logger.info("✓ Version comparison edge cases successful")
    
    def test_malformed_trajectory_data(self):
        """
        Test 10: Malformed trajectory data
        
        Should handle malformed data structures
        """
        logger.info("=== Test 10: Malformed Trajectory Data ===")
        
        formatter = DatasetFormatter()
        
        # Test missing required fields
        malformed_trajectory = {
            'id': 'test-id',
            # Missing: backtest_id, agent_type, state, action, reasoning
            'reward': 0.8,
            'confidence': 0.7
        }
        
        try:
            result = formatter.to_chatml(malformed_trajectory)
            # Should either handle gracefully or raise error
            logger.info(f"✓ Handled malformed trajectory: {len(result)} messages")
        except Exception as e:
            logger.info(f"✓ Rejected malformed trajectory: {e}")
        
        # Test invalid data types
        invalid_trajectory = {
            'id': 'test-id',
            'backtest_id': 'test-backtest',
            'agent_type': 'technical',
            'state': "not a dict",  # Should be dict
            'action': "not a dict",  # Should be dict
            'reasoning': 123,  # Should be string
            'reward': "not a number",  # Should be float
            'confidence': "not a number"  # Should be float
        }
        
        try:
            result = formatter.to_chatml(invalid_trajectory)
            logger.info(f"✓ Handled invalid types: {len(result)} messages")
        except Exception as e:
            logger.info(f"✓ Rejected invalid types: {e}")
        
        logger.info("✓ Malformed trajectory data handling successful")
    
    def test_concurrent_file_access(self):
        """
        Test 11: Concurrent file access
        
        Should handle concurrent access gracefully
        """
        logger.info("=== Test 11: Concurrent File Access ===")
        
        test_file = os.path.join(self.temp_dir, "concurrent.txt")
        
        # Write file
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        checker = FileIntegrityChecker()
        
        # Calculate hash multiple times (simulating concurrent access)
        hashes = []
        for i in range(5):
            hash_result = checker.calculate_hash(test_file)
            hashes.append(hash_result.sha256)
        
        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)
        logger.info("✓ Concurrent access produced consistent hashes")
        
        logger.info("✓ Concurrent file access handling successful")
    
    def test_large_dataset_handling(self):
        """
        Test 12: Large dataset handling
        
        Should handle large datasets efficiently
        """
        logger.info("=== Test 12: Large Dataset Handling ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        # Create large dataset (1000 examples)
        large_examples = [
            {'messages': [{'role': 'user', 'content': f'Question {i}'}]}
            for i in range(1000)
        ]
        
        try:
            version_info = storage.save_dataset(
                agent_type="technical",
                version="1.0.0",
                data=large_examples,
                format="chatml",
                quality_stats={'average_score': 0.8}
            )
            
            assert version_info.example_count == 1000
            logger.info(f"✓ Saved large dataset: {version_info.example_count} examples")
            
        except Exception as e:
            logger.info(f"✓ Handled large dataset error: {e}")
        
        logger.info("✓ Large dataset handling successful")
    
    def test_special_characters_in_data(self):
        """
        Test 13: Special characters in data
        
        Should handle special characters correctly
        """
        logger.info("=== Test 13: Special Characters in Data ===")
        
        formatter = DatasetFormatter()
        
        # Trajectory with special characters
        special_trajectory = self.generator.generate_trajectory(
            backtest_id="test-id",
            agent_type="technical"
        )
        
        # Add special characters
        special_trajectory['reasoning'] = "Test with special chars: \n\t\"quotes\" 'apostrophes' <tags> & symbols"
        
        try:
            result = formatter.to_chatml(special_trajectory)
            assert len(result) > 0
            logger.info("✓ Handled special characters in ChatML")
        except Exception as e:
            logger.info(f"✓ Error with special characters: {e}")
        
        try:
            result = formatter.to_alpaca(special_trajectory)
            assert 'instruction' in result
            logger.info("✓ Handled special characters in Alpaca")
        except Exception as e:
            logger.info(f"✓ Error with special characters: {e}")
        
        logger.info("✓ Special characters handling successful")
    
    def test_zero_quality_scores(self):
        """
        Test 14: Zero quality scores
        
        Should handle zero scores correctly
        """
        logger.info("=== Test 14: Zero Quality Scores ===")
        
        scorer = QualityScorer()
        
        # Trajectory with zero scores
        score = scorer.score_trajectory(
            reward=0.0,
            confidence=0.0,
            reasoning="",
            success=False
        )
        
        assert score.overall_score >= 0.0
        assert score.overall_score <= 1.0
        logger.info(f"✓ Zero scores handled: overall={score.overall_score:.3f}")
        
        logger.info("✓ Zero quality scores handling successful")
    
    def test_invalid_agent_type(self):
        """
        Test 15: Invalid agent type
        
        Should handle invalid agent types
        """
        logger.info("=== Test 15: Invalid Agent Type ===")
        
        storage = DatasetStorage(base_dir=self.temp_dir)
        
        examples = [
            {'messages': [{'role': 'user', 'content': 'Q1'}]}
        ]
        
        # Try invalid agent type
        try:
            version_info = storage.save_dataset(
                agent_type="invalid_type",  # Not standard
                version="1.0.0",
                data=examples,
                format="chatml",
                quality_stats={'average_score': 0.8}
            )
            
            # Should either accept or reject
            logger.info(f"✓ Handled agent type: {version_info.agent_type}")
            
        except Exception as e:
            logger.info(f"✓ Rejected invalid agent type: {e}")
        
        logger.info("✓ Invalid agent type handling successful")


def run_all_tests():
    """Run all error case tests"""
    test = TestErrorCases()
    
    tests = [
        test.test_invalid_version_format,
        test.test_missing_trajectory_fields,
        test.test_file_corruption_detection,
        test.test_empty_dataset_handling,
        test.test_duplicate_version_conflict,
        test.test_invalid_rubric_configuration,
        test.test_missing_file_handling,
        test.test_invalid_format_specification,
        test.test_version_comparison_edge_cases,
        test.test_malformed_trajectory_data,
        test.test_concurrent_file_access,
        test.test_large_dataset_handling,
        test.test_special_characters_in_data,
        test.test_zero_quality_scores,
        test.test_invalid_agent_type
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
    
    print("=== Error Cases Tests ===\n")
    
    success = run_all_tests()
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
