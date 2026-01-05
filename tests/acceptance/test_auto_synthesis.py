"""
Acceptance Tests for Auto Data-Synthesis Pipeline

Tests the complete auto-synthesis workflow from backtest completion
to versioned SFT dataset generation.

Phase A1 Week 3-4: Task 4
- End-to-end pipeline testing
- Quality threshold validation
- Judge filtering integration
- Dataset registry integration
- Version management
- Error handling

Based on:
- TradingAgents acceptance tests
- MLflow integration tests
- Continuous learning pipeline tests
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from training.pipelines.auto_data_synthesis_pipeline import (
    AutoDataSynthesisPipeline,
    AutoSynthesisConfig,
    SynthesisResult
)
from training.dataset_registry import DatasetRegistry, DatasetVersion
from training.judge_filtering import JudgeApprovedFilter, JudgeFilterConfig
from data_pipeline.experience_library_postgres import Trajectory, ExperienceLibraryPostgres
from data_pipeline.data_synthesis import SFTExample
from judge.llm_judge import LLMJudge


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_experience_library():
    """Mock experience library with sample trajectories"""
    library = Mock(spec=ExperienceLibraryPostgres)
    
    # Create sample trajectories
    trajectories = []
    for i in range(10):
        traj = Trajectory(
            trajectory_id=f"traj_{i}",
            symbol="AAPL",
            agent_type="technical",
            market_state={"price": 150.0 + i},
            agent_inputs={"indicators": {"rsi": 50 + i}},
            reasoning=f"Technical analysis suggests {'bullish' if i % 2 == 0 else 'bearish'} trend",
            confidence=0.7 + (i * 0.02),
            recommendation="buy" if i % 2 == 0 else "sell",
            position_size=0.1,
            stop_loss=145.0,
            take_profit=160.0,
            success=i % 3 != 0,  # 66% success rate
            reward=0.5 + (i * 0.05) if i % 3 != 0 else -0.2,
            metadata={}
        )
        trajectories.append(traj)
    
    library.get_successful_trajectories.return_value = trajectories
    return library


@pytest.fixture
def mock_judge():
    """Mock LLM Judge"""
    judge = Mock(spec=LLMJudge)
    
    def mock_evaluate(agent_output, rubric_name, context=None):
        # Simulate judge evaluation
        confidence = agent_output.get('confidence', 0.5)
        return {
            'overall_score': confidence + 0.1,
            'dimension_scores': {
                'reasoning': confidence,
                'accuracy': confidence + 0.05,
                'confidence_calibration': confidence - 0.05
            },
            'feedback': f"Good analysis with confidence {confidence}",
            'strengths': ["Clear reasoning"],
            'weaknesses': ["Could improve risk assessment"],
            'suggestions': ["Add more technical indicators"]
        }
    
    judge.evaluate.side_effect = mock_evaluate
    return judge


@pytest.fixture
def synthesis_config(temp_dir):
    """Create synthesis config"""
    return AutoSynthesisConfig(
        min_reward=0.4,
        min_confidence=0.6,
        quality_threshold=0.5,
        enable_judge_filtering=True,
        judge_min_score=0.6,
        max_examples_per_agent=20,
        output_dir=str(Path(temp_dir) / "datasets"),
        versioning=True
    )


@pytest.fixture
def registry(temp_dir):
    """Create dataset registry"""
    return DatasetRegistry(registry_dir=str(Path(temp_dir) / "registry"))


class TestAutoSynthesisPipeline:
    """Test Auto Data-Synthesis Pipeline"""
    
    def test_pipeline_initialization(self, mock_experience_library, synthesis_config):
        """Test pipeline initialization"""
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            config=synthesis_config
        )
        
        assert pipeline.library == mock_experience_library
        assert pipeline.config == synthesis_config
        assert pipeline.synthesis_module is not None
        assert pipeline.output_dir.exists()
    
    def test_post_backtest_synthesis_without_judge(
        self,
        mock_experience_library,
        synthesis_config,
        temp_dir
    ):
        """Test post-backtest synthesis without judge filtering"""
        # Disable judge filtering
        synthesis_config.enable_judge_filtering = False
        
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            judge=None,
            config=synthesis_config
        )
        
        # Run synthesis
        results = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_001",
            agent_types=['technical']
        )
        
        # Assertions
        assert len(results) == 1
        result = results[0]
        
        assert result.agent_type == 'technical'
        assert result.num_examples_generated > 0
        assert result.avg_quality_score > 0.0
        assert Path(result.output_path).exists()
        
        # Check dataset file
        with open(result.output_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Validate ChatML format
            example = json.loads(lines[0])
            assert 'messages' in example
            assert len(example['messages']) >= 2  # At least user + assistant
    
    def test_post_backtest_synthesis_with_judge(
        self,
        mock_experience_library,
        mock_judge,
        synthesis_config,
        temp_dir
    ):
        """Test post-backtest synthesis with judge filtering"""
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            judge=mock_judge,
            config=synthesis_config
        )
        
        # Run synthesis
        results = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_002",
            agent_types=['technical']
        )
        
        # Assertions
        assert len(results) == 1
        result = results[0]
        
        assert result.agent_type == 'technical'
        assert result.num_examples_generated > 0
        assert result.num_judge_filtered >= 0
        assert result.avg_judge_score > 0.0
        
        # Verify judge was called
        assert mock_judge.evaluate.called
    
    def test_synthesis_quality_filtering(
        self,
        mock_experience_library,
        synthesis_config,
        temp_dir
    ):
        """Test quality filtering thresholds"""
        # Set high quality threshold
        synthesis_config.min_confidence = 0.8
        synthesis_config.quality_threshold = 0.75
        
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            config=synthesis_config
        )
        
        results = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_003",
            agent_types=['technical']
        )
        
        result = results[0]
        
        # Should have fewer examples due to stricter filtering
        assert result.num_examples_generated < 10
        assert result.avg_quality_score >= synthesis_config.quality_threshold
    
    def test_dataset_versioning(
        self,
        mock_experience_library,
        synthesis_config,
        temp_dir
    ):
        """Test dataset versioning"""
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            config=synthesis_config
        )
        
        # Run synthesis twice
        results1 = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_004",
            agent_types=['technical']
        )
        
        results2 = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_005",
            agent_types=['technical']
        )
        
        # Check versions are different
        version1 = results1[0].dataset_version
        version2 = results2[0].dataset_version
        
        assert version1 != version2
        assert version1.startswith('v1.0.')
        assert version2.startswith('v1.0.')
        
        # Version 2 should have higher patch number
        patch1 = int(version1.split('.')[2].split('_')[0])
        patch2 = int(version2.split('.')[2].split('_')[0])
        assert patch2 > patch1
    
    def test_multiple_agent_types(
        self,
        mock_experience_library,
        synthesis_config,
        temp_dir
    ):
        """Test synthesis for multiple agent types"""
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            config=synthesis_config
        )
        
        # Run synthesis for all agent types
        results = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_006",
            agent_types=['news', 'technical', 'fundamental', 'strategist']
        )
        
        # Should have results for all agent types
        assert len(results) == 4
        
        agent_types = {r.agent_type for r in results}
        assert agent_types == {'news', 'technical', 'fundamental', 'strategist'}
        
        # Each should have generated examples
        for result in results:
            assert result.num_examples_generated > 0
            assert Path(result.output_path).exists()


class TestDatasetRegistry:
    """Test Dataset Registry"""
    
    def test_registry_initialization(self, registry):
        """Test registry initialization"""
        assert registry.registry_dir.exists()
        assert registry.db_file.exists()
        assert 'datasets' in registry.db
        assert 'metadata' in registry.db
    
    def test_register_dataset(self, registry, temp_dir):
        """Test dataset registration"""
        # Create dummy dataset file
        dataset_file = Path(temp_dir) / "test_dataset.jsonl"
        with open(dataset_file, 'w') as f:
            f.write('{"messages": []}\n')
        
        # Register dataset
        version = registry.register_dataset(
            agent_type='technical',
            file_path=str(dataset_file),
            num_examples=100,
            avg_quality_score=0.75,
            avg_judge_score=0.8,
            source_backtest_id='test_backtest_001',
            tags=['high-quality', 'bull-market']
        )
        
        # Assertions
        assert version.version == 'v1.0.0'
        assert version.agent_type == 'technical'
        assert version.num_examples == 100
        assert version.avg_quality_score == 0.75
        assert version.avg_judge_score == 0.8
        assert 'high-quality' in version.tags
    
    def test_get_latest_version(self, registry, temp_dir):
        """Test getting latest version"""
        # Register multiple versions
        for i in range(3):
            dataset_file = Path(temp_dir) / f"dataset_{i}.jsonl"
            with open(dataset_file, 'w') as f:
                f.write('{"messages": []}\n')
            
            registry.register_dataset(
                agent_type='technical',
                file_path=str(dataset_file),
                num_examples=100 + i,
                avg_quality_score=0.7 + (i * 0.05),
                avg_judge_score=0.75
            )
        
        # Get latest
        latest = registry.get_latest_version('technical')
        
        assert latest is not None
        assert latest.version == 'v1.0.2'  # Third version (patch incremented)
        assert latest.num_examples == 102
    
    def test_list_versions_with_filters(self, registry, temp_dir):
        """Test listing versions with filters"""
        # Register versions with different tags and quality
        for i in range(5):
            dataset_file = Path(temp_dir) / f"dataset_{i}.jsonl"
            with open(dataset_file, 'w') as f:
                f.write('{"messages": []}\n')
            
            tags = ['high-quality'] if i % 2 == 0 else ['low-quality']
            
            registry.register_dataset(
                agent_type='technical',
                file_path=str(dataset_file),
                num_examples=100,
                avg_quality_score=0.6 + (i * 0.05),
                avg_judge_score=0.7,
                tags=tags
            )
        
        # Filter by tags
        high_quality = registry.list_versions('technical', tags=['high-quality'])
        assert len(high_quality) == 3
        
        # Filter by quality
        high_score = registry.list_versions('technical', min_quality=0.7)
        assert len(high_score) == 2
    
    def test_compare_versions(self, registry, temp_dir):
        """Test version comparison"""
        # Register two versions
        for i in range(2):
            dataset_file = Path(temp_dir) / f"dataset_{i}.jsonl"
            with open(dataset_file, 'w') as f:
                f.write('{"messages": []}\n' * (100 + i * 50))
            
            registry.register_dataset(
                agent_type='technical',
                file_path=str(dataset_file),
                num_examples=100 + i * 50,
                avg_quality_score=0.7 + (i * 0.1),
                avg_judge_score=0.75 + (i * 0.05)
            )
        
        # Compare
        comparison = registry.compare_versions('technical', 'v1.0.0', 'v1.0.1')
        
        assert comparison['num_examples_diff'] == 50
        assert comparison['quality_score_diff'] == pytest.approx(0.1, abs=0.01)
        assert comparison['judge_score_diff'] == pytest.approx(0.05, abs=0.01)
    
    def test_get_stats(self, registry, temp_dir):
        """Test aggregated statistics"""
        # Register multiple versions
        for i in range(3):
            dataset_file = Path(temp_dir) / f"dataset_{i}.jsonl"
            with open(dataset_file, 'w') as f:
                f.write('{"messages": []}\n')
            
            registry.register_dataset(
                agent_type='technical',
                file_path=str(dataset_file),
                num_examples=100 + i * 10,
                avg_quality_score=0.7 + (i * 0.05),
                avg_judge_score=0.75
            )
        
        # Get stats
        stats = registry.get_stats('technical')
        
        assert stats.total_versions == 3
        assert stats.total_examples == 330  # 100 + 110 + 120
        assert stats.latest_version == 'v1.0.2'
        assert stats.avg_quality_score == pytest.approx(0.75, abs=0.01)


class TestJudgeFiltering:
    """Test Judge-Approved Filtering"""
    
    def test_filter_initialization(self, mock_judge):
        """Test filter initialization"""
        config = JudgeFilterConfig.from_strategy('balanced')
        filter_system = JudgeApprovedFilter(judge=mock_judge, config=config)
        
        assert filter_system.judge == mock_judge
        assert filter_system.config.strategy == 'balanced'
        assert filter_system.config.min_overall_score == 0.7
    
    def test_filter_trajectories_strict(self, mock_judge, mock_experience_library):
        """Test strict filtering strategy"""
        config = JudgeFilterConfig.from_strategy('strict')
        filter_system = JudgeApprovedFilter(judge=mock_judge, config=config)
        
        # Get sample trajectories
        trajectories = mock_experience_library.get_successful_trajectories()
        
        # Filter
        passed, stats = filter_system.filter_trajectories(trajectories, 'technical')
        
        # Assertions
        assert stats.total_evaluated == len(trajectories)
        assert stats.total_passed < stats.total_evaluated  # Some should fail strict filtering
        assert stats.pass_rate < 1.0
        assert len(passed) == stats.total_passed
    
    def test_filter_trajectories_lenient(self, mock_judge, mock_experience_library):
        """Test lenient filtering strategy"""
        config = JudgeFilterConfig.from_strategy('lenient')
        filter_system = JudgeApprovedFilter(judge=mock_judge, config=config)
        
        trajectories = mock_experience_library.get_successful_trajectories()
        passed, stats = filter_system.filter_trajectories(trajectories, 'technical')
        
        # Lenient should pass more
        assert stats.pass_rate > 0.5
    
    def test_filtering_stats(self, mock_judge, mock_experience_library):
        """Test filtering statistics"""
        config = JudgeFilterConfig.from_strategy('balanced')
        filter_system = JudgeApprovedFilter(judge=mock_judge, config=config)
        
        trajectories = mock_experience_library.get_successful_trajectories()
        passed, stats = filter_system.filter_trajectories(trajectories, 'technical')
        
        # Check stats structure
        assert stats.total_evaluated > 0
        assert stats.total_passed + stats.total_failed == stats.total_evaluated
        assert 0.0 <= stats.pass_rate <= 1.0
        assert 0.0 <= stats.avg_overall_score <= 1.0
        assert len(stats.avg_dimension_scores) > 0
        assert isinstance(stats.failure_reasons, dict)


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    def test_complete_workflow(
        self,
        mock_experience_library,
        mock_judge,
        synthesis_config,
        registry,
        temp_dir
    ):
        """Test complete workflow: synthesis → filtering → registry"""
        # Step 1: Run synthesis with judge filtering
        pipeline = AutoDataSynthesisPipeline(
            experience_library=mock_experience_library,
            judge=mock_judge,
            config=synthesis_config
        )
        
        results = pipeline.run_post_backtest_synthesis(
            backtest_id="test_backtest_e2e",
            agent_types=['technical']
        )
        
        result = results[0]
        
        # Step 2: Register dataset
        version = registry.register_dataset(
            agent_type=result.agent_type,
            file_path=result.output_path,
            num_examples=result.num_examples_generated,
            avg_quality_score=result.avg_quality_score,
            avg_judge_score=result.avg_judge_score,
            source_backtest_id="test_backtest_e2e",
            tags=['auto-synthesis', 'judge-filtered']
        )
        
        # Step 3: Verify registration
        latest = registry.get_latest_version('technical')
        
        assert latest is not None
        assert latest.agent_type == 'technical'
        assert latest.num_examples == result.num_examples_generated
        assert 'auto-synthesis' in latest.tags
        assert 'judge-filtered' in latest.tags
        
        # Step 4: Verify dataset file exists and is valid
        assert Path(latest.file_path).exists()
        
        with open(latest.file_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == latest.num_examples
            
            # Validate format
            for line in lines:
                example = json.loads(line)
                assert 'messages' in example


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
