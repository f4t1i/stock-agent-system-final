"""
Unit Tests for Pairwise Comparison Framework

Tests:
1. Pairwise Judge comparison logic
2. Data generation (strategy pairs)
3. Reward model predictions
4. Dataset management
5. Comparison consistency
"""

import pytest
import time
from unittest.mock import Mock, patch
from training.pairwise.pairwise_comparison import (
    PairwiseJudge,
    PairwiseDataGenerator,
    PairwiseRewardModel,
    PairwiseTrainingDataset,
    ComparisonResult,
    StrategyOutput,
    PairwiseComparison
)


class TestStrategyOutput:
    """Test StrategyOutput dataclass"""
    
    def test_strategy_output_creation(self):
        """Test: Create strategy output"""
        strategy = StrategyOutput(
            recommendation='buy',
            confidence=0.8,
            reasoning='Strong bullish signals',
            position_size=0.1,
            entry_target=180.0,
            stop_loss=175.0,
            take_profit=195.0
        )
        
        assert strategy.recommendation == 'buy'
        assert strategy.confidence == 0.8
        assert strategy.entry_target == 180.0
    
    def test_strategy_to_dict(self):
        """Test: Convert strategy to dict"""
        strategy = StrategyOutput(
            recommendation='sell',
            confidence=0.7,
            reasoning='Bearish pattern',
            position_size=0.05
        )
        
        data = strategy.to_dict()
        
        assert data['recommendation'] == 'sell'
        assert data['confidence'] == 0.7
        assert 'reasoning' in data


class TestPairwiseJudge:
    """Test PairwiseJudge"""
    
    @pytest.fixture
    def judge(self):
        return PairwiseJudge(config={'model': 'gpt-4.1-mini'})
    
    @pytest.fixture
    def market_state(self):
        return {
            'symbol': 'AAPL',
            'current_price': 180.0,
            'volatility': 0.25,
            'trend': 'bullish',
            'news_sentiment': 0.7
        }
    
    @pytest.fixture
    def good_strategy(self):
        """Strategy with good risk management"""
        return StrategyOutput(
            recommendation='buy',
            confidence=0.8,
            reasoning='Strong bullish trend with positive news sentiment. Technical indicators show momentum. Risk-reward ratio of 1:2.5 is favorable.',
            position_size=0.08,
            entry_target=178.0,
            stop_loss=172.0,
            take_profit=195.0,
            risk_assessment='Moderate risk. Stop loss at 3.3% below entry provides downside protection.'
        )
    
    @pytest.fixture
    def poor_strategy(self):
        """Strategy with poor risk management"""
        return StrategyOutput(
            recommendation='buy',
            confidence=0.9,
            reasoning='Buy signal.',
            position_size=0.15,
            entry_target=180.0,
            stop_loss=None,  # No stop loss!
            take_profit=None,  # No take profit!
            risk_assessment=''
        )
    
    @patch('training.pairwise.pairwise_comparison.OpenAI')
    def test_compare_strategies(self, mock_openai, judge, market_state, good_strategy, poor_strategy):
        """Test: Compare two strategies"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "winner": "A",
            "reasoning": "Strategy A has better risk management with defined stop loss and take profit",
            "confidence": 0.85,
            "criteria_scores": {
                "reasoning_quality": "A",
                "risk_management": "A",
                "confidence_calibration": "A",
                "actionability": "A",
                "market_context_awareness": "A"
            }
        }
        '''
        
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Compare
        comparison = judge.compare(market_state, good_strategy, poor_strategy)
        
        # Assertions
        assert comparison.winner == ComparisonResult.A_BETTER
        assert comparison.confidence > 0.5
        assert 'risk management' in comparison.reasoning.lower()
    
    def test_build_comparison_prompt(self, judge, market_state, good_strategy, poor_strategy):
        """Test: Build comparison prompt"""
        prompt = judge._build_comparison_prompt(market_state, good_strategy, poor_strategy)
        
        # Check prompt contains key elements
        assert 'AAPL' in prompt
        assert 'Strategy A' in prompt
        assert 'Strategy B' in prompt
        assert 'reasoning_quality' in prompt
        assert 'risk_management' in prompt


class TestPairwiseDataGenerator:
    """Test PairwiseDataGenerator"""
    
    @pytest.fixture
    def mock_strategist(self):
        """Mock strategist agent"""
        strategist = Mock()
        
        def mock_synthesize(symbol, agent_outputs, portfolio_state, temperature, seed):
            # Return different strategies based on temperature
            if temperature < 0.8:
                return {
                    'recommendation': 'buy',
                    'confidence': 0.8,
                    'reasoning': 'Conservative strategy with good risk management.',
                    'position_size': 0.08,
                    'entry_target': 178.0,
                    'stop_loss': 172.0,
                    'take_profit': 195.0,
                    'risk_assessment': 'Moderate risk.'
                }
            else:
                return {
                    'recommendation': 'strong_buy',
                    'confidence': 0.9,
                    'reasoning': 'Aggressive strategy with higher upside.',
                    'position_size': 0.12,
                    'entry_target': 180.0,
                    'stop_loss': 170.0,
                    'take_profit': 200.0,
                    'risk_assessment': 'Higher risk, higher reward.'
                }
        
        strategist.synthesize = mock_synthesize
        return strategist
    
    def test_generate_pair(self, mock_strategist):
        """Test: Generate pair of strategies"""
        generator = PairwiseDataGenerator(mock_strategist)
        
        market_state = {'symbol': 'AAPL', 'volatility': 0.3}
        agent_outputs = {'news': {}, 'technical': {}}
        
        strategy_a, strategy_b = generator.generate_pair(market_state, agent_outputs)
        
        # Should generate two different strategies
        assert strategy_a.recommendation != strategy_b.recommendation or \
               strategy_a.confidence != strategy_b.confidence
        
        # Both should be valid
        assert strategy_a.recommendation in ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
        assert 0 <= strategy_a.confidence <= 1
        assert 0 <= strategy_b.confidence <= 1


class TestPairwiseRewardModel:
    """Test PairwiseRewardModel"""
    
    @pytest.fixture
    def reward_model(self):
        return PairwiseRewardModel()
    
    @pytest.fixture
    def market_state(self):
        return {
            'symbol': 'AAPL',
            'volatility': 0.25
        }
    
    def test_predict_preference(self, reward_model, market_state):
        """Test: Predict preference between strategies"""
        strategy_a = StrategyOutput(
            recommendation='buy',
            confidence=0.8,
            reasoning='Detailed analysis with multiple factors considered. Volatility is moderate, trend is bullish, risk is managed.',
            position_size=0.08,
            stop_loss=175.0,
            take_profit=195.0
        )
        
        strategy_b = StrategyOutput(
            recommendation='buy',
            confidence=0.9,
            reasoning='Buy.',
            position_size=0.1,
            stop_loss=None,
            take_profit=None
        )
        
        score_a, score_b = reward_model.predict_preference(strategy_a, strategy_b, market_state)
        
        # Strategy A should score higher (better reasoning, risk management)
        assert score_a > score_b
    
    def test_add_comparison(self, reward_model):
        """Test: Add comparison to model"""
        comparison = PairwiseComparison(
            market_state={'symbol': 'AAPL'},
            strategy_a=StrategyOutput('buy', 0.8, 'Good', 0.1),
            strategy_b=StrategyOutput('sell', 0.7, 'Bad', 0.05),
            winner=ComparisonResult.A_BETTER,
            reasoning='A is better',
            confidence=0.9
        )
        
        reward_model.add_comparison(comparison)
        
        assert len(reward_model.comparisons) == 1
    
    def test_get_statistics(self, reward_model):
        """Test: Get model statistics"""
        # Add some comparisons
        for i in range(10):
            comparison = PairwiseComparison(
                market_state={'symbol': 'AAPL'},
                strategy_a=StrategyOutput('buy', 0.8, 'A', 0.1),
                strategy_b=StrategyOutput('sell', 0.7, 'B', 0.05),
                winner=ComparisonResult.A_BETTER if i % 2 == 0 else ComparisonResult.B_BETTER,
                reasoning='Test',
                confidence=0.8
            )
            reward_model.add_comparison(comparison)
        
        stats = reward_model.get_statistics()
        
        assert stats['total_comparisons'] == 10
        assert 'winner_distribution' in stats
        assert 'avg_judge_confidence' in stats


class TestPairwiseTrainingDataset:
    """Test PairwiseTrainingDataset"""
    
    @pytest.fixture
    def dataset(self, tmp_path):
        """Create dataset with temp path"""
        save_path = tmp_path / "test_comparisons.jsonl"
        return PairwiseTrainingDataset(save_path=str(save_path))
    
    def test_add_comparison(self, dataset):
        """Test: Add comparison to dataset"""
        comparison = PairwiseComparison(
            market_state={'symbol': 'AAPL'},
            strategy_a=StrategyOutput('buy', 0.8, 'Good reasoning', 0.1),
            strategy_b=StrategyOutput('sell', 0.7, 'Poor reasoning', 0.05),
            winner=ComparisonResult.A_BETTER,
            reasoning='A has better reasoning',
            confidence=0.85
        )
        
        dataset.add_comparison(comparison)
        
        assert len(dataset.comparisons) == 1
        assert dataset.comparisons[0].winner == ComparisonResult.A_BETTER
    
    def test_save_and_load(self, dataset):
        """Test: Save and load dataset"""
        # Add comparisons
        for i in range(5):
            comparison = PairwiseComparison(
                market_state={'symbol': f'STOCK{i}'},
                strategy_a=StrategyOutput('buy', 0.8, f'Reasoning A{i}', 0.1),
                strategy_b=StrategyOutput('sell', 0.7, f'Reasoning B{i}', 0.05),
                winner=ComparisonResult.A_BETTER,
                reasoning=f'Comparison {i}',
                confidence=0.8
            )
            dataset.add_comparison(comparison)
        
        # Save
        dataset.save()
        
        # Load into new dataset
        new_dataset = PairwiseTrainingDataset(save_path=dataset.save_path)
        new_dataset.load()
        
        # Check
        assert len(new_dataset.comparisons) == 5
        assert new_dataset.comparisons[0].market_state['symbol'] == 'STOCK0'
    
    def test_get_statistics(self, dataset):
        """Test: Get dataset statistics"""
        # Add comparisons with different winners
        for i in range(10):
            if i < 5:
                winner = ComparisonResult.A_BETTER
            elif i < 8:
                winner = ComparisonResult.B_BETTER
            else:
                winner = ComparisonResult.TIE
            
            comparison = PairwiseComparison(
                market_state={'symbol': 'AAPL'},
                strategy_a=StrategyOutput('buy', 0.8, 'A', 0.1),
                strategy_b=StrategyOutput('sell', 0.7, 'B', 0.05),
                winner=winner,
                reasoning='Test',
                confidence=0.8
            )
            dataset.add_comparison(comparison)
        
        stats = dataset.get_statistics()
        
        assert stats['total_comparisons'] == 10
        assert stats['winner_distribution']['A_better'] == 5
        assert stats['winner_distribution']['B_better'] == 3
        assert stats['winner_distribution']['tie'] == 2


class TestComparisonConsistency:
    """Test consistency of comparisons"""
    
    def test_same_strategies_should_tie(self):
        """Test: Comparing identical strategies should result in tie"""
        reward_model = PairwiseRewardModel()
        
        strategy = StrategyOutput(
            recommendation='buy',
            confidence=0.8,
            reasoning='Good analysis',
            position_size=0.1,
            stop_loss=175.0,
            take_profit=195.0
        )
        
        market_state = {'symbol': 'AAPL'}
        
        score_a, score_b = reward_model.predict_preference(strategy, strategy, market_state)
        
        # Scores should be identical
        assert abs(score_a - score_b) < 0.01
    
    def test_better_risk_management_scores_higher(self):
        """Test: Strategy with better risk management scores higher"""
        reward_model = PairwiseRewardModel()
        
        with_risk_mgmt = StrategyOutput(
            recommendation='buy',
            confidence=0.8,
            reasoning='Analysis',
            position_size=0.1,
            stop_loss=175.0,
            take_profit=195.0
        )
        
        without_risk_mgmt = StrategyOutput(
            recommendation='buy',
            confidence=0.8,
            reasoning='Analysis',
            position_size=0.1,
            stop_loss=None,
            take_profit=None
        )
        
        market_state = {'symbol': 'AAPL'}
        
        score_with, score_without = reward_model.predict_preference(
            with_risk_mgmt,
            without_risk_mgmt,
            market_state
        )
        
        # With risk management should score higher
        assert score_with > score_without


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
