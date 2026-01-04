"""
Unit Tests for LLM Judge System
"""

import pytest
from unittest.mock import Mock, patch
from judge.llm_judge import LLMJudge


class TestLLMJudge:
    """Test suite for LLM Judge"""
    
    @pytest.fixture
    def mock_judge(self):
        """Create mock judge"""
        with patch('judge.llm_judge.Anthropic'):
            judge = LLMJudge()
            return judge
    
    @pytest.fixture
    def mock_agent_output(self):
        """Mock agent output"""
        return {
            'sentiment_score': 1.5,
            'confidence': 0.85,
            'key_events': ['Product launch'],
            'reasoning': 'Strong positive sentiment from news',
            'recommendation': 'bullish'
        }
    
    def test_judge_initialization(self, mock_judge):
        """Test judge initializes correctly"""
        assert mock_judge is not None
    
    @patch('judge.llm_judge.Anthropic')
    def test_evaluate_returns_dict(self, mock_anthropic, mock_judge, mock_agent_output):
        """Test evaluate returns a dictionary"""
        # Mock Anthropic response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"overall_score": 0.85, "dimensions": {}}')]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        result = mock_judge.evaluate(mock_agent_output, 'news')
        
        assert isinstance(result, dict)
    
    def test_calculate_reward_range(self, mock_judge):
        """Test reward is within valid range"""
        evaluation = {
            'overall_score': 0.85,
            'dimensions': {}
        }
        
        reward = mock_judge.calculate_reward(evaluation)
        
        assert 0 <= reward <= 1
    
    def test_calculate_reward_from_score(self, mock_judge):
        """Test reward calculation from score"""
        evaluation = {
            'overall_score': 0.75
        }
        
        reward = mock_judge.calculate_reward(evaluation)
        
        assert reward == pytest.approx(0.75, rel=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
