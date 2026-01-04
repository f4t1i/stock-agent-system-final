"""
Unit Tests for Fundamental Analysis Agent
"""

import pytest
from unittest.mock import Mock, patch
from agents.junior.fundamental_agent import FundamentalAgent


class TestFundamentalAgent:
    """Test suite for Fundamental Analysis Agent"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'model_path': 'mock/model/path',
            'fp16': False,
            'max_new_tokens': 512,
            'temperature': 0.6
        }
    
    @pytest.fixture
    def mock_agent(self, mock_config):
        """Create mock fundamental agent"""
        with patch('agents.junior.fundamental_agent.AutoModelForCausalLM'):
            with patch('agents.junior.fundamental_agent.AutoTokenizer'):
                agent = FundamentalAgent(
                    model_path=mock_config['model_path'],
                    config=mock_config
                )
                return agent
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initializes correctly"""
        assert mock_agent is not None
        assert mock_agent.name == "FundamentalAgent"
    
    @patch('agents.junior.fundamental_agent.yf.Ticker')
    def test_analyze_with_data(self, mock_ticker, mock_agent):
        """Test analysis with financial data"""
        # Mock yfinance data
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'trailingPE': 25.5,
            'priceToBook': 8.2,
            'returnOnEquity': 0.45,
            'debtToEquity': 150.0,
            'currentPrice': 180.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock LLM response
        mock_agent._generate_response = Mock(return_value="""
        {
            "valuation": "fairly_valued",
            "financial_health_score": 0.75,
            "growth_score": 0.80,
            "reasoning": "Strong fundamentals with moderate valuation",
            "recommendation": "hold"
        }
        """)
        
        result = mock_agent.analyze('AAPL', period='Q')
        
        assert 'valuation' in result
        assert 'financial_health_score' in result
        assert 'recommendation' in result
    
    def test_valuation_values(self, mock_agent):
        """Test valuation is one of valid values"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "valuation": "undervalued",
            "financial_health_score": 0.8,
            "growth_score": 0.7,
            "reasoning": "Test",
            "recommendation": "buy"
        }
        """)
        
        with patch('agents.junior.fundamental_agent.yf.Ticker'):
            result = mock_agent.analyze('AAPL')
        
        assert result['valuation'] in ['undervalued', 'fairly_valued', 'overvalued']
    
    def test_score_ranges(self, mock_agent):
        """Test scores are within valid range"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "valuation": "fairly_valued",
            "financial_health_score": 0.75,
            "growth_score": 0.65,
            "reasoning": "Test",
            "recommendation": "hold"
        }
        """)
        
        with patch('agents.junior.fundamental_agent.yf.Ticker'):
            result = mock_agent.analyze('AAPL')
        
        assert 0 <= result['financial_health_score'] <= 1
        assert 0 <= result['growth_score'] <= 1
    
    def test_error_handling(self, mock_agent):
        """Test error handling when analysis fails"""
        mock_agent._generate_response = Mock(side_effect=Exception("Test error"))
        
        with patch('agents.junior.fundamental_agent.yf.Ticker'):
            result = mock_agent.analyze('AAPL')
        
        assert 'error' in result or result['recommendation'] == 'hold'
    
    def test_recommendation_values(self, mock_agent):
        """Test recommendation is one of valid values"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "valuation": "undervalued",
            "financial_health_score": 0.85,
            "growth_score": 0.75,
            "reasoning": "Test",
            "recommendation": "buy"
        }
        """)
        
        with patch('agents.junior.fundamental_agent.yf.Ticker'):
            result = mock_agent.analyze('AAPL')
        
        assert result['recommendation'] in ['buy', 'hold', 'sell']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
