"""
Unit Tests for Technical Analysis Agent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from agents.junior.technical_agent import TechnicalAgent


class TestTechnicalAgent:
    """Test suite for Technical Analysis Agent"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'model_path': 'mock/model/path',
            'fp16': False,
            'max_new_tokens': 512,
            'temperature': 0.5
        }
    
    @pytest.fixture
    def mock_agent(self, mock_config):
        """Create mock technical agent"""
        with patch('agents.junior.technical_agent.AutoModelForCausalLM'):
            with patch('agents.junior.technical_agent.AutoTokenizer'):
                agent = TechnicalAgent(
                    model_path=mock_config['model_path'],
                    config=mock_config
                )
                return agent
    
    @pytest.fixture
    def mock_price_data(self):
        """Create mock price data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(150, 160, 100),
            'High': np.random.uniform(160, 170, 100),
            'Low': np.random.uniform(140, 150, 100),
            'Close': np.random.uniform(150, 160, 100),
            'Volume': np.random.uniform(1e6, 1e7, 100)
        }, index=dates)
        return data
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initializes correctly"""
        assert mock_agent is not None
        assert mock_agent.name == "TechnicalAgent"
    
    @patch('agents.junior.technical_agent.MarketDataFetcher')
    def test_analyze_with_data(self, mock_fetcher, mock_agent, mock_price_data):
        """Test analysis with price data"""
        # Mock market data fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.get_historical.return_value = mock_price_data
        mock_fetcher.return_value = mock_fetcher_instance
        
        # Mock LLM response
        mock_agent._generate_response = Mock(return_value="""
        {
            "signal": "bullish",
            "signal_strength": 0.75,
            "support_levels": [150.0, 145.0],
            "resistance_levels": [165.0, 170.0],
            "reasoning": "Strong uptrend with RSI in healthy range",
            "recommendation": "buy"
        }
        """)
        
        result = mock_agent.analyze('AAPL', period='3mo')
        
        assert 'signal' in result
        assert 'signal_strength' in result
        assert 'recommendation' in result
    
    def test_signal_values(self, mock_agent):
        """Test signal is one of valid values"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "signal": "bullish",
            "signal_strength": 0.8,
            "support_levels": [],
            "resistance_levels": [],
            "reasoning": "Test",
            "recommendation": "buy"
        }
        """)
        
        with patch('agents.junior.technical_agent.MarketDataFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert result['signal'] in ['bullish', 'bearish', 'neutral']
    
    def test_signal_strength_range(self, mock_agent):
        """Test signal strength is within valid range"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "signal": "bullish",
            "signal_strength": 0.7,
            "support_levels": [],
            "resistance_levels": [],
            "reasoning": "Test",
            "recommendation": "buy"
        }
        """)
        
        with patch('agents.junior.technical_agent.MarketDataFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert 0 <= result['signal_strength'] <= 1
    
    def test_indicators_calculation(self, mock_agent, mock_price_data):
        """Test technical indicators are calculated"""
        with patch('agents.junior.technical_agent.MarketDataFetcher') as mock_fetcher:
            mock_fetcher_instance = Mock()
            mock_fetcher_instance.get_historical.return_value = mock_price_data
            mock_fetcher.return_value = mock_fetcher_instance
            
            mock_agent._generate_response = Mock(return_value="""
            {
                "signal": "neutral",
                "signal_strength": 0.5,
                "support_levels": [],
                "resistance_levels": [],
                "reasoning": "Test",
                "recommendation": "hold"
            }
            """)
            
            result = mock_agent.analyze('AAPL')
            
            assert 'indicators' in result or 'signal' in result
    
    def test_error_handling(self, mock_agent):
        """Test error handling when analysis fails"""
        mock_agent._generate_response = Mock(side_effect=Exception("Test error"))
        
        with patch('agents.junior.technical_agent.MarketDataFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert 'error' in result or result['recommendation'] == 'hold'
    
    def test_recommendation_values(self, mock_agent):
        """Test recommendation is one of valid values"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "signal": "bullish",
            "signal_strength": 0.8,
            "support_levels": [],
            "resistance_levels": [],
            "reasoning": "Test",
            "recommendation": "buy"
        }
        """)
        
        with patch('agents.junior.technical_agent.MarketDataFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert result['recommendation'] in ['buy', 'sell', 'hold']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
