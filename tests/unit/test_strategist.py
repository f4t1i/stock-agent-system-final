"""
Unit Tests for Senior Strategist Agent
"""

import pytest
from unittest.mock import Mock, patch
from agents.senior.strategist_agent import StrategistAgent


class TestStrategistAgent:
    """Test suite for Senior Strategist Agent"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'model_path': 'mock/model/path',
            'fp16': False,
            'max_new_tokens': 768,
            'temperature': 0.7,
            'max_position_size': 0.10,
            'max_drawdown': 0.15,
            'stop_loss': 0.05
        }
    
    @pytest.fixture
    def mock_agent(self, mock_config):
        """Create mock strategist agent"""
        with patch('agents.senior.strategist_agent.AutoModelForCausalLM'):
            with patch('agents.senior.strategist_agent.AutoTokenizer'):
                agent = StrategistAgent(
                    model_path=mock_config['model_path'],
                    config=mock_config
                )
                return agent
    
    @pytest.fixture
    def mock_agent_outputs(self):
        """Mock outputs from junior agents"""
        return {
            'news': {
                'sentiment_score': 1.2,
                'confidence': 0.85,
                'recommendation': 'bullish'
            },
            'technical': {
                'signal': 'bullish',
                'signal_strength': 0.75,
                'recommendation': 'buy'
            },
            'fundamental': {
                'valuation': 'fairly_valued',
                'financial_health_score': 0.80,
                'recommendation': 'hold'
            }
        }
    
    @pytest.fixture
    def mock_portfolio_state(self):
        """Mock portfolio state"""
        return {
            'cash': 100000,
            'positions': {},
            'total_value': 100000,
            'current_drawdown': 0.0
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data"""
        return {
            'regularMarketPrice': 180.0,
            'regularMarketVolume': 50000000,
            'regularMarketChangePercent': 1.5
        }
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initializes correctly"""
        assert mock_agent is not None
        assert mock_agent.name == "StrategistAgent"
    
    def test_analyze_with_all_inputs(
        self,
        mock_agent,
        mock_agent_outputs,
        mock_portfolio_state,
        mock_market_data
    ):
        """Test analysis with all inputs"""
        # Mock LLM response
        mock_agent._generate_response = Mock(return_value="""
        {
            "decision": "buy",
            "confidence": 0.80,
            "position_size": 0.08,
            "entry_target": 180.0,
            "stop_loss": 171.0,
            "take_profit": 198.0,
            "reasoning": "Strong bullish signals from all agents",
            "risk_assessment": "Moderate risk with favorable risk/reward"
        }
        """)
        
        result = mock_agent.analyze(
            'AAPL',
            mock_agent_outputs,
            mock_portfolio_state,
            mock_market_data
        )
        
        assert 'decision' in result
        assert 'confidence' in result
        assert 'position_size' in result
    
    def test_decision_values(
        self,
        mock_agent,
        mock_agent_outputs,
        mock_portfolio_state,
        mock_market_data
    ):
        """Test decision is one of valid values"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "decision": "buy",
            "confidence": 0.75,
            "position_size": 0.05,
            "reasoning": "Test",
            "risk_assessment": "Test"
        }
        """)
        
        result = mock_agent.analyze(
            'AAPL',
            mock_agent_outputs,
            mock_portfolio_state,
            mock_market_data
        )
        
        assert result['decision'] in ['buy', 'sell', 'hold']
    
    def test_confidence_range(
        self,
        mock_agent,
        mock_agent_outputs,
        mock_portfolio_state,
        mock_market_data
    ):
        """Test confidence is within valid range"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "decision": "buy",
            "confidence": 0.85,
            "position_size": 0.07,
            "reasoning": "Test",
            "risk_assessment": "Test"
        }
        """)
        
        result = mock_agent.analyze(
            'AAPL',
            mock_agent_outputs,
            mock_portfolio_state,
            mock_market_data
        )
        
        assert 0 <= result['confidence'] <= 1
    
    def test_position_size_range(
        self,
        mock_agent,
        mock_agent_outputs,
        mock_portfolio_state,
        mock_market_data
    ):
        """Test position size is within valid range"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "decision": "buy",
            "confidence": 0.80,
            "position_size": 0.08,
            "reasoning": "Test",
            "risk_assessment": "Test"
        }
        """)
        
        result = mock_agent.analyze(
            'AAPL',
            mock_agent_outputs,
            mock_portfolio_state,
            mock_market_data
        )
        
        assert 0 <= result['position_size'] <= 1
    
    def test_risk_management_fields(
        self,
        mock_agent,
        mock_agent_outputs,
        mock_portfolio_state,
        mock_market_data
    ):
        """Test risk management fields are present"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "decision": "buy",
            "confidence": 0.80,
            "position_size": 0.08,
            "entry_target": 180.0,
            "stop_loss": 171.0,
            "take_profit": 198.0,
            "reasoning": "Test",
            "risk_assessment": "Test"
        }
        """)
        
        result = mock_agent.analyze(
            'AAPL',
            mock_agent_outputs,
            mock_portfolio_state,
            mock_market_data
        )
        
        assert 'stop_loss' in result or 'entry_target' in result
        assert 'risk_assessment' in result
    
    def test_error_handling(
        self,
        mock_agent,
        mock_agent_outputs,
        mock_portfolio_state,
        mock_market_data
    ):
        """Test error handling when analysis fails"""
        mock_agent._generate_response = Mock(side_effect=Exception("Test error"))
        
        result = mock_agent.analyze(
            'AAPL',
            mock_agent_outputs,
            mock_portfolio_state,
            mock_market_data
        )
        
        assert 'error' in result or result['decision'] == 'hold'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
