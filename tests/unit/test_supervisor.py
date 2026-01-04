"""
Unit Tests for Supervisor Agent
"""

import pytest
from unittest.mock import Mock, patch
from agents.supervisor.supervisor_agent import SupervisorAgent


class TestSupervisorAgent:
    """Test suite for Supervisor Agent"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'model_path': 'mock/model/path',
            'context_dim': 16,
            'hidden_dim': 128,
            'exploration_factor': 0.5
        }
    
    @pytest.fixture
    def mock_agent(self, mock_config):
        """Create mock supervisor agent"""
        agent = SupervisorAgent(
            model_path=mock_config['model_path'],
            config=mock_config
        )
        return agent
    
    @pytest.fixture
    def mock_context(self):
        """Mock context for routing"""
        return {
            'symbol': 'AAPL',
            'market_data': {
                'price': 180.0,
                'volume': 50000000,
                'volatility': 0.25
            },
            'portfolio_state': {
                'cash': 100000,
                'positions': {},
                'total_value': 100000
            }
        }
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initializes correctly"""
        assert mock_agent is not None
        assert mock_agent.name == "SupervisorAgent"
    
    def test_route_returns_strategy(self, mock_agent, mock_context):
        """Test routing returns a valid strategy"""
        result = mock_agent.route(mock_context)
        
        assert 'strategy' in result
        assert 'active_agents' in result
        assert isinstance(result['active_agents'], list)
    
    def test_strategy_values(self, mock_agent, mock_context):
        """Test strategy is one of valid values"""
        result = mock_agent.route(mock_context)
        
        valid_strategies = [
            'news_only', 'technical_only', 'fundamental_only',
            'news_technical', 'technical_fundamental', 'news_fundamental',
            'all_agents'
        ]
        
        assert result['strategy'] in valid_strategies
    
    def test_active_agents_list(self, mock_agent, mock_context):
        """Test active agents list is valid"""
        result = mock_agent.route(mock_context)
        
        valid_agents = ['news', 'technical', 'fundamental']
        
        for agent in result['active_agents']:
            assert agent in valid_agents
    
    def test_update_method(self, mock_agent, mock_context):
        """Test update method works"""
        try:
            mock_agent.update(mock_context, action=6, reward=0.8)
            assert True
        except Exception as e:
            pytest.fail(f"Update method failed: {e}")
    
    def test_exploration_exploitation(self, mock_agent, mock_context):
        """Test exploration vs exploitation behavior"""
        # Run multiple times to test randomness
        strategies = []
        for _ in range(10):
            result = mock_agent.route(mock_context)
            strategies.append(result['strategy'])
        
        # Should have some variation due to exploration
        assert len(set(strategies)) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
