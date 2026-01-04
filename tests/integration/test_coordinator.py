"""
Integration Tests for System Coordinator

Tests the orchestration logic and agent coordination.
"""

import pytest
from unittest.mock import Mock, patch
from orchestration.coordinator import SystemCoordinator


class TestCoordinator:
    """Integration tests for SystemCoordinator"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'agents': {
                'news': {
                    'enabled': True,
                    'model_path': 'mock/news',
                    'weight': 0.4,
                    'fp16': False,
                    'max_new_tokens': 512,
                    'temperature': 0.7
                },
                'technical': {
                    'enabled': True,
                    'model_path': 'mock/technical',
                    'weight': 0.35,
                    'fp16': False,
                    'max_new_tokens': 512,
                    'temperature': 0.5
                },
                'fundamental': {
                    'enabled': True,
                    'model_path': 'mock/fundamental',
                    'weight': 0.25,
                    'fp16': False,
                    'max_new_tokens': 512,
                    'temperature': 0.6
                }
            },
            'supervisor': {
                'enabled': False
            },
            'strategist': {
                'model_path': 'mock/strategist',
                'fp16': False,
                'max_new_tokens': 768,
                'temperature': 0.7,
                'max_position_size': 0.10,
                'max_drawdown': 0.15,
                'stop_loss': 0.05
            }
        }
    
    def test_coordinator_initialization(self, mock_config):
        """Test coordinator initializes correctly"""
        with patch('orchestration.coordinator.NewsAgent'):
            with patch('orchestration.coordinator.TechnicalAgent'):
                with patch('orchestration.coordinator.FundamentalAgent'):
                    with patch('orchestration.coordinator.StrategistAgent'):
                        with patch('orchestration.coordinator.NewsFetcher'):
                            with patch('orchestration.coordinator.MarketDataFetcher'):
                                coord = SystemCoordinator.__new__(SystemCoordinator)
                                coord.config = mock_config
                                coord.portfolio_state = {
                                    'cash': 100000,
                                    'positions': {},
                                    'total_value': 100000,
                                    'current_drawdown': 0.0
                                }
                                
                                assert coord is not None
                                assert coord.portfolio_state['cash'] == 100000
    
    def test_portfolio_state_management(self, mock_config):
        """Test portfolio state is managed correctly"""
        with patch('orchestration.coordinator.NewsAgent'):
            with patch('orchestration.coordinator.TechnicalAgent'):
                with patch('orchestration.coordinator.FundamentalAgent'):
                    with patch('orchestration.coordinator.StrategistAgent'):
                        with patch('orchestration.coordinator.NewsFetcher'):
                            with patch('orchestration.coordinator.MarketDataFetcher'):
                                coord = SystemCoordinator.__new__(SystemCoordinator)
                                coord.config = mock_config
                                coord.portfolio_state = {
                                    'cash': 100000,
                                    'positions': {},
                                    'total_value': 100000,
                                    'current_drawdown': 0.0
                                }
                                
                                # Check initial state
                                assert coord.portfolio_state['cash'] == 100000
                                assert len(coord.portfolio_state['positions']) == 0
    
    def test_agent_coordination(self, mock_config):
        """Test agents are coordinated correctly"""
        with patch('orchestration.coordinator.NewsAgent') as mock_news:
            with patch('orchestration.coordinator.TechnicalAgent') as mock_tech:
                with patch('orchestration.coordinator.FundamentalAgent') as mock_fund:
                    with patch('orchestration.coordinator.StrategistAgent') as mock_strat:
                        with patch('orchestration.coordinator.NewsFetcher'):
                            with patch('orchestration.coordinator.MarketDataFetcher') as mock_market:
                                # Setup mocks
                                mock_news_instance = Mock()
                                mock_news_instance.analyze.return_value = {
                                    'sentiment_score': 1.0,
                                    'confidence': 0.8,
                                    'recommendation': 'bullish'
                                }
                                mock_news.return_value = mock_news_instance
                                
                                mock_tech_instance = Mock()
                                mock_tech_instance.analyze.return_value = {
                                    'signal': 'bullish',
                                    'signal_strength': 0.7,
                                    'recommendation': 'buy'
                                }
                                mock_tech.return_value = mock_tech_instance
                                
                                mock_fund_instance = Mock()
                                mock_fund_instance.analyze.return_value = {
                                    'valuation': 'fairly_valued',
                                    'financial_health_score': 0.75,
                                    'recommendation': 'hold'
                                }
                                mock_fund.return_value = mock_fund_instance
                                
                                mock_strat_instance = Mock()
                                mock_strat_instance.analyze.return_value = {
                                    'decision': 'buy',
                                    'confidence': 0.75,
                                    'position_size': 0.05,
                                    'reasoning': 'Test',
                                    'risk_assessment': 'Test'
                                }
                                mock_strat.return_value = mock_strat_instance
                                
                                mock_market_instance = Mock()
                                mock_market_instance.get_realtime.return_value = {
                                    'regularMarketPrice': 150.0
                                }
                                mock_market.return_value = mock_market_instance
                                
                                # Create coordinator
                                coord = SystemCoordinator.__new__(SystemCoordinator)
                                coord.config = mock_config
                                coord.agents = {
                                    'news': mock_news_instance,
                                    'technical': mock_tech_instance,
                                    'fundamental': mock_fund_instance
                                }
                                coord.strategist = mock_strat_instance
                                coord.supervisor = None
                                coord.market_data_fetcher = mock_market_instance
                                coord.portfolio_state = {
                                    'cash': 100000,
                                    'positions': {},
                                    'total_value': 100000,
                                    'current_drawdown': 0.0
                                }
                                
                                # Run analysis
                                result = coord.analyze_symbol('AAPL')
                                
                                # Verify coordination
                                assert mock_news_instance.analyze.called
                                assert mock_tech_instance.analyze.called
                                assert mock_fund_instance.analyze.called
                                assert mock_strat_instance.analyze.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
