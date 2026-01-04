"""
Integration Tests for Full Workflow

Tests the complete multi-agent analysis workflow end-to-end.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from orchestration.coordinator import SystemCoordinator


class TestFullWorkflow:
    """Integration tests for complete analysis workflow"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock system configuration"""
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
                'enabled': False,
                'model_path': 'mock/supervisor',
                'exploration_factor': 0.5
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
    
    @pytest.fixture
    def coordinator(self, mock_config):
        """Create coordinator with mocked agents"""
        with patch('orchestration.coordinator.NewsAgent'):
            with patch('orchestration.coordinator.TechnicalAgent'):
                with patch('orchestration.coordinator.FundamentalAgent'):
                    with patch('orchestration.coordinator.StrategistAgent'):
                        with patch('orchestration.coordinator.NewsFetcher'):
                            with patch('orchestration.coordinator.MarketDataFetcher'):
                                coord = SystemCoordinator.__new__(SystemCoordinator)
                                coord.config = mock_config
                                coord.agents = {}
                                coord.supervisor = None
                                coord.portfolio_state = {
                                    'cash': 100000,
                                    'positions': {},
                                    'total_value': 100000,
                                    'current_drawdown': 0.0
                                }
                                return coord
    
    def test_single_symbol_analysis(self, coordinator):
        """Test analyzing a single symbol"""
        # Mock agents
        coordinator.agents['news'] = Mock()
        coordinator.agents['news'].analyze.return_value = {
            'sentiment_score': 1.2,
            'confidence': 0.85,
            'recommendation': 'bullish'
        }
        
        coordinator.agents['technical'] = Mock()
        coordinator.agents['technical'].analyze.return_value = {
            'signal': 'bullish',
            'signal_strength': 0.75,
            'recommendation': 'buy'
        }
        
        coordinator.agents['fundamental'] = Mock()
        coordinator.agents['fundamental'].analyze.return_value = {
            'valuation': 'fairly_valued',
            'financial_health_score': 0.80,
            'recommendation': 'hold'
        }
        
        coordinator.strategist = Mock()
        coordinator.strategist.analyze.return_value = {
            'decision': 'buy',
            'confidence': 0.80,
            'position_size': 0.08,
            'reasoning': 'Strong bullish signals',
            'risk_assessment': 'Moderate risk'
        }
        
        coordinator.market_data_fetcher = Mock()
        coordinator.market_data_fetcher.get_realtime.return_value = {
            'regularMarketPrice': 180.0
        }
        
        # Run analysis
        result = coordinator.analyze_symbol('AAPL')
        
        # Verify result structure
        assert 'symbol' in result
        assert 'recommendation' in result
        assert 'confidence' in result
        assert 'agent_outputs' in result
        assert 'strategist_output' in result
        
        # Verify agents were called
        coordinator.agents['news'].analyze.assert_called_once()
        coordinator.agents['technical'].analyze.assert_called_once()
        coordinator.agents['fundamental'].analyze.assert_called_once()
        coordinator.strategist.analyze.assert_called_once()
    
    def test_batch_analysis(self, coordinator):
        """Test batch analysis of multiple symbols"""
        # Mock agents
        coordinator.agents['news'] = Mock()
        coordinator.agents['news'].analyze.return_value = {
            'sentiment_score': 1.0,
            'confidence': 0.8,
            'recommendation': 'bullish'
        }
        
        coordinator.agents['technical'] = Mock()
        coordinator.agents['technical'].analyze.return_value = {
            'signal': 'bullish',
            'signal_strength': 0.7,
            'recommendation': 'buy'
        }
        
        coordinator.agents['fundamental'] = Mock()
        coordinator.agents['fundamental'].analyze.return_value = {
            'valuation': 'fairly_valued',
            'financial_health_score': 0.75,
            'recommendation': 'hold'
        }
        
        coordinator.strategist = Mock()
        coordinator.strategist.analyze.return_value = {
            'decision': 'buy',
            'confidence': 0.75,
            'position_size': 0.05,
            'reasoning': 'Test',
            'risk_assessment': 'Test'
        }
        
        coordinator.market_data_fetcher = Mock()
        coordinator.market_data_fetcher.get_realtime.return_value = {
            'regularMarketPrice': 150.0
        }
        
        # Run batch analysis
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = coordinator.batch_analyze(symbols)
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert 'symbol' in result
            assert 'recommendation' in result
    
    def test_workflow_with_supervisor(self, coordinator):
        """Test workflow with supervisor routing"""
        # Enable supervisor
        coordinator.supervisor = Mock()
        coordinator.supervisor.route.return_value = {
            'strategy': 'news_technical',
            'active_agents': ['news', 'technical'],
            'reasoning': 'Test routing'
        }
        
        # Mock agents
        coordinator.agents['news'] = Mock()
        coordinator.agents['news'].analyze.return_value = {
            'sentiment_score': 1.0,
            'confidence': 0.8,
            'recommendation': 'bullish'
        }
        
        coordinator.agents['technical'] = Mock()
        coordinator.agents['technical'].analyze.return_value = {
            'signal': 'bullish',
            'signal_strength': 0.7,
            'recommendation': 'buy'
        }
        
        coordinator.agents['fundamental'] = Mock()
        
        coordinator.strategist = Mock()
        coordinator.strategist.analyze.return_value = {
            'decision': 'buy',
            'confidence': 0.75,
            'position_size': 0.05,
            'reasoning': 'Test',
            'risk_assessment': 'Test'
        }
        
        coordinator.market_data_fetcher = Mock()
        coordinator.market_data_fetcher.get_realtime.return_value = {
            'regularMarketPrice': 150.0
        }
        
        # Run with supervisor
        result = coordinator.analyze_symbol('AAPL', use_supervisor=True)
        
        # Verify supervisor was called
        coordinator.supervisor.route.assert_called_once()
        
        # Verify only selected agents were called
        coordinator.agents['news'].analyze.assert_called_once()
        coordinator.agents['technical'].analyze.assert_called_once()
        coordinator.agents['fundamental'].analyze.assert_not_called()
    
    def test_error_handling_in_workflow(self, coordinator):
        """Test error handling when agents fail"""
        # Mock agents with errors
        coordinator.agents['news'] = Mock()
        coordinator.agents['news'].analyze.side_effect = Exception("News agent error")
        
        coordinator.agents['technical'] = Mock()
        coordinator.agents['technical'].analyze.return_value = {
            'signal': 'neutral',
            'signal_strength': 0.5,
            'recommendation': 'hold'
        }
        
        coordinator.agents['fundamental'] = Mock()
        coordinator.agents['fundamental'].analyze.return_value = {
            'valuation': 'fairly_valued',
            'financial_health_score': 0.7,
            'recommendation': 'hold'
        }
        
        coordinator.strategist = Mock()
        coordinator.strategist.analyze.return_value = {
            'decision': 'hold',
            'confidence': 0.5,
            'position_size': 0.0,
            'reasoning': 'Insufficient data',
            'risk_assessment': 'High uncertainty'
        }
        
        coordinator.market_data_fetcher = Mock()
        coordinator.market_data_fetcher.get_realtime.return_value = {
            'regularMarketPrice': 150.0
        }
        
        # Run analysis
        result = coordinator.analyze_symbol('AAPL')
        
        # Should still return a result
        assert 'symbol' in result
        assert 'recommendation' in result
        
        # Should have error in news output
        assert 'error' in result.get('agent_outputs', {}).get('news', {})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
