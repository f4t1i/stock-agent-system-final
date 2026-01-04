"""
Unit Tests for News Sentiment Agent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.junior.news_agent import NewsAgent


class TestNewsAgent:
    """Test suite for News Sentiment Agent"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'model_path': 'mock/model/path',
            'fp16': False,
            'max_new_tokens': 512,
            'temperature': 0.7
        }
    
    @pytest.fixture
    def mock_agent(self, mock_config):
        """Create mock news agent"""
        with patch('agents.junior.news_agent.AutoModelForCausalLM'):
            with patch('agents.junior.news_agent.AutoTokenizer'):
                agent = NewsAgent(
                    model_path=mock_config['model_path'],
                    config=mock_config
                )
                return agent
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initializes correctly"""
        assert mock_agent is not None
        assert mock_agent.name == "NewsAgent"
    
    @patch('agents.junior.news_agent.NewsFetcher')
    def test_analyze_with_news(self, mock_fetcher, mock_agent):
        """Test analysis with news articles"""
        # Mock news fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.get_news.return_value = [
            {
                'title': 'Apple announces new iPhone',
                'summary': 'Apple unveiled the new iPhone with advanced features',
                'sentiment': 'positive',
                'source': 'TechNews'
            }
        ]
        mock_fetcher.return_value = mock_fetcher_instance
        
        # Mock LLM response
        mock_agent._generate_response = Mock(return_value="""
        {
            "sentiment_score": 1.5,
            "confidence": 0.85,
            "key_events": ["New iPhone announcement"],
            "reasoning": "Positive product launch news",
            "recommendation": "bullish"
        }
        """)
        
        result = mock_agent.analyze('AAPL', lookback_days=7)
        
        assert 'sentiment_score' in result
        assert 'confidence' in result
        assert 'recommendation' in result
    
    def test_sentiment_score_range(self, mock_agent):
        """Test sentiment score is within valid range"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "sentiment_score": 1.5,
            "confidence": 0.85,
            "key_events": [],
            "reasoning": "Test",
            "recommendation": "bullish"
        }
        """)
        
        with patch('agents.junior.news_agent.NewsFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert -2 <= result['sentiment_score'] <= 2
    
    def test_confidence_range(self, mock_agent):
        """Test confidence is within valid range"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "sentiment_score": 1.0,
            "confidence": 0.75,
            "key_events": [],
            "reasoning": "Test",
            "recommendation": "bullish"
        }
        """)
        
        with patch('agents.junior.news_agent.NewsFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert 0 <= result['confidence'] <= 1
    
    def test_error_handling(self, mock_agent):
        """Test error handling when analysis fails"""
        mock_agent._generate_response = Mock(side_effect=Exception("Test error"))
        
        with patch('agents.junior.news_agent.NewsFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert 'error' in result or result['recommendation'] == 'neutral'
    
    def test_no_news_available(self, mock_agent):
        """Test handling when no news is available"""
        with patch('agents.junior.news_agent.NewsFetcher') as mock_fetcher:
            mock_fetcher_instance = Mock()
            mock_fetcher_instance.get_news.return_value = []
            mock_fetcher.return_value = mock_fetcher_instance
            
            result = mock_agent.analyze('AAPL')
            
            assert result is not None
            assert 'sentiment_score' in result or 'error' in result
    
    def test_recommendation_values(self, mock_agent):
        """Test recommendation is one of valid values"""
        mock_agent._generate_response = Mock(return_value="""
        {
            "sentiment_score": 1.0,
            "confidence": 0.8,
            "key_events": [],
            "reasoning": "Test",
            "recommendation": "bullish"
        }
        """)
        
        with patch('agents.junior.news_agent.NewsFetcher'):
            result = mock_agent.analyze('AAPL')
        
        assert result['recommendation'] in ['bullish', 'bearish', 'neutral']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
