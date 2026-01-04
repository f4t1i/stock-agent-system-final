"""
Integration Tests for FastAPI Server

Tests API endpoints and request/response handling.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class TestAPI:
    """Integration tests for FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('api.server.SystemCoordinator'):
            from api.server import app
            client = TestClient(app)
            return client
    
    @pytest.fixture
    def mock_coordinator(self):
        """Mock coordinator"""
        coordinator = Mock()
        coordinator.config = {
            'agents': {
                'news': {'enabled': True, 'model_path': 'mock'},
                'technical': {'enabled': True, 'model_path': 'mock'},
                'fundamental': {'enabled': True, 'model_path': 'mock'}
            },
            'supervisor': {'enabled': False}
        }
        coordinator.agents = {
            'news': Mock(),
            'technical': Mock(),
            'fundamental': Mock()
        }
        coordinator.supervisor = None
        coordinator.strategist = Mock()
        return coordinator
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
    
    def test_health_endpoint(self, client, mock_coordinator):
        """Test health check endpoint"""
        with patch('api.server.coordinator', mock_coordinator):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "agents_loaded" in data
    
    def test_models_endpoint(self, client, mock_coordinator):
        """Test models info endpoint"""
        with patch('api.server.coordinator', mock_coordinator):
            response = client.get("/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "supervisor_enabled" in data
    
    def test_analyze_endpoint(self, client, mock_coordinator):
        """Test single analysis endpoint"""
        mock_coordinator.analyze_symbol.return_value = {
            'symbol': 'AAPL',
            'recommendation': 'buy',
            'confidence': 0.80,
            'reasoning': 'Test',
            'position_size': 0.08,
            'risk_assessment': 'Moderate',
            'agent_outputs': {},
            'strategist_output': {},
            'timestamp': '2024-01-01T00:00:00',
            'errors': []
        }
        
        with patch('api.server.coordinator', mock_coordinator):
            response = client.post(
                "/analyze",
                json={"symbol": "AAPL", "use_supervisor": False, "lookback_days": 7}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['symbol'] == 'AAPL'
            assert data['recommendation'] == 'buy'
    
    def test_batch_endpoint(self, client, mock_coordinator):
        """Test batch analysis endpoint"""
        mock_coordinator.batch_analyze.return_value = [
            {
                'symbol': 'AAPL',
                'recommendation': 'buy',
                'confidence': 0.80,
                'reasoning': 'Test',
                'position_size': 0.08,
                'risk_assessment': 'Moderate',
                'agent_outputs': {},
                'strategist_output': {},
                'timestamp': '2024-01-01T00:00:00',
                'errors': []
            }
        ]
        
        with patch('api.server.coordinator', mock_coordinator):
            response = client.post(
                "/batch",
                json={
                    "symbols": ["AAPL"],
                    "use_supervisor": False,
                    "lookback_days": 7
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
            assert data['total_analyzed'] == 1
    
    def test_invalid_request(self, client):
        """Test invalid request handling"""
        response = client.post(
            "/analyze",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
