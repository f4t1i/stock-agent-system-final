# Testing Guide

Comprehensive testing documentation for the Stock Analysis Multi-Agent System.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [Continuous Integration](#continuous-integration)

## Overview

The project uses **pytest** as the testing framework with comprehensive unit and integration tests covering:

- All agent implementations
- Orchestration logic
- API endpoints
- Error handling
- Data validation

## Test Structure

```
tests/
├── __init__.py
├── unit/                    # Unit tests for individual components
│   ├── __init__.py
│   ├── test_news_agent.py
│   ├── test_technical_agent.py
│   ├── test_fundamental_agent.py
│   ├── test_strategist.py
│   ├── test_supervisor.py
│   └── test_judge.py
└── integration/             # Integration tests for workflows
    ├── __init__.py
    ├── test_full_workflow.py
    ├── test_coordinator.py
    └── test_api.py
```

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Run Specific Test Suites

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_news_agent.py

# Run specific test class
pytest tests/unit/test_news_agent.py::TestNewsAgent

# Run specific test method
pytest tests/unit/test_news_agent.py::TestNewsAgent::test_agent_initialization
```

### Run Tests with Markers

```bash
# Run only fast tests
pytest -m fast

# Skip slow tests
pytest -m "not slow"
```

## Unit Tests

Unit tests verify individual components in isolation using mocks.

### News Agent Tests

**File:** `tests/unit/test_news_agent.py`

**Test Cases:**
- `test_agent_initialization`: Verify agent initializes correctly
- `test_analyze_with_news`: Test analysis with news articles
- `test_sentiment_score_range`: Validate sentiment score is within [-2, 2]
- `test_confidence_range`: Validate confidence is within [0, 1]
- `test_error_handling`: Test error handling when analysis fails
- `test_no_news_available`: Test handling when no news is available
- `test_recommendation_values`: Validate recommendation is one of valid values

**Example:**

```python
def test_sentiment_score_range(mock_agent):
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
```

### Technical Agent Tests

**File:** `tests/unit/test_technical_agent.py`

**Test Cases:**
- `test_agent_initialization`
- `test_analyze_with_data`
- `test_signal_values`
- `test_signal_strength_range`
- `test_indicators_calculation`
- `test_error_handling`
- `test_recommendation_values`

### Fundamental Agent Tests

**File:** `tests/unit/test_fundamental_agent.py`

**Test Cases:**
- `test_agent_initialization`
- `test_analyze_with_data`
- `test_valuation_values`
- `test_score_ranges`
- `test_error_handling`
- `test_recommendation_values`

### Strategist Tests

**File:** `tests/unit/test_strategist.py`

**Test Cases:**
- `test_agent_initialization`
- `test_analyze_with_all_inputs`
- `test_decision_values`
- `test_confidence_range`
- `test_position_size_range`
- `test_risk_management_fields`
- `test_error_handling`

### Supervisor Tests

**File:** `tests/unit/test_supervisor.py`

**Test Cases:**
- `test_agent_initialization`
- `test_route_returns_strategy`
- `test_strategy_values`
- `test_active_agents_list`
- `test_update_method`
- `test_exploration_exploitation`

### Judge Tests

**File:** `tests/unit/test_judge.py`

**Test Cases:**
- `test_judge_initialization`
- `test_evaluate_returns_dict`
- `test_calculate_reward_range`
- `test_calculate_reward_from_score`

## Integration Tests

Integration tests verify complete workflows and component interactions.

### Full Workflow Tests

**File:** `tests/integration/test_full_workflow.py`

**Test Cases:**
- `test_single_symbol_analysis`: Test complete single symbol analysis
- `test_batch_analysis`: Test batch analysis of multiple symbols
- `test_workflow_with_supervisor`: Test workflow with supervisor routing
- `test_error_handling_in_workflow`: Test error handling when agents fail

**Example:**

```python
def test_single_symbol_analysis(coordinator):
    """Test analyzing a single symbol"""
    # Mock agents
    coordinator.agents['news'] = Mock()
    coordinator.agents['news'].analyze.return_value = {
        'sentiment_score': 1.2,
        'confidence': 0.85,
        'recommendation': 'bullish'
    }
    
    # ... setup other mocks ...
    
    # Run analysis
    result = coordinator.analyze_symbol('AAPL')
    
    # Verify result structure
    assert 'symbol' in result
    assert 'recommendation' in result
    assert 'confidence' in result
```

### Coordinator Tests

**File:** `tests/integration/test_coordinator.py`

**Test Cases:**
- `test_coordinator_initialization`
- `test_portfolio_state_management`
- `test_agent_coordination`

### API Tests

**File:** `tests/integration/test_api.py`

**Test Cases:**
- `test_root_endpoint`
- `test_health_endpoint`
- `test_models_endpoint`
- `test_analyze_endpoint`
- `test_batch_endpoint`
- `test_invalid_request`

**Example:**

```python
def test_analyze_endpoint(client, mock_coordinator):
    """Test single analysis endpoint"""
    mock_coordinator.analyze_symbol.return_value = {
        'symbol': 'AAPL',
        'recommendation': 'buy',
        'confidence': 0.80,
        # ... other fields ...
    }
    
    with patch('api.server.coordinator', mock_coordinator):
        response = client.post(
            "/analyze",
            json={"symbol": "AAPL", "use_supervisor": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['symbol'] == 'AAPL'
```

## Test Coverage

### Current Coverage

- **Unit Tests:** 39 test cases
- **Integration Tests:** 14 test cases
- **Total:** 53 test cases

### Coverage by Component

| Component | Unit Tests | Integration Tests | Coverage |
|-----------|------------|-------------------|----------|
| News Agent | 8 | - | High |
| Technical Agent | 8 | - | High |
| Fundamental Agent | 6 | - | High |
| Strategist | 7 | - | High |
| Supervisor | 6 | - | Medium |
| Judge | 4 | - | Medium |
| Coordinator | - | 3 | High |
| Full Workflow | - | 5 | High |
| API | - | 6 | High |

### Generate Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

## Writing New Tests

### Test Template

```python
"""
Unit Tests for [Component Name]
"""

import pytest
from unittest.mock import Mock, patch
from [module] import [Component]


class Test[Component]:
    """Test suite for [Component]"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'param1': 'value1',
            'param2': 'value2'
        }
    
    @pytest.fixture
    def mock_component(self, mock_config):
        """Create mock component"""
        with patch('[dependency]'):
            component = [Component](config=mock_config)
            return component
    
    def test_initialization(self, mock_component):
        """Test component initializes correctly"""
        assert mock_component is not None
    
    def test_main_functionality(self, mock_component):
        """Test main functionality"""
        result = mock_component.method()
        assert result is not None
```

### Best Practices

1. **Use Fixtures:** Create reusable test fixtures for common setups
2. **Mock External Dependencies:** Use `unittest.mock` to mock external services
3. **Test Edge Cases:** Include tests for error conditions and edge cases
4. **Descriptive Names:** Use clear, descriptive test names
5. **Single Assertion:** Each test should verify one specific behavior
6. **Arrange-Act-Assert:** Follow the AAA pattern

### Example: Adding a New Test

```python
def test_new_feature(self, mock_agent):
    """Test new feature functionality"""
    # Arrange
    mock_agent.setup_feature()
    
    # Act
    result = mock_agent.new_feature('input')
    
    # Assert
    assert result['status'] == 'success'
    assert result['value'] > 0
```

## Continuous Integration

### GitHub Actions Configuration

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Install pre-commit hooks to run tests before commits:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
EOF

# Install hooks
pre-commit install
```

## Debugging Tests

### Run Tests in Debug Mode

```bash
# Run with pdb on failure
pytest --pdb

# Run with verbose output
pytest -vv

# Show print statements
pytest -s
```

### Using VSCode Debugger

Add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

## Performance Testing

### Benchmark Tests

```python
import pytest

@pytest.mark.benchmark
def test_analysis_performance(benchmark):
    """Benchmark analysis performance"""
    result = benchmark(coordinator.analyze_symbol, 'AAPL')
    assert result is not None
```

### Load Testing

Use `locust` for API load testing:

```python
from locust import HttpUser, task, between

class StockAnalysisUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def analyze_stock(self):
        self.client.post("/analyze", json={
            "symbol": "AAPL",
            "use_supervisor": False
        })
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Mock Not Working:**
```python
# Use correct import path
with patch('module.where.used.Class') as mock:
    # not 'module.where.defined.Class'
```

**Fixture Not Found:**
```python
# Ensure fixture is in conftest.py or same file
# Check fixture scope
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
