# API Documentation

Complete REST API documentation for the Stock Analysis Multi-Agent System.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Models Info](#models-info)
  - [Single Analysis](#single-analysis)
  - [Batch Analysis](#batch-analysis)
  - [Backtesting](#backtesting)
- [Request/Response Schemas](#requestresponse-schemas)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

The Stock Analysis API provides programmatic access to the multi-agent stock analysis system. It supports:

- Single symbol analysis
- Batch processing of multiple symbols
- Historical backtesting
- Model information queries
- Health monitoring

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your domain.

## Authentication

Currently, the API does not require authentication. For production use, implement:

- API key authentication
- JWT tokens
- OAuth 2.0

## Endpoints

### Health Check

Check system health and agent availability.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agents_loaded": {
    "news": true,
    "technical": true,
    "fundamental": true,
    "supervisor": false,
    "strategist": true
  },
  "timestamp": "2024-01-04T12:00:00"
}
```

### Models Info

Get information about available models and configuration.

**Endpoint:** `GET /models`

**Response:**

```json
{
  "models": [
    {
      "agent_type": "news",
      "model_path": "models/news_agent_v1",
      "enabled": true,
      "version": null
    },
    {
      "agent_type": "technical",
      "model_path": "models/technical_agent_v1",
      "enabled": true,
      "version": null
    }
  ],
  "supervisor_enabled": false,
  "timestamp": "2024-01-04T12:00:00"
}
```

### Single Analysis

Analyze a single stock symbol.

**Endpoint:** `POST /analyze`

**Request Body:**

```json
{
  "symbol": "AAPL",
  "use_supervisor": false,
  "lookback_days": 7
}
```

**Parameters:**

- `symbol` (string, required): Stock ticker symbol
- `use_supervisor` (boolean, optional): Enable supervisor routing (default: false)
- `lookback_days` (integer, optional): Days to look back for news (default: 7, range: 1-90)

**Response:**

```json
{
  "symbol": "AAPL",
  "recommendation": "buy",
  "confidence": 0.85,
  "reasoning": "Strong bullish signals from all agents with positive news sentiment and technical momentum.",
  "position_size": 0.08,
  "entry_target": 185.50,
  "stop_loss": 178.00,
  "take_profit": 195.00,
  "risk_assessment": "Moderate risk with favorable risk/reward ratio of 1:2.5",
  "agent_outputs": {
    "news": {
      "sentiment_score": 1.5,
      "confidence": 0.85,
      "key_events": ["New product launch", "Positive earnings"],
      "recommendation": "bullish"
    },
    "technical": {
      "signal": "bullish",
      "signal_strength": 0.75,
      "support_levels": [180.0, 175.0],
      "resistance_levels": [190.0, 195.0],
      "recommendation": "buy"
    },
    "fundamental": {
      "valuation": "fairly_valued",
      "financial_health_score": 0.80,
      "growth_score": 0.75,
      "recommendation": "hold"
    }
  },
  "strategist_output": {
    "decision": "buy",
    "confidence": 0.85,
    "position_size": 0.08,
    "reasoning": "...",
    "risk_assessment": "..."
  },
  "timestamp": "2024-01-04T12:00:00",
  "errors": []
}
```

### Batch Analysis

Analyze multiple stock symbols in batch.

**Endpoint:** `POST /batch`

**Request Body:**

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "use_supervisor": false,
  "lookback_days": 7
}
```

**Parameters:**

- `symbols` (array of strings, required): List of stock ticker symbols
- `use_supervisor` (boolean, optional): Enable supervisor routing (default: false)
- `lookback_days` (integer, optional): Days to look back for news (default: 7, range: 1-90)

**Response:**

```json
{
  "results": [
    {
      "symbol": "AAPL",
      "recommendation": "buy",
      "confidence": 0.85,
      "reasoning": "...",
      "position_size": 0.08,
      "entry_target": 185.50,
      "stop_loss": 178.00,
      "take_profit": 195.00,
      "risk_assessment": "...",
      "agent_outputs": {},
      "strategist_output": {},
      "timestamp": "2024-01-04T12:00:00",
      "errors": []
    }
  ],
  "total_analyzed": 3,
  "successful": 3,
  "failed": 0,
  "timestamp": "2024-01-04T12:00:00"
}
```

### Backtesting

Run historical backtesting.

**Endpoint:** `POST /backtest`

**Request Body:**

```json
{
  "symbols": ["AAPL", "MSFT"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 100000.0,
  "use_supervisor": false
}
```

**Parameters:**

- `symbols` (array of strings, required): List of stock ticker symbols
- `start_date` (string, required): Start date in YYYY-MM-DD format
- `end_date` (string, required): End date in YYYY-MM-DD format
- `initial_capital` (float, optional): Initial capital (default: 100000.0)
- `use_supervisor` (boolean, optional): Enable supervisor routing (default: false)

**Response:**

```json
{
  "symbols": ["AAPL", "MSFT"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 100000.0,
  "metrics": {
    "total_return": 0.25,
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.1,
    "max_drawdown": 0.12,
    "win_rate": 0.65,
    "profit_factor": 2.3,
    "total_trades": 45,
    "winning_trades": 29,
    "losing_trades": 16,
    "avg_win": 0.08,
    "avg_loss": 0.04,
    "final_portfolio_value": 125000.0
  },
  "trades": [],
  "equity_curve": [],
  "timestamp": "2024-01-04T12:00:00"
}
```

## Request/Response Schemas

### Common Types

**Recommendation Values:**
- `"buy"` / `"bullish"`: Positive outlook
- `"sell"` / `"bearish"`: Negative outlook
- `"hold"` / `"neutral"`: Neutral outlook

**Confidence Range:** `0.0` to `1.0`

**Position Size Range:** `0.0` to `1.0` (as fraction of portfolio)

**Sentiment Score Range:** `-2.0` to `2.0`

**Signal Strength Range:** `0.0` to `1.0`

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-04T12:00:00"
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

### Common Errors

**Invalid Symbol:**
```json
{
  "error": "Invalid symbol",
  "detail": "Symbol 'INVALID' not found"
}
```

**Validation Error:**
```json
{
  "error": "Validation error",
  "detail": "lookback_days must be between 1 and 90"
}
```

## Rate Limiting

Currently not implemented. For production:

- Implement rate limiting per IP/API key
- Suggested limits:
  - 100 requests per minute for single analysis
  - 10 requests per minute for batch analysis
  - 5 requests per hour for backtesting

## Examples

### Python Example

```python
import requests

# Single analysis
response = requests.post(
    'http://localhost:8000/analyze',
    json={
        'symbol': 'AAPL',
        'use_supervisor': False,
        'lookback_days': 7
    }
)

result = response.json()
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}")
print(f"Position Size: {result['position_size']}")
```

### cURL Example

```bash
# Health check
curl http://localhost:8000/health

# Single analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "use_supervisor": false,
    "lookback_days": 7
  }'

# Batch analysis
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "use_supervisor": false,
    "lookback_days": 7
  }'
```

### JavaScript Example

```javascript
// Using fetch API
async function analyzeStock(symbol) {
  const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbol: symbol,
      use_supervisor: false,
      lookback_days: 7
    })
  });
  
  const result = await response.json();
  console.log('Recommendation:', result.recommendation);
  console.log('Confidence:', result.confidence);
  return result;
}

analyzeStock('AAPL');
```

## Interactive Documentation

Access interactive API documentation at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

These provide:
- Interactive API testing
- Request/response examples
- Schema validation
- Authentication testing

## Deployment

### Running the Server

```bash
# Development
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Production
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
CONFIG_PATH=config/system.yaml
LOG_LEVEL=INFO
```

## Support

For issues or questions:
- GitHub Issues: [repository]/issues
- Email: support@example.com
- Documentation: [repository]/docs
