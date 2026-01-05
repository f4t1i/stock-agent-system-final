# Signal Contract - Trading Signal Specification

**Version:** 1.0.0
**Status:** Production-Ready
**Compliance:** Institutional-Grade Validation

---

## Overview

The **Signal Contract** is a standardized JSON schema that enforces structured, validated trading signals across the multi-agent stock trading system. It ensures:

✅ **Reproducibility** - Deterministic signal format
✅ **Traceability** - Full audit trail with evidence
✅ **Risk Management** - Mandatory risk parameters
✅ **Quality Assurance** - Automated validation gates

---

## Signal Structure

```
Signal Contract
├── analysis          # Multi-agent analysis results
│   ├── news          # News sentiment analysis
│   ├── technical     # Technical indicators & signals
│   └── fundamental   # Fundamental valuation
├── signal            # Final decision (buy/sell/hold)
├── sizing            # Position sizing & rationale
├── risk              # Risk management parameters
├── rationale         # Comprehensive reasoning
├── evidence          # Supporting sources & confidence
└── metadata          # Tracking & versioning info
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install jsonschema loguru
```

### 2. Validate a Signal

```python
from contracts.signal_validator import validate_signal

signal = {
    "analysis": {...},
    "signal": "buy",
    "sizing": {...},
    "risk": {...},
    "rationale": "...",
    "evidence": {...},
    "metadata": {...}
}

# Validate
is_valid, errors = validate_signal(signal, strict=False)

if is_valid:
    print("✅ Signal is valid!")
else:
    print(f"❌ Validation failed:")
    for error in errors:
        print(f"  - {error}")
```

### 3. Validate from File

```python
from contracts.signal_validator import validate_signal_file

is_valid, errors = validate_signal_file("signal.json", strict=False)
```

---

## Validation Layers

### Layer 1: JSON Schema Validation

- **Data types** (string, number, object, array)
- **Required fields** (mandatory properties)
- **Value ranges** (min/max, enum constraints)
- **Format validation** (date-time, URI, regex patterns)

### Layer 2: Business Rules Validation

- **Price sanity checks** (stop loss < current price for buy)
- **Indicator ranges** (RSI ∈ [0, 100], Bollinger bands ordering)
- **Position sizing logic** (hold signal → position_size = 0)
- **Risk/reward ratios** (institutional standard: R/R ≥ 1.5)

### Layer 3: Cross-Field Validation

- **Agent consensus** (technical vs final signal alignment)
- **Timestamp freshness** (signals < 1 hour old)
- **Confidence vs position size** (low confidence → small size)

---

## Field Specifications

### 📊 Analysis

#### News Analysis
```json
{
  "sentiment_score": 1.5,        // -2 (very bearish) to +2 (very bullish)
  "confidence": 0.85,            // 0.0 - 1.0
  "key_events": [                // Up to 10 events
    "Strong earnings beat",
    "New product launch"
  ],
  "sources": [                   // Optional news sources
    {
      "title": "...",
      "url": "https://...",
      "sentiment": 1.8,
      "timestamp": "2026-01-05T10:30:00Z"
    }
  ]
}
```

#### Technical Analysis
```json
{
  "signal": "bullish",           // bullish | bearish | neutral
  "signal_strength": 0.8,        // 0.0 - 1.0
  "indicators": {
    "rsi": 65,                   // 0 - 100
    "macd": {
      "value": 2.5,
      "signal": 2.0,
      "histogram": 0.5
    },
    "sma_20": 148.5,
    "sma_50": 145.0,
    "sma_200": 140.0,
    "bollinger_bands": {
      "upper": 155.0,
      "middle": 150.0,           // lower < middle < upper
      "lower": 145.0
    }
  },
  "support_levels": [145.0, 142.0],
  "resistance_levels": [155.0, 160.0],
  "patterns": [
    {
      "name": "Cup and Handle",
      "type": "bullish",
      "confidence": 0.75
    }
  ]
}
```

#### Fundamental Analysis
```json
{
  "valuation": "undervalued",    // undervalued | fairly_valued | overvalued
  "financial_health_score": 0.9, // 0.0 - 1.0
  "growth_score": 0.85,          // 0.0 - 1.0
  "key_metrics": {
    "pe_ratio": 22.5,
    "pb_ratio": 3.2,
    "debt_to_equity": 0.45,
    "roe": 0.28,
    "roa": 0.15,
    "current_ratio": 1.8,
    "revenue_growth_yoy": 0.18,
    "earnings_growth_yoy": 0.22
  }
}
```

### 🎯 Signal

```json
"signal": "buy"  // buy | sell | hold
```

### 📏 Sizing

```json
{
  "position_size": 0.15,         // 0.0 - 1.0 (fraction of portfolio)
  "rationale": "Strong fundamentals with bullish technical setup justify 15% position.",
  "kelly_fraction": 0.18,        // Optional Kelly Criterion
  "max_position_value": 15000    // Optional max value in USD
}
```

**Rules:**
- `hold` → `position_size` must be 0
- `buy/sell` → `position_size` must be > 0
- Low confidence → small position size recommended

### ⚠️ Risk

```json
{
  "stop_loss": 145.0,            // Stop loss price level
  "take_profit": 165.0,          // Take profit price level
  "max_drawdown": 0.05,          // 0.0 - 1.0 (5% max drawdown)
  "risk_reward_ratio": 3.0,      // Reward / Risk
  "time_horizon": "short_term",  // intraday | swing | short_term | medium_term | long_term
  "volatility_adjusted": true    // Risk params adjusted for volatility
}
```

**Rules:**
- Buy signal: `stop_loss < current_price < take_profit`
- Sell signal: `stop_loss > current_price > take_profit`
- Institutional standard: `risk_reward_ratio ≥ 1.5`

### 📝 Rationale

```json
"rationale": "Based on strong earnings beat (+15% vs expectations), bullish technical breakout above $150 resistance with cup-and-handle pattern confirmation, and favorable news sentiment (score: 1.5), we recommend BUY with 15% position size..."
```

**Requirements:**
- Minimum 50 characters
- Maximum 2000 characters
- Must explain: Why this signal? What evidence supports it? How were risk parameters determined?

### 🔍 Evidence

```json
{
  "sources": [
    {
      "type": "earnings_report",
      "description": "Q4 2025 earnings beat by 15%",
      "url": "https://example.com/earnings",
      "timestamp": "2026-01-05T08:00:00Z",
      "weight": 0.9
    }
  ],
  "confidence": 0.85,            // Overall confidence 0.0 - 1.0
  "consensus": {
    "agent_agreement": 0.88,     // Agreement between agents
    "conflicting_signals": []    // Any conflicts flagged
  }
}
```

**Source Types:**
- `news`
- `technical_indicator`
- `fundamental_metric`
- `market_data`
- `analyst_report`
- `earnings_report`

**Requirements:**
- Minimum 1 source, maximum 20
- Each source must have `type` and `description`

### 📋 Metadata

```json
{
  "symbol": "AAPL",              // Uppercase, 1-5 letters
  "timestamp": "2026-01-05T10:45:00Z",
  "version": "1.0.0",            // Semver
  "agent_versions": {
    "news_agent": "v1.2.3",
    "technical_agent": "v1.4.1",
    "fundamental_agent": "v1.3.0",
    "strategist": "v2.0.1"
  },
  "execution_time_ms": 3450,
  "market_regime": "bull",       // bull | bear | sideways | high_volatility | low_volatility
  "current_price": 150.0
}
```

---

## Examples

### ✅ Valid Signal

See: [`examples/valid_buy_signal.json`](examples/valid_buy_signal.json)

### ❌ Invalid Signal (with errors)

See: [`examples/invalid_signal.json`](examples/invalid_signal.json)

**Errors:**
- `sentiment_score` = 2.5 (exceeds max 2.0)
- `confidence` = 1.2 (exceeds max 1.0)
- `rsi` = 150 (exceeds max 100)
- `position_size` = 1.5 (exceeds max 1.0)
- `rationale` too short (< 50 chars)
- `stop_loss` (155) > `current_price` (150) for buy signal
- `take_profit` (145) < `current_price` (150) for buy signal
- Bollinger bands ordering violated (upper < lower)
- Empty evidence sources (min 1 required)

---

## Validation Reports

Get comprehensive validation report:

```python
from contracts.signal_validator import SignalValidator

validator = SignalValidator()
report = validator.get_validation_report(signal)

print(report)
# {
#   'is_valid': True,
#   'error_count': 0,
#   'errors': [],
#   'validated_at': '2026-01-05T10:45:00Z',
#   'signal_metadata': {...},
#   'validation_summary': {
#     'schema_valid': True,
#     'business_rules_valid': True
#   }
# }
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Validate Trading Signals
  run: |
    python -m contracts.signal_validator examples/valid_buy_signal.json
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

for signal_file in contracts/examples/*.json; do
  python -c "from contracts.signal_validator import validate_signal_file; validate_signal_file('$signal_file', strict=True)"
done
```

---

## Schema Versioning

**Current Version:** 1.0.0

**Changelog:**
- `1.0.0` (2026-01-05): Initial release with full validation

**Migration Guide:**
- Future schema changes will follow semver
- Breaking changes = major version bump
- New optional fields = minor version bump

---

## FAQ

**Q: Can I add custom fields?**
A: No, `additionalProperties: false` is enforced for strict contract compliance.

**Q: What if my signal doesn't pass validation?**
A: Fix the errors reported by the validator. All fields are required for institutional-grade compliance.

**Q: Can I disable certain validation rules?**
A: Use `check_business_rules=False` to skip business logic validation (not recommended for production).

**Q: How do I update the schema?**
A: Edit `signal_schema.json`, increment version, update this README, and add migration notes.

---

## Related Documentation

- [Backtest Harness](../training/rl/README.md)
- [Judge Rubrics](../evaluation/README.md)
- [Data Synthesis](../data_pipeline/README.md)

---

## License

MIT License - See [../LICENSE](../LICENSE)
