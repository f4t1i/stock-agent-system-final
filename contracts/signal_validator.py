"""
Signal Validator - Validates trading signals against Signal Contract JSON Schema

This module provides:
- JSON Schema validation for trading signals
- Hard validation rules (format, data types, ranges)
- Business logic validation (price sanity checks, risk parameters)
- Comprehensive error reporting

Based on TradingAgents-style contracts and institutional-grade validation.
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
from loguru import logger


class SignalValidationError(Exception):
    """Raised when signal validation fails"""
    pass


class SignalValidator:
    """
    Validates trading signals against Signal Contract JSON Schema.

    Features:
    - JSON Schema validation
    - Hard business rules (price sanity, indicator ranges)
    - Comprehensive error reporting
    - Performance tracking
    """

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize Signal Validator.

        Args:
            schema_path: Path to signal schema JSON file
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "signal_schema.json"
        else:
            schema_path = Path(schema_path)

        # Load schema
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        # Create validator
        self.validator = Draft7Validator(self.schema)

        logger.info(f"Signal Validator initialized with schema: {schema_path}")

    def validate(
        self,
        signal: Dict,
        strict: bool = True,
        check_business_rules: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a trading signal.

        Args:
            signal: Trading signal dictionary
            strict: If True, raise exception on validation failure
            check_business_rules: If True, apply business logic validation

        Returns:
            Tuple of (is_valid, errors_list)

        Raises:
            SignalValidationError: If strict=True and validation fails
        """
        errors = []

        # 1. JSON Schema Validation
        schema_errors = self._validate_schema(signal)
        errors.extend(schema_errors)

        # 2. Business Rules Validation
        if check_business_rules:
            business_errors = self._validate_business_rules(signal)
            errors.extend(business_errors)

        # 3. Cross-field Validation
        cross_field_errors = self._validate_cross_fields(signal)
        errors.extend(cross_field_errors)

        is_valid = len(errors) == 0

        if not is_valid:
            error_summary = f"Signal validation failed with {len(errors)} error(s):\n" + \
                          "\n".join(f"  - {err}" for err in errors)
            logger.warning(error_summary)

            if strict:
                raise SignalValidationError(error_summary)

        return is_valid, errors

    def _validate_schema(self, signal: Dict) -> List[str]:
        """Validate against JSON Schema"""
        errors = []

        try:
            validate(instance=signal, schema=self.schema)
        except ValidationError as e:
            # Parse validation error
            error_path = " -> ".join(str(p) for p in e.path) if e.path else "root"
            errors.append(f"Schema error at {error_path}: {e.message}")

        # Check all errors (not just first)
        for error in self.validator.iter_errors(signal):
            error_path = " -> ".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"Schema error at {error_path}: {error.message}")

        return list(set(errors))  # Deduplicate

    def _validate_business_rules(self, signal: Dict) -> List[str]:
        """Validate business logic rules"""
        errors = []

        # 1. Price Sanity Checks
        if 'metadata' in signal and 'current_price' in signal['metadata']:
            current_price = signal['metadata']['current_price']

            # Stop loss should be below current price for buy, above for sell
            if 'risk' in signal:
                stop_loss = signal['risk'].get('stop_loss', 0)
                take_profit = signal['risk'].get('take_profit', 0)

                if signal['signal'] == 'buy':
                    if stop_loss >= current_price:
                        errors.append(
                            f"Buy signal: stop_loss ({stop_loss}) must be < current_price ({current_price})"
                        )
                    if take_profit <= current_price:
                        errors.append(
                            f"Buy signal: take_profit ({take_profit}) must be > current_price ({current_price})"
                        )

                elif signal['signal'] == 'sell':
                    if stop_loss <= current_price:
                        errors.append(
                            f"Sell signal: stop_loss ({stop_loss}) must be > current_price ({current_price})"
                        )
                    if take_profit >= current_price:
                        errors.append(
                            f"Sell signal: take_profit ({take_profit}) must be < current_price ({current_price})"
                        )

        # 2. Indicator Sanity Checks
        if 'analysis' in signal and 'technical' in signal['analysis']:
            indicators = signal['analysis']['technical'].get('indicators', {})

            # RSI range check (redundant with schema, but explicit)
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if not (0 <= rsi <= 100):
                    errors.append(f"RSI ({rsi}) must be in range [0, 100]")

            # SMA ordering check (SMA_20 < SMA_50 < SMA_200 is common but not required)
            # Bollinger Bands check (lower < middle < upper)
            if 'bollinger_bands' in indicators:
                bb = indicators['bollinger_bands']
                if 'lower' in bb and 'middle' in bb and 'upper' in bb:
                    if not (bb['lower'] < bb['middle'] < bb['upper']):
                        errors.append(
                            f"Bollinger Bands: lower ({bb['lower']}) < middle ({bb['middle']}) < upper ({bb['upper']}) must hold"
                        )

        # 3. Position Sizing Sanity
        if 'sizing' in signal:
            position_size = signal['sizing']['position_size']

            # Hold signal should have position_size = 0
            if signal['signal'] == 'hold' and position_size > 0:
                errors.append(
                    f"Hold signal should have position_size = 0, got {position_size}"
                )

            # Buy/Sell should have position_size > 0
            if signal['signal'] in ['buy', 'sell'] and position_size == 0:
                errors.append(
                    f"{signal['signal'].upper()} signal should have position_size > 0, got {position_size}"
                )

        # 4. Risk/Reward Ratio Check
        if 'risk' in signal and 'metadata' in signal:
            current_price = signal['metadata'].get('current_price', 0)
            stop_loss = signal['risk'].get('stop_loss', 0)
            take_profit = signal['risk'].get('take_profit', 0)

            if current_price > 0 and stop_loss > 0 and take_profit > 0:
                if signal['signal'] == 'buy':
                    risk = current_price - stop_loss
                    reward = take_profit - current_price

                    if risk > 0:
                        rr_ratio = reward / risk

                        # Institutional standard: R/R >= 1.5
                        if rr_ratio < 1.0:
                            errors.append(
                                f"Poor risk/reward ratio ({rr_ratio:.2f}). Institutional standard: >= 1.5"
                            )

        # 5. Confidence vs Position Size Alignment
        if 'evidence' in signal and 'sizing' in signal:
            confidence = signal['evidence'].get('confidence', 0)
            position_size = signal['sizing']['position_size']

            # Low confidence should have small position size
            if confidence < 0.5 and position_size > 0.2:
                errors.append(
                    f"Low confidence ({confidence}) with large position_size ({position_size}). Risk mismatch."
                )

        return errors

    def _validate_cross_fields(self, signal: Dict) -> List[str]:
        """Validate cross-field consistency"""
        errors = []

        # 1. Agent Consensus Check
        if 'analysis' in signal:
            analysis = signal['analysis']

            # Check if technical signal aligns with final signal
            if 'technical' in analysis and 'signal' in signal:
                tech_signal = analysis['technical'].get('signal', 'neutral')
                final_signal = signal['signal']

                # Map technical to final
                tech_to_final = {
                    'bullish': 'buy',
                    'bearish': 'sell',
                    'neutral': 'hold'
                }

                expected_signal = tech_to_final.get(tech_signal, 'hold')

                # Not a hard error, but flag divergence
                if expected_signal != final_signal and signal['evidence'].get('confidence', 0) > 0.8:
                    # High confidence but divergent signals = potential issue
                    errors.append(
                        f"High confidence ({signal['evidence'].get('confidence', 0)}) but technical signal ({tech_signal}) diverges from final ({final_signal})"
                    )

        # 2. Timestamp Freshness
        if 'metadata' in signal and 'timestamp' in signal['metadata']:
            try:
                signal_time = datetime.fromisoformat(signal['metadata']['timestamp'].replace('Z', '+00:00'))
                now = datetime.now(signal_time.tzinfo)
                age_seconds = (now - signal_time).total_seconds()

                # Signals older than 1 hour are stale
                if age_seconds > 3600:
                    errors.append(
                        f"Signal is stale (age: {age_seconds / 60:.1f} minutes). Signals should be < 1 hour old."
                    )
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid timestamp format: {e}")

        return errors

    def validate_file(self, filepath: str, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate a signal from JSON file.

        Args:
            filepath: Path to signal JSON file
            strict: If True, raise exception on validation failure

        Returns:
            Tuple of (is_valid, errors_list)
        """
        with open(filepath, 'r') as f:
            signal = json.load(f)

        return self.validate(signal, strict=strict)

    def get_validation_report(self, signal: Dict) -> Dict:
        """
        Get comprehensive validation report.

        Args:
            signal: Trading signal dictionary

        Returns:
            Validation report dictionary
        """
        is_valid, errors = self.validate(signal, strict=False)

        report = {
            'is_valid': is_valid,
            'error_count': len(errors),
            'errors': errors,
            'validated_at': datetime.now().isoformat(),
            'signal_metadata': signal.get('metadata', {}),
            'validation_summary': {
                'schema_valid': len([e for e in errors if 'Schema error' in e]) == 0,
                'business_rules_valid': len([e for e in errors if 'Schema error' not in e]) == 0,
            }
        }

        return report


# Convenience functions
def validate_signal(signal: Dict, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate a trading signal.

    Args:
        signal: Trading signal dictionary
        strict: If True, raise exception on validation failure

    Returns:
        Tuple of (is_valid, errors_list)
    """
    validator = SignalValidator()
    return validator.validate(signal, strict=strict)


def validate_signal_file(filepath: str, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate a signal from JSON file.

    Args:
        filepath: Path to signal JSON file
        strict: If True, raise exception on validation failure

    Returns:
        Tuple of (is_valid, errors_list)
    """
    validator = SignalValidator()
    return validator.validate_file(filepath, strict=strict)


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timezone

    # Create test signal
    test_signal = {
        "analysis": {
            "news": {
                "sentiment_score": 1.5,
                "confidence": 0.85,
                "key_events": ["Strong earnings beat", "New product launch"]
            },
            "technical": {
                "signal": "bullish",
                "signal_strength": 0.8,
                "indicators": {
                    "rsi": 65,
                    "macd": {"value": 2.5, "signal": 2.0, "histogram": 0.5}
                }
            },
            "fundamental": {
                "valuation": "undervalued",
                "financial_health_score": 0.9,
                "growth_score": 0.85
            }
        },
        "signal": "buy",
        "sizing": {
            "position_size": 0.15,
            "rationale": "Strong fundamentals with bullish technical setup justify 15% position."
        },
        "risk": {
            "stop_loss": 145.0,
            "take_profit": 165.0,
            "max_drawdown": 0.05
        },
        "rationale": "Based on strong earnings beat (+15% vs expectations), bullish technical breakout above $150 resistance, and favorable news sentiment (score: 1.5), we recommend BUY with 15% position size. Stop loss at $145 (-3.3%), take profit at $165 (+10%).",
        "evidence": {
            "sources": [
                {
                    "type": "earnings_report",
                    "description": "Q4 earnings beat by 15%",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "confidence": 0.85
        },
        "metadata": {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "current_price": 150.0
        }
    }

    # Validate
    validator = SignalValidator()
    is_valid, errors = validator.validate(test_signal, strict=False)

    print(f"\n{'='*60}")
    print("SIGNAL VALIDATION TEST")
    print(f"{'='*60}\n")
    print(f"Valid: {is_valid}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("\nNo errors - Signal is valid!")

    # Get full report
    report = validator.get_validation_report(test_signal)
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}\n")
    print(json.dumps(report, indent=2))
