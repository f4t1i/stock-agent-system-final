"""
Pass/Fail Gates Evaluator - Quality gates for backtest reports

Evaluates backtest results against configurable thresholds from
report_gates_config.yaml and determines PASS/FAIL status.

Usage:
    evaluator = PassFailGatesEvaluator()
    result = evaluator.evaluate(backtest_metrics)
    if result.final_judgment == "FAIL":
        print(f"Backtest failed: {result.rejection_reason}")
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class GateResult:
    """Result of evaluating a single gate"""
    gate_name: str
    category: str  # performance, risk, trading, data_quality
    threshold: float
    actual_value: float
    comparison: str  # ">=", "<=", ">", "<", "=="
    passed: bool
    severity: str  # critical, high, medium, low
    description: str
    institutional_standard: Optional[float] = None


@dataclass
class PassFailGatesResult:
    """Complete pass/fail gates evaluation result"""
    backtest_id: str
    timestamp: str

    # Gate results
    gate_results: List[GateResult]

    # Summary
    total_gates: int
    passed_gates: int
    failed_gates: int
    critical_failures: int
    high_failures: int
    medium_failures: int

    # Final judgment
    final_judgment: str  # "PASS" or "FAIL"
    rejection_reason: Optional[str]
    warnings: List[str]


class PassFailGatesEvaluator:
    """
    Evaluates backtest results against quality gates.

    Loads gate configuration from report_gates_config.yaml and
    determines if backtest meets minimum quality standards.
    """

    def __init__(self, gates_config_path: Optional[str] = None):
        """
        Initialize gates evaluator.

        Args:
            gates_config_path: Path to report_gates_config.yaml
        """
        if gates_config_path is None:
            gates_config_path = Path(__file__).parent / "report_gates_config.yaml"

        with open(gates_config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info("Pass/Fail Gates Evaluator initialized")

    def evaluate(self, metrics: Dict) -> PassFailGatesResult:
        """
        Evaluate backtest metrics against all gates.

        Args:
            metrics: Backtest metrics dictionary

        Returns:
            PassFailGatesResult with judgment and details
        """
        gate_results = []

        # Evaluate performance gates
        perf_gates = self.config['performance_gates']
        for gate_name, gate_config in perf_gates.items():
            if gate_name == 'description':
                continue

            result = self._evaluate_gate(
                gate_name,
                'performance',
                gate_config,
                metrics
            )
            if result:
                gate_results.append(result)

        # Evaluate risk gates
        risk_gates = self.config['risk_gates']
        for gate_name, gate_config in risk_gates.items():
            if gate_name == 'description':
                continue

            result = self._evaluate_gate(
                gate_name,
                'risk',
                gate_config,
                metrics
            )
            if result:
                gate_results.append(result)

        # Evaluate trading gates
        trading_gates = self.config['trading_gates']
        for gate_name, gate_config in trading_gates.items():
            if gate_name == 'description':
                continue

            result = self._evaluate_gate(
                gate_name,
                'trading',
                gate_config,
                metrics
            )
            if result:
                gate_results.append(result)

        # Evaluate data quality gates
        data_gates = self.config['data_quality_gates']
        for gate_name, gate_config in data_gates.items():
            if gate_name == 'description':
                continue

            result = self._evaluate_gate(
                gate_name,
                'data_quality',
                gate_config,
                metrics
            )
            if result:
                gate_results.append(result)

        # Calculate summary
        total_gates = len(gate_results)
        passed_gates = sum(1 for g in gate_results if g.passed)
        failed_gates = total_gates - passed_gates

        critical_failures = sum(1 for g in gate_results if not g.passed and g.severity == 'critical')
        high_failures = sum(1 for g in gate_results if not g.passed and g.severity == 'high')
        medium_failures = sum(1 for g in gate_results if not g.passed and g.severity == 'medium')

        # Make final judgment
        final_judgment, rejection_reason, warnings = self._make_final_judgment(
            gate_results,
            critical_failures,
            high_failures,
            medium_failures
        )

        result = PassFailGatesResult(
            backtest_id=metrics.get('config', {}).get('symbols', ['unknown'])[0] if 'config' in metrics else 'unknown',
            timestamp=datetime.now().isoformat(),
            gate_results=gate_results,
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            critical_failures=critical_failures,
            high_failures=high_failures,
            medium_failures=medium_failures,
            final_judgment=final_judgment,
            rejection_reason=rejection_reason,
            warnings=warnings
        )

        logger.info(
            f"Gates evaluation: {final_judgment} "
            f"({passed_gates}/{total_gates} passed, {critical_failures} critical failures)"
        )

        return result

    def _evaluate_gate(
        self,
        gate_name: str,
        category: str,
        gate_config: Dict,
        metrics: Dict
    ) -> Optional[GateResult]:
        """Evaluate a single gate"""

        # Map gate names to metric keys
        metric_key_mapping = {
            'sharpe_ratio': 'sharpe_ratio',
            'max_drawdown': 'max_drawdown',
            'win_rate': 'win_rate',
            'profit_factor': 'profit_factor',
            'sortino_ratio': 'sortino_ratio',
            'calmar_ratio': 'calmar_ratio',
            'volatility': 'volatility',
            'downside_deviation': 'downside_deviation',
            'max_drawdown_duration': 'max_drawdown_duration_days',
            'min_trades': 'total_trades',
            'avg_trade_pnl': 'avg_trade_pnl',
            'avg_win_to_avg_loss_ratio': 'avg_win_to_avg_loss_ratio',
            'min_trading_days': 'trading_days',
            'failed_signals_rate': 'failed_signals_rate',
            'delisted_symbols_rate': 'delisted_symbols_rate'
        }

        metric_key = metric_key_mapping.get(gate_name)

        if not metric_key:
            logger.warning(f"Unknown gate: {gate_name}")
            return None

        # Get actual value
        actual_value = metrics.get(metric_key)

        if actual_value is None:
            # Calculate derived metrics
            if gate_name == 'avg_win_to_avg_loss_ratio':
                avg_win = metrics.get('avg_win', 0)
                avg_loss = metrics.get('avg_loss', 0)
                actual_value = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            elif gate_name == 'failed_signals_rate':
                failed_signals = metrics.get('failed_signals', 0)
                total_signals = metrics.get('num_signals', 1)
                actual_value = failed_signals / total_signals if total_signals > 0 else 0

            elif gate_name == 'delisted_symbols_rate':
                delisted = len(metrics.get('delisted_symbols', []))
                total_symbols = len(metrics.get('config', {}).get('symbols', []))
                actual_value = delisted / total_symbols if total_symbols > 0 else 0

        if actual_value is None:
            logger.warning(f"Metric not found: {metric_key} for gate {gate_name}")
            return None

        # Evaluate gate
        threshold = gate_config['threshold']
        comparison = gate_config['comparison']

        passed = self._compare(actual_value, threshold, comparison)

        return GateResult(
            gate_name=gate_name,
            category=category,
            threshold=threshold,
            actual_value=actual_value,
            comparison=comparison,
            passed=passed,
            severity=gate_config['severity'],
            description=gate_config['description'],
            institutional_standard=gate_config.get('institutional_standard')
        )

    def _compare(self, actual: float, threshold: float, comparison: str) -> bool:
        """Compare actual value against threshold"""
        if comparison == '>=':
            return actual >= threshold
        elif comparison == '<=':
            return actual <= threshold
        elif comparison == '>':
            return actual > threshold
        elif comparison == '<':
            return actual < threshold
        elif comparison == '==':
            return abs(actual - threshold) < 1e-9
        else:
            logger.error(f"Unknown comparison operator: {comparison}")
            return False

    def _make_final_judgment(
        self,
        gate_results: List[GateResult],
        critical_failures: int,
        high_failures: int,
        medium_failures: int
    ) -> Tuple[str, Optional[str], List[str]]:
        """
        Make final pass/fail judgment.

        Logic from config:
        - Any critical failure → FAIL
        - High severity failures > 0 → FAIL
        - Medium severity failures > 2 → FAIL
        - Otherwise → PASS (with warnings)
        """
        warnings = []

        # Critical failures always reject
        if critical_failures > 0:
            critical_gates = [g.gate_name for g in gate_results if not g.passed and g.severity == 'critical']
            return "FAIL", f"Critical gate failures: {', '.join(critical_gates)}", []

        # High severity failures
        high_threshold = self.config['composite_judgment']['decision_logic']['high_severity_threshold']
        if high_failures > high_threshold:
            high_gates = [g.gate_name for g in gate_results if not g.passed and g.severity == 'high']
            return "FAIL", f"Too many high-severity failures ({high_failures}): {', '.join(high_gates)}", []

        # Medium severity failures
        medium_threshold = self.config['composite_judgment']['decision_logic']['medium_severity_threshold']
        if medium_failures > medium_threshold:
            medium_gates = [g.gate_name for g in gate_results if not g.passed and g.severity == 'medium']
            return "FAIL", f"Too many medium-severity failures ({medium_failures}): {', '.join(medium_gates)}", []

        # Collect warnings for passed backtest
        if medium_failures > 0:
            for gate in gate_results:
                if not gate.passed and gate.severity == 'medium':
                    warnings.append(f"{gate.gate_name}: {gate.actual_value:.4f} (threshold: {gate.threshold})")

        return "PASS", None, warnings

    def to_dict(self, result: PassFailGatesResult) -> Dict:
        """Convert result to dictionary"""
        return {
            'backtest_id': result.backtest_id,
            'timestamp': result.timestamp,
            'gate_results': [
                {
                    'gate_name': g.gate_name,
                    'category': g.category,
                    'threshold': g.threshold,
                    'actual_value': g.actual_value,
                    'comparison': g.comparison,
                    'passed': g.passed,
                    'severity': g.severity,
                    'description': g.description,
                    'institutional_standard': g.institutional_standard
                }
                for g in result.gate_results
            ],
            'summary': {
                'total_gates': result.total_gates,
                'passed_gates': result.passed_gates,
                'failed_gates': result.failed_gates,
                'critical_failures': result.critical_failures,
                'high_failures': result.high_failures,
                'medium_failures': result.medium_failures
            },
            'final_judgment': result.final_judgment,
            'rejection_reason': result.rejection_reason,
            'warnings': result.warnings
        }

    def get_exit_code(self, result: PassFailGatesResult) -> int:
        """Get exit code for CI/CD integration (0=PASS, 1=FAIL)"""
        return 0 if result.final_judgment == "PASS" else 1


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*60)
    print("PASS/FAIL GATES EVALUATOR TEST")
    print("="*60 + "\n")

    # Sample backtest metrics (good performance)
    good_metrics = {
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.12,
        'win_rate': 0.58,
        'profit_factor': 2.1,
        'sortino_ratio': 1.8,
        'calmar_ratio': 2.5,
        'volatility': 0.22,
        'downside_deviation': 0.15,
        'max_drawdown_duration_days': 45,
        'total_trades': 50,
        'avg_trade_pnl': 250.50,
        'avg_win': 450.0,
        'avg_loss': -200.0,
        'trading_days': 252,
        'failed_signals': 2,
        'num_signals': 52,
        'delisted_symbols': [],
        'config': {'symbols': ['AAPL', 'MSFT', 'GOOGL']}
    }

    evaluator = PassFailGatesEvaluator()
    result = evaluator.evaluate(good_metrics)

    print(f"Backtest: {result.backtest_id}")
    print(f"Final Judgment: {result.final_judgment}")
    print(f"Gates: {result.passed_gates}/{result.total_gates} passed")
    print(f"Failures: {result.critical_failures} critical, {result.high_failures} high, {result.medium_failures} medium")

    if result.final_judgment == "FAIL":
        print(f"\nRejection Reason: {result.rejection_reason}")

    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")

    print(f"\nGate Results:")
    for gate in result.gate_results:
        status = "✅ PASS" if gate.passed else "❌ FAIL"
        print(f"  {gate.gate_name:30s} {status:10s} ({gate.actual_value:.4f} {gate.comparison} {gate.threshold})")

    print("\n" + "="*60 + "\n")

    # Test with poor performance
    print("Testing with poor performance metrics...\n")

    poor_metrics = good_metrics.copy()
    poor_metrics['sharpe_ratio'] = 0.3  # Critical failure
    poor_metrics['max_drawdown'] = 0.35  # Critical failure
    poor_metrics['win_rate'] = 0.42  # High failure

    result_poor = evaluator.evaluate(poor_metrics)
    print(f"Final Judgment: {result_poor.final_judgment}")
    print(f"Rejection Reason: {result_poor.rejection_reason}")

    print("\n" + "="*60 + "\n")
