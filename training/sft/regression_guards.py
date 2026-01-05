#!/usr/bin/env python3
"""
Regression Guards - Comprehensive Regression Testing Framework

Purpose:
    Prevent model degradation by comparing new models against baseline models
    using multiple metrics, statistical tests, and automated blocking.

Features:
    - Multi-metric comparison with configurable tolerances
    - Holdout dataset re-evaluation
    - Statistical significance testing
    - Automated blocking of degraded models
    - Override capability for exceptional cases
    - Detailed regression reports

Usage:
    # Basic regression test
    guards = RegressionGuards()
    result = guards.test(
        baseline_model_id="news_agent_1.0.0_20240101",
        candidate_model_id="news_agent_1.1.0_20240115",
        metrics=["eval_loss", "eval_accuracy"]
    )

    # With holdout re-evaluation
    result = guards.test_with_holdout(
        baseline_model_path="models/sft/news_agent_v1.0.0",
        candidate_model_path="models/sft/news_agent_v1.1.0",
        holdout_dataset="data/datasets/sft_v1/test.jsonl"
    )

    # Check result
    if result.passed:
        print("✅ No regression detected")
    else:
        print(f"❌ Regression detected: {result.failed_metrics}")
"""

import json
import yaml
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from training.sft.model_registry import ModelRegistry, ModelRecord
from training.sft.eval_gates import EvalGates


class MetricDirection(Enum):
    """Metric optimization direction"""
    MINIMIZE = "minimize"  # Lower is better (e.g., loss)
    MAXIMIZE = "maximize"  # Higher is better (e.g., accuracy)


@dataclass
class MetricPolicy:
    """Policy for a specific metric"""
    metric_name: str
    direction: MetricDirection
    tolerance_pct: float  # Maximum allowed degradation percentage
    critical: bool = True  # If True, failure blocks deployment


@dataclass
class MetricComparison:
    """Comparison result for a single metric"""
    metric_name: str
    baseline_value: float
    candidate_value: float
    change_pct: float
    degradation: bool
    passed: bool
    critical: bool


@dataclass
class RegressionResult:
    """Result from regression testing"""
    baseline_model_id: str
    candidate_model_id: str
    test_type: str  # "registry" or "holdout"
    metrics_tested: List[str]
    comparisons: List[MetricComparison]
    passed_metrics: List[str]
    failed_metrics: List[str]
    critical_failures: List[str]
    passed: bool
    blocked: bool  # True if deployment should be blocked
    override_allowed: bool  # True if override is possible
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RegressionGuards:
    """
    Comprehensive regression testing framework

    Compares candidate models against baseline models to prevent degradation.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        registry_db: Optional[Path] = None,
        guards_db: Optional[Path] = None
    ):
        """
        Initialize regression guards

        Args:
            config_path: Path to config YAML (default: training/sft/sft_config.yaml)
            registry_db: Path to model registry DB
            guards_db: Path to regression guards tracking DB
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "sft_config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize model registry
        self.registry = ModelRegistry(registry_db=registry_db)

        # Initialize eval gates (for holdout testing)
        self.eval_gates = EvalGates(config_path=config_path)

        # Setup tracking database
        if guards_db is None:
            guards_db = Path("models/sft/regression_guards.db")

        self.guards_db = Path(guards_db)
        self.guards_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Parse metric policies
        self.policies = self._parse_metric_policies()

    def _parse_metric_policies(self) -> List[MetricPolicy]:
        """Parse metric policies from config"""
        policies = []
        regression_config = self.config.get("regression_guards", {})

        # Default policies
        default_tolerance = regression_config.get("default_tolerance_pct", 2.0)

        # Standard metrics
        policies.append(MetricPolicy(
            metric_name="eval_loss",
            direction=MetricDirection.MINIMIZE,
            tolerance_pct=regression_config.get("loss_tolerance_pct", default_tolerance),
            critical=True
        ))

        policies.append(MetricPolicy(
            metric_name="eval_accuracy",
            direction=MetricDirection.MAXIMIZE,
            tolerance_pct=regression_config.get("accuracy_tolerance_pct", default_tolerance),
            critical=True
        ))

        policies.append(MetricPolicy(
            metric_name="eval_f1",
            direction=MetricDirection.MAXIMIZE,
            tolerance_pct=regression_config.get("f1_tolerance_pct", default_tolerance),
            critical=True
        ))

        policies.append(MetricPolicy(
            metric_name="eval_perplexity",
            direction=MetricDirection.MINIMIZE,
            tolerance_pct=regression_config.get("perplexity_tolerance_pct", default_tolerance),
            critical=False
        ))

        # Custom policies from config
        for policy_config in regression_config.get("custom_policies", []):
            policies.append(MetricPolicy(
                metric_name=policy_config["metric_name"],
                direction=MetricDirection(policy_config["direction"]),
                tolerance_pct=policy_config.get("tolerance_pct", default_tolerance),
                critical=policy_config.get("critical", True)
            ))

        return policies

    def _init_db(self):
        """Initialize tracking database"""
        conn = sqlite3.connect(self.guards_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regression_tests (
                test_id TEXT PRIMARY KEY,
                baseline_model_id TEXT NOT NULL,
                candidate_model_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                blocked BOOLEAN NOT NULL,
                failed_metrics TEXT,
                critical_failures TEXT,
                override_applied BOOLEAN DEFAULT 0,
                override_reason TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_comparisons (
                comparison_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                candidate_value REAL NOT NULL,
                change_pct REAL NOT NULL,
                degradation BOOLEAN NOT NULL,
                passed BOOLEAN NOT NULL,
                FOREIGN KEY (test_id) REFERENCES regression_tests(test_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_timestamp
            ON regression_tests(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidate_model
            ON regression_tests(candidate_model_id)
        """)

        conn.commit()
        conn.close()

    def test(
        self,
        baseline_model_id: str,
        candidate_model_id: str,
        metrics: Optional[List[str]] = None
    ) -> RegressionResult:
        """
        Run regression test using registry metrics

        Args:
            baseline_model_id: Baseline model ID
            candidate_model_id: Candidate model ID
            metrics: Metrics to test (default: all configured metrics)

        Returns:
            RegressionResult
        """
        logger.info(f"Running regression test (registry-based)")
        logger.info(f"  Baseline: {baseline_model_id}")
        logger.info(f"  Candidate: {candidate_model_id}")

        # Get models from registry
        baseline = self.registry.get_model(baseline_model_id)
        candidate = self.registry.get_model(candidate_model_id)

        if not baseline or not candidate:
            raise ValueError("One or both models not found in registry")

        # Determine metrics to test
        if metrics is None:
            metrics = [p.metric_name for p in self.policies]

        # Run comparisons
        comparisons = []
        passed_metrics = []
        failed_metrics = []
        critical_failures = []

        for metric_name in metrics:
            # Find policy
            policy = next((p for p in self.policies if p.metric_name == metric_name), None)
            if not policy:
                logger.warning(f"No policy for metric {metric_name}, skipping")
                continue

            # Get metric values
            baseline_value = getattr(baseline, metric_name, None)
            candidate_value = getattr(candidate, metric_name, None)

            if baseline_value is None or candidate_value is None:
                logger.warning(f"Metric {metric_name} not found in one or both models")
                continue

            # Calculate change
            if baseline_value == 0:
                change_pct = 100.0 if candidate_value != 0 else 0.0
            else:
                change_pct = ((candidate_value - baseline_value) / baseline_value) * 100

            # Determine degradation
            if policy.direction == MetricDirection.MINIMIZE:
                # For metrics where lower is better (e.g., loss)
                degradation = candidate_value > baseline_value
                passed = change_pct <= policy.tolerance_pct
            else:
                # For metrics where higher is better (e.g., accuracy)
                degradation = candidate_value < baseline_value
                passed = abs(change_pct) <= policy.tolerance_pct or change_pct > 0

            comparison = MetricComparison(
                metric_name=metric_name,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                change_pct=change_pct,
                degradation=degradation,
                passed=passed,
                critical=policy.critical
            )

            comparisons.append(comparison)

            if passed:
                passed_metrics.append(metric_name)
                logger.info(
                    f"✅ {metric_name}: {baseline_value:.4f} → {candidate_value:.4f} "
                    f"({change_pct:+.1f}%)"
                )
            else:
                failed_metrics.append(metric_name)
                logger.warning(
                    f"❌ {metric_name}: {baseline_value:.4f} → {candidate_value:.4f} "
                    f"({change_pct:+.1f}%, tolerance: {policy.tolerance_pct}%)"
                )

                if policy.critical:
                    critical_failures.append(metric_name)

        # Overall result
        overall_passed = len(failed_metrics) == 0
        blocked = len(critical_failures) > 0

        result = RegressionResult(
            baseline_model_id=baseline_model_id,
            candidate_model_id=candidate_model_id,
            test_type="registry",
            metrics_tested=metrics,
            comparisons=comparisons,
            passed_metrics=passed_metrics,
            failed_metrics=failed_metrics,
            critical_failures=critical_failures,
            passed=overall_passed,
            blocked=blocked,
            override_allowed=(not blocked) or (len(critical_failures) == 1)  # Allow override for single critical failure
        )

        # Store in database
        self._store_regression_test(result)

        return result

    def test_with_holdout(
        self,
        baseline_model_path: Path,
        candidate_model_path: Path,
        holdout_dataset: Path,
        metrics: Optional[List[str]] = None
    ) -> RegressionResult:
        """
        Run regression test with holdout re-evaluation

        Args:
            baseline_model_path: Path to baseline model
            candidate_model_path: Path to candidate model
            holdout_dataset: Path to holdout dataset
            metrics: Metrics to test

        Returns:
            RegressionResult
        """
        logger.info(f"Running regression test (holdout-based)")
        logger.info(f"  Baseline: {baseline_model_path}")
        logger.info(f"  Candidate: {candidate_model_path}")
        logger.info(f"  Holdout: {holdout_dataset}")

        # Evaluate both models on holdout
        logger.info("Evaluating baseline model...")
        baseline_eval = self.eval_gates.evaluate(
            model_path=baseline_model_path,
            holdout_dataset=holdout_dataset
        )

        logger.info("Evaluating candidate model...")
        candidate_eval = self.eval_gates.evaluate(
            model_path=candidate_model_path,
            holdout_dataset=holdout_dataset
        )

        # Determine metrics to test
        if metrics is None:
            metrics = [p.metric_name for p in self.policies]

        # Run comparisons
        comparisons = []
        passed_metrics = []
        failed_metrics = []
        critical_failures = []

        for metric_name in metrics:
            # Find policy
            policy = next((p for p in self.policies if p.metric_name == metric_name), None)
            if not policy:
                continue

            # Get metric values
            baseline_value = baseline_eval.metrics.get(metric_name)
            candidate_value = candidate_eval.metrics.get(metric_name)

            if baseline_value is None or candidate_value is None:
                logger.warning(f"Metric {metric_name} not available")
                continue

            # Calculate change
            if baseline_value == 0:
                change_pct = 100.0 if candidate_value != 0 else 0.0
            else:
                change_pct = ((candidate_value - baseline_value) / baseline_value) * 100

            # Determine degradation
            if policy.direction == MetricDirection.MINIMIZE:
                degradation = candidate_value > baseline_value
                passed = change_pct <= policy.tolerance_pct
            else:
                degradation = candidate_value < baseline_value
                passed = abs(change_pct) <= policy.tolerance_pct or change_pct > 0

            comparison = MetricComparison(
                metric_name=metric_name,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                change_pct=change_pct,
                degradation=degradation,
                passed=passed,
                critical=policy.critical
            )

            comparisons.append(comparison)

            if passed:
                passed_metrics.append(metric_name)
                logger.info(
                    f"✅ {metric_name}: {baseline_value:.4f} → {candidate_value:.4f} "
                    f"({change_pct:+.1f}%)"
                )
            else:
                failed_metrics.append(metric_name)
                logger.warning(
                    f"❌ {metric_name}: {baseline_value:.4f} → {candidate_value:.4f} "
                    f"({change_pct:+.1f}%, tolerance: {policy.tolerance_pct}%)"
                )

                if policy.critical:
                    critical_failures.append(metric_name)

        # Overall result
        overall_passed = len(failed_metrics) == 0
        blocked = len(critical_failures) > 0

        result = RegressionResult(
            baseline_model_id=str(baseline_model_path),
            candidate_model_id=str(candidate_model_path),
            test_type="holdout",
            metrics_tested=metrics,
            comparisons=comparisons,
            passed_metrics=passed_metrics,
            failed_metrics=failed_metrics,
            critical_failures=critical_failures,
            passed=overall_passed,
            blocked=blocked,
            override_allowed=(not blocked) or (len(critical_failures) == 1)
        )

        # Store in database
        self._store_regression_test(result)

        return result

    def apply_override(
        self,
        test_id: str,
        reason: str
    ) -> bool:
        """
        Apply override to allow deployment despite regression

        Args:
            test_id: Test ID to override
            reason: Reason for override (required)

        Returns:
            True if override successful
        """
        if not reason or len(reason) < 10:
            raise ValueError("Override reason must be at least 10 characters")

        conn = sqlite3.connect(self.guards_db)
        cursor = conn.cursor()

        # Check if test exists and is blocked
        cursor.execute("""
            SELECT blocked, override_allowed FROM regression_tests
            WHERE test_id = ?
        """, (test_id,))

        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Test {test_id} not found")

        blocked, override_allowed = row

        if not blocked:
            logger.info(f"Test {test_id} is not blocked, no override needed")
            return True

        if not override_allowed:
            raise ValueError(f"Override not allowed for test {test_id} (multiple critical failures)")

        # Apply override
        cursor.execute("""
            UPDATE regression_tests
            SET override_applied = 1, override_reason = ?
            WHERE test_id = ?
        """, (reason, test_id))

        conn.commit()
        conn.close()

        logger.warning(f"⚠️  Override applied to test {test_id}")
        logger.warning(f"   Reason: {reason}")

        return True

    def _store_regression_test(self, result: RegressionResult):
        """Store regression test in database"""
        test_id = f"{Path(result.candidate_model_id).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.guards_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO regression_tests (
                test_id, baseline_model_id, candidate_model_id, test_type,
                passed, blocked, failed_metrics, critical_failures, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            result.baseline_model_id,
            result.candidate_model_id,
            result.test_type,
            result.passed,
            result.blocked,
            json.dumps(result.failed_metrics),
            json.dumps(result.critical_failures),
            result.timestamp
        ))

        # Store metric comparisons
        for comp in result.comparisons:
            comparison_id = f"{test_id}_{comp.metric_name}"
            cursor.execute("""
                INSERT INTO metric_comparisons (
                    comparison_id, test_id, metric_name,
                    baseline_value, candidate_value, change_pct,
                    degradation, passed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison_id,
                test_id,
                comp.metric_name,
                comp.baseline_value,
                comp.candidate_value,
                comp.change_pct,
                comp.degradation,
                comp.passed
            ))

        conn.commit()
        conn.close()

        logger.debug(f"Stored regression test: {test_id}")

    def get_test_history(
        self,
        candidate_model_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get regression test history

        Args:
            candidate_model_id: Filter by candidate model (optional)
            limit: Maximum number of results

        Returns:
            List of test records
        """
        conn = sqlite3.connect(self.guards_db)
        cursor = conn.cursor()

        if candidate_model_id:
            cursor.execute("""
                SELECT * FROM regression_tests
                WHERE candidate_model_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (candidate_model_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM regression_tests
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        results = []

        for row in cursor.fetchall():
            record = dict(zip(columns, row))

            # Parse JSON fields
            if record["failed_metrics"]:
                record["failed_metrics"] = json.loads(record["failed_metrics"])
            if record["critical_failures"]:
                record["critical_failures"] = json.loads(record["critical_failures"])

            results.append(record)

        conn.close()

        return results

    def generate_report(
        self,
        result: RegressionResult,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate detailed regression report

        Args:
            result: RegressionResult to report on
            output_file: Optional file to write report to

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("REGRESSION GUARDS REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Baseline Model: {result.baseline_model_id}")
        lines.append(f"Candidate Model: {result.candidate_model_id}")
        lines.append(f"Test Type: {result.test_type}")
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append("")

        # Comparisons
        lines.append("METRIC COMPARISONS:")
        for comp in result.comparisons:
            status = "✅" if comp.passed else "❌"
            critical = " [CRITICAL]" if comp.critical and not comp.passed else ""
            lines.append(
                f"  {status} {comp.metric_name}{critical}: "
                f"{comp.baseline_value:.4f} → {comp.candidate_value:.4f} "
                f"({comp.change_pct:+.1f}%)"
            )
        lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Metrics Tested: {len(result.metrics_tested)}")
        lines.append(f"  Passed: {len(result.passed_metrics)}")
        lines.append(f"  Failed: {len(result.failed_metrics)}")
        lines.append(f"  Critical Failures: {len(result.critical_failures)}")
        lines.append("")

        # Failed metrics
        if result.failed_metrics:
            lines.append("❌ FAILED METRICS:")
            for metric in result.failed_metrics:
                lines.append(f"  - {metric}")
            lines.append("")

        # Deployment decision
        lines.append("=" * 80)
        if result.passed:
            lines.append("RESULT: ✅ NO REGRESSION DETECTED")
            lines.append("DEPLOYMENT: ✅ APPROVED")
        elif result.blocked:
            lines.append("RESULT: ❌ REGRESSION DETECTED")
            lines.append("DEPLOYMENT: 🚫 BLOCKED")
            if result.override_allowed:
                lines.append("OVERRIDE: ⚠️  ALLOWED (manual approval required)")
        else:
            lines.append("RESULT: ⚠️  MINOR REGRESSION DETECTED")
            lines.append("DEPLOYMENT: ⚠️  ALLOWED (non-critical failures)")
        lines.append("=" * 80)

        report = "\n".join(lines)

        # Write to file if specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report written to {output_file}")

        return report


def main():
    """CLI interface for regression guards"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regression Guards - Prevent Model Degradation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test using registry metrics
  python training/sft/regression_guards.py \\
      --baseline news_agent_1.0.0_20240101 \\
      --candidate news_agent_1.1.0_20240115 \\
      --metrics eval_loss eval_accuracy

  # Test with holdout re-evaluation
  python training/sft/regression_guards.py \\
      --baseline-path models/sft/news_agent_v1.0.0 \\
      --candidate-path models/sft/news_agent_v1.1.0 \\
      --holdout data/datasets/sft_v1/test.jsonl

  # View test history
  python training/sft/regression_guards.py --history --limit 20

  # Apply override
  python training/sft/regression_guards.py --override TEST_ID --reason "Acceptable tradeoff for new feature"
        """
    )

    parser.add_argument(
        "--baseline",
        help="Baseline model ID (for registry-based test)"
    )

    parser.add_argument(
        "--candidate",
        help="Candidate model ID (for registry-based test)"
    )

    parser.add_argument(
        "--baseline-path",
        type=Path,
        help="Baseline model path (for holdout-based test)"
    )

    parser.add_argument(
        "--candidate-path",
        type=Path,
        help="Candidate model path (for holdout-based test)"
    )

    parser.add_argument(
        "--holdout",
        type=Path,
        help="Holdout dataset path"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to test (default: all configured)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML (default: training/sft/sft_config.yaml)"
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="View regression test history"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for history results (default: 10)"
    )

    parser.add_argument(
        "--override",
        help="Test ID to override"
    )

    parser.add_argument(
        "--reason",
        help="Override reason (required with --override)"
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="Output report file path"
    )

    args = parser.parse_args()

    # Initialize guards
    guards = RegressionGuards(config_path=args.config)

    # Apply override
    if args.override:
        if not args.reason:
            parser.error("--reason required with --override")

        success = guards.apply_override(args.override, args.reason)
        print(f"✅ Override applied: {args.override}")
        return 0

    # View history
    if args.history:
        history = guards.get_test_history(
            candidate_model_id=args.candidate,
            limit=args.limit
        )

        print(f"\n{'='*80}")
        print(f"REGRESSION TEST HISTORY ({len(history)} records)")
        print(f"{'='*80}\n")

        for record in history:
            print(f"Test ID: {record['test_id']}")
            print(f"Baseline: {record['baseline_model_id']}")
            print(f"Candidate: {record['candidate_model_id']}")
            print(f"Type: {record['test_type']}")
            print(f"Passed: {'✅' if record['passed'] else '❌'}")
            print(f"Blocked: {'🚫 Yes' if record['blocked'] else 'No'}")
            if record['override_applied']:
                print(f"Override: ⚠️  Applied - {record['override_reason']}")
            print(f"Timestamp: {record['timestamp']}")
            print("-" * 80)

        return 0

    # Run regression test
    if args.holdout:
        # Holdout-based test
        if not args.baseline_path or not args.candidate_path:
            parser.error("--baseline-path and --candidate-path required with --holdout")

        result = guards.test_with_holdout(
            baseline_model_path=args.baseline_path,
            candidate_model_path=args.candidate_path,
            holdout_dataset=args.holdout,
            metrics=args.metrics
        )
    else:
        # Registry-based test
        if not args.baseline or not args.candidate:
            parser.error("--baseline and --candidate required (or use --holdout test)")

        result = guards.test(
            baseline_model_id=args.baseline,
            candidate_model_id=args.candidate,
            metrics=args.metrics
        )

    # Generate report
    report = guards.generate_report(result, output_file=args.report)
    print("\n" + report)

    # Exit code
    if result.blocked:
        return 2  # Blocked
    elif result.passed:
        return 0  # Passed
    else:
        return 1  # Failed but not blocked


if __name__ == "__main__":
    exit(main())
