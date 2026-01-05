#!/usr/bin/env python3
"""
Eval Gates - Standalone Evaluation System for Model Quality Control

Purpose:
    Automated quality gates for trained models using holdout datasets.
    Checks models against configurable thresholds and tracks performance drift.

Features:
    - Holdout dataset evaluation
    - Multi-metric quality gates
    - Performance drift detection
    - Detailed pass/fail reporting
    - Historical tracking

Usage:
    # Basic evaluation
    gates = EvalGates(config_path="training/sft/sft_config.yaml")
    result = gates.evaluate(
        model_path="models/sft/news_agent_v1.0.0",
        holdout_dataset="data/datasets/sft_v1/test.jsonl"
    )

    # With drift detection
    result = gates.evaluate_with_drift_detection(
        model_path="models/sft/news_agent_v2.0.0",
        holdout_dataset="data/datasets/sft_v1/test.jsonl",
        baseline_metrics={"eval_loss": 0.45, "eval_accuracy": 0.85}
    )

    # Check gates
    if result.passed:
        print("✅ Model passed all gates")
    else:
        print(f"❌ Model failed: {result.failed_gates}")
"""

import json
import yaml
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

# Optional dependencies for model evaluation
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("Torch/Transformers not available - model evaluation disabled")


@dataclass
class GateConfig:
    """Configuration for a single evaluation gate"""
    metric_name: str
    threshold: float
    comparison: str  # "min" (>=) or "max" (<=)
    required: bool = True


@dataclass
class EvalResult:
    """Result from evaluation gates"""
    model_path: str
    holdout_dataset: str
    metrics: Dict[str, float]
    gates_checked: List[str]
    passed_gates: List[str]
    failed_gates: List[str]
    passed: bool
    drift_detected: bool = False
    drift_metrics: Optional[Dict[str, float]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DriftResult:
    """Result from drift detection"""
    metric_name: str
    baseline_value: float
    current_value: float
    drift_pct: float
    threshold_pct: float
    exceeded: bool


class EvalGates:
    """
    Standalone evaluation gates system

    Evaluates trained models against holdout datasets and quality thresholds.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        gates_db: Optional[Path] = None
    ):
        """
        Initialize evaluation gates

        Args:
            config_path: Path to SFT config YAML (default: training/sft/sft_config.yaml)
            gates_db: Path to gates tracking DB (default: models/sft/eval_gates.db)
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "sft_config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Parse gate configurations
        self.gates = self._parse_gate_configs()

        # Setup tracking database
        if gates_db is None:
            gates_db = Path("models/sft/eval_gates.db")

        self.gates_db = Path(gates_db)
        self.gates_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _parse_gate_configs(self) -> List[GateConfig]:
        """Parse gate configurations from config"""
        gates = []
        gates_config = self.config.get("eval_gates", {})

        # Standard gates
        if "min_eval_loss" in gates_config:
            gates.append(GateConfig(
                metric_name="eval_loss",
                threshold=gates_config["min_eval_loss"],
                comparison="max",
                required=True
            ))

        if "min_eval_accuracy" in gates_config:
            gates.append(GateConfig(
                metric_name="eval_accuracy",
                threshold=gates_config["min_eval_accuracy"],
                comparison="min",
                required=True
            ))

        if "min_eval_f1" in gates_config:
            gates.append(GateConfig(
                metric_name="eval_f1",
                threshold=gates_config["min_eval_f1"],
                comparison="min",
                required=True
            ))

        if "min_eval_perplexity" in gates_config:
            gates.append(GateConfig(
                metric_name="eval_perplexity",
                threshold=gates_config["min_eval_perplexity"],
                comparison="max",
                required=False
            ))

        return gates

    def _init_db(self):
        """Initialize tracking database"""
        conn = sqlite3.connect(self.gates_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_history (
                eval_id TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                holdout_dataset TEXT NOT NULL,
                eval_loss REAL,
                eval_accuracy REAL,
                eval_f1 REAL,
                eval_perplexity REAL,
                passed BOOLEAN NOT NULL,
                failed_gates TEXT,
                drift_detected BOOLEAN,
                drift_metrics TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_path
            ON eval_history(model_path)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON eval_history(timestamp DESC)
        """)

        conn.commit()
        conn.close()

    def evaluate(
        self,
        model_path: Path,
        holdout_dataset: Path,
        agent_name: Optional[str] = None
    ) -> EvalResult:
        """
        Evaluate model on holdout dataset

        Args:
            model_path: Path to trained model
            holdout_dataset: Path to holdout dataset (JSONL)
            agent_name: Agent name (for loading base model config)

        Returns:
            EvalResult with pass/fail and metrics
        """
        model_path = Path(model_path)
        holdout_dataset = Path(holdout_dataset)

        logger.info(f"Evaluating model: {model_path}")
        logger.info(f"Holdout dataset: {holdout_dataset}")

        # Load model metadata to get agent name
        if agent_name is None:
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    agent_name = metadata.get("agent_name")

        # Run evaluation
        metrics = self._evaluate_model(model_path, holdout_dataset)

        # Check gates
        passed_gates = []
        failed_gates = []

        for gate in self.gates:
            metric_value = metrics.get(gate.metric_name)

            if metric_value is None:
                if gate.required:
                    logger.warning(f"Required metric {gate.metric_name} not found")
                    failed_gates.append(f"{gate.metric_name} (missing)")
                continue

            # Check threshold
            if gate.comparison == "min":
                passed = metric_value >= gate.threshold
                gate_str = f"{gate.metric_name} >= {gate.threshold}"
            else:  # max
                passed = metric_value <= gate.threshold
                gate_str = f"{gate.metric_name} <= {gate.threshold}"

            if passed:
                passed_gates.append(gate_str)
                logger.info(f"✅ Gate passed: {gate_str} (actual: {metric_value:.4f})")
            else:
                failed_gates.append(gate_str)
                logger.warning(f"❌ Gate failed: {gate_str} (actual: {metric_value:.4f})")

        # Overall result
        overall_passed = len(failed_gates) == 0

        result = EvalResult(
            model_path=str(model_path),
            holdout_dataset=str(holdout_dataset),
            metrics=metrics,
            gates_checked=[f"{g.metric_name} {g.comparison} {g.threshold}" for g in self.gates],
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            passed=overall_passed
        )

        # Store in database
        self._store_eval_result(result)

        return result

    def _evaluate_model(
        self,
        model_path: Path,
        holdout_dataset: Path
    ) -> Dict[str, float]:
        """
        Run model evaluation on holdout dataset

        Args:
            model_path: Path to model
            holdout_dataset: Path to JSONL dataset

        Returns:
            Dictionary of metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Torch/Transformers not available. Install with: "
                "pip install torch transformers datasets peft"
            )

        logger.info("Loading model and tokenizer...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(str(model_path))

        # Load dataset
        logger.info("Loading holdout dataset...")
        dataset = load_dataset("json", data_files=str(holdout_dataset), split="train")

        # Tokenize dataset
        def tokenize_function(examples):
            # Format as chat (assumes "messages" format)
            texts = []
            for messages in examples["messages"]:
                # Simple chat formatting
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"{role}: {content}\n"
                texts.append(text)

            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=512
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Setup trainer for evaluation
        training_args = TrainingArguments(
            output_dir="/tmp/eval_gates_tmp",
            per_device_eval_batch_size=1,
            dataloader_drop_last=False,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )

        # Run evaluation
        logger.info("Running evaluation...")
        eval_results = trainer.evaluate()

        # Extract metrics
        metrics = {
            "eval_loss": eval_results.get("eval_loss", 0.0),
            "eval_runtime": eval_results.get("eval_runtime", 0.0),
            "eval_samples_per_second": eval_results.get("eval_samples_per_second", 0.0)
        }

        # Calculate perplexity
        if "eval_loss" in metrics:
            metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"])).item()

        logger.info(f"Evaluation complete: {metrics}")

        return metrics

    def evaluate_with_drift_detection(
        self,
        model_path: Path,
        holdout_dataset: Path,
        baseline_metrics: Dict[str, float],
        drift_threshold_pct: float = 5.0,
        agent_name: Optional[str] = None
    ) -> EvalResult:
        """
        Evaluate model with drift detection

        Args:
            model_path: Path to trained model
            holdout_dataset: Path to holdout dataset
            baseline_metrics: Baseline metrics to compare against
            drift_threshold_pct: Drift threshold percentage (default: 5%)
            agent_name: Agent name

        Returns:
            EvalResult with drift information
        """
        # Run standard evaluation
        result = self.evaluate(model_path, holdout_dataset, agent_name)

        # Check for drift
        drift_results = []
        drift_detected = False

        for metric_name, baseline_value in baseline_metrics.items():
            current_value = result.metrics.get(metric_name)

            if current_value is None:
                continue

            # Calculate drift percentage
            if baseline_value == 0:
                drift_pct = 100.0 if current_value != 0 else 0.0
            else:
                drift_pct = abs((current_value - baseline_value) / baseline_value) * 100

            exceeded = drift_pct > drift_threshold_pct

            drift_result = DriftResult(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                drift_pct=drift_pct,
                threshold_pct=drift_threshold_pct,
                exceeded=exceeded
            )

            drift_results.append(drift_result)

            if exceeded:
                drift_detected = True
                logger.warning(
                    f"⚠️  Drift detected in {metric_name}: "
                    f"{baseline_value:.4f} → {current_value:.4f} "
                    f"({drift_pct:.1f}% > {drift_threshold_pct}%)"
                )
            else:
                logger.info(
                    f"✅ {metric_name} within drift threshold: "
                    f"{baseline_value:.4f} → {current_value:.4f} "
                    f"({drift_pct:.1f}%)"
                )

        # Update result with drift information
        result.drift_detected = drift_detected
        result.drift_metrics = {
            dr.metric_name: {
                "baseline": dr.baseline_value,
                "current": dr.current_value,
                "drift_pct": dr.drift_pct,
                "exceeded": dr.exceeded
            }
            for dr in drift_results
        }

        # Update database
        self._store_eval_result(result)

        return result

    def _store_eval_result(self, result: EvalResult):
        """Store evaluation result in database"""
        eval_id = f"{Path(result.model_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.gates_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO eval_history (
                eval_id, model_path, holdout_dataset,
                eval_loss, eval_accuracy, eval_f1, eval_perplexity,
                passed, failed_gates, drift_detected, drift_metrics, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_id,
            result.model_path,
            result.holdout_dataset,
            result.metrics.get("eval_loss"),
            result.metrics.get("eval_accuracy"),
            result.metrics.get("eval_f1"),
            result.metrics.get("eval_perplexity"),
            result.passed,
            json.dumps(result.failed_gates),
            result.drift_detected,
            json.dumps(result.drift_metrics) if result.drift_metrics else None,
            result.timestamp
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Stored evaluation result: {eval_id}")

    def get_eval_history(
        self,
        model_path: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get evaluation history

        Args:
            model_path: Filter by model path (optional)
            limit: Maximum number of results

        Returns:
            List of evaluation records
        """
        conn = sqlite3.connect(self.gates_db)
        cursor = conn.cursor()

        if model_path:
            cursor.execute("""
                SELECT * FROM eval_history
                WHERE model_path = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (model_path, limit))
        else:
            cursor.execute("""
                SELECT * FROM eval_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        results = []

        for row in cursor.fetchall():
            record = dict(zip(columns, row))

            # Parse JSON fields
            if record["failed_gates"]:
                record["failed_gates"] = json.loads(record["failed_gates"])
            if record["drift_metrics"]:
                record["drift_metrics"] = json.loads(record["drift_metrics"])

            results.append(record)

        conn.close()

        return results

    def generate_report(
        self,
        result: EvalResult,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate detailed evaluation report

        Args:
            result: EvalResult to report on
            output_file: Optional file to write report to

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EVALUATION GATES REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Model: {result.model_path}")
        lines.append(f"Holdout Dataset: {result.holdout_dataset}")
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append("")

        # Metrics
        lines.append("METRICS:")
        for metric, value in result.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        lines.append("")

        # Gates
        lines.append("GATES:")
        lines.append(f"  Checked: {len(result.gates_checked)}")
        lines.append(f"  Passed: {len(result.passed_gates)}")
        lines.append(f"  Failed: {len(result.failed_gates)}")
        lines.append("")

        if result.passed_gates:
            lines.append("✅ PASSED GATES:")
            for gate in result.passed_gates:
                lines.append(f"  - {gate}")
            lines.append("")

        if result.failed_gates:
            lines.append("❌ FAILED GATES:")
            for gate in result.failed_gates:
                lines.append(f"  - {gate}")
            lines.append("")

        # Drift detection
        if result.drift_detected:
            lines.append("⚠️  DRIFT DETECTED:")
            for metric, drift_info in result.drift_metrics.items():
                if drift_info["exceeded"]:
                    lines.append(f"  {metric}:")
                    lines.append(f"    Baseline: {drift_info['baseline']:.4f}")
                    lines.append(f"    Current: {drift_info['current']:.4f}")
                    lines.append(f"    Drift: {drift_info['drift_pct']:.1f}%")
            lines.append("")

        # Overall result
        lines.append("=" * 80)
        if result.passed:
            lines.append("RESULT: ✅ PASSED")
        else:
            lines.append("RESULT: ❌ FAILED")
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
    """CLI interface for eval gates"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Eval Gates - Model Quality Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model on holdout dataset
  python training/sft/eval_gates.py \\
      --model models/sft/news_agent_v1.0.0 \\
      --dataset data/datasets/sft_v1/test.jsonl

  # With drift detection
  python training/sft/eval_gates.py \\
      --model models/sft/news_agent_v2.0.0 \\
      --dataset data/datasets/sft_v1/test.jsonl \\
      --baseline-metrics '{"eval_loss": 0.45, "eval_accuracy": 0.85}' \\
      --drift-threshold 5.0

  # View evaluation history
  python training/sft/eval_gates.py --history --limit 20
        """
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained model"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to holdout dataset (JSONL)"
    )

    parser.add_argument(
        "--agent",
        help="Agent name (optional, will be inferred from metadata)"
    )

    parser.add_argument(
        "--baseline-metrics",
        type=str,
        help="Baseline metrics for drift detection (JSON string)"
    )

    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=5.0,
        help="Drift threshold percentage (default: 5.0)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML (default: training/sft/sft_config.yaml)"
    )

    parser.add_argument(
        "--gates-db",
        type=Path,
        help="Path to gates tracking DB (default: models/sft/eval_gates.db)"
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="View evaluation history"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for history results (default: 10)"
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="Output report file path"
    )

    args = parser.parse_args()

    # Initialize gates
    gates = EvalGates(
        config_path=args.config,
        gates_db=args.gates_db
    )

    # View history
    if args.history:
        history = gates.get_eval_history(
            model_path=str(args.model) if args.model else None,
            limit=args.limit
        )

        print(f"\n{'='*80}")
        print(f"EVALUATION HISTORY ({len(history)} records)")
        print(f"{'='*80}\n")

        for record in history:
            print(f"Model: {record['model_path']}")
            print(f"Dataset: {record['holdout_dataset']}")
            print(f"Loss: {record['eval_loss']:.4f}" if record['eval_loss'] else "Loss: N/A")
            print(f"Passed: {'✅' if record['passed'] else '❌'}")
            print(f"Drift: {'⚠️  Yes' if record['drift_detected'] else 'No'}")
            print(f"Timestamp: {record['timestamp']}")
            print("-" * 80)

        return 0

    # Validate arguments
    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required (unless using --history)")

    # Run evaluation
    if args.baseline_metrics:
        baseline_metrics = json.loads(args.baseline_metrics)
        result = gates.evaluate_with_drift_detection(
            model_path=args.model,
            holdout_dataset=args.dataset,
            baseline_metrics=baseline_metrics,
            drift_threshold_pct=args.drift_threshold,
            agent_name=args.agent
        )
    else:
        result = gates.evaluate(
            model_path=args.model,
            holdout_dataset=args.dataset,
            agent_name=args.agent
        )

    # Generate report
    report = gates.generate_report(result, output_file=args.report)
    print("\n" + report)

    # Exit code
    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
