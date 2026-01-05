#!/usr/bin/env python3
"""
Dataset Synthesis CLI - End-to-end pipeline for creating training datasets from backtests

Purpose:
    Orchestrates the complete workflow:
    1. Load backtest results → Experience Store
    2. Apply judge filtering (optional)
    3. Synthesize dataset with chosen strategy
    4. Save to disk with versioning

Usage:
    # Quick start with preset
    python scripts/synthesize_dataset.py --preset sft_v1

    # Custom synthesis
    python scripts/synthesize_dataset.py \
        --backtest-results results/backtest_20240115.json \
        --strategy judge_approved \
        --format chat \
        --min-judge-score 6.0 \
        --output data/datasets/my_dataset_v1

    # From existing experience store
    python scripts/synthesize_dataset.py \
        --experience-store data/experiences \
        --strategy positive_only \
        --output data/datasets/positive_examples
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.data_synthesis.experience_store import ExperienceStore, ExperienceStoreConfig, Experience
from training.data_synthesis.dataset_synthesizer import DatasetSynthesizer
from training.data_synthesis.judge_filter import JudgeFilter


class DatasetSynthesisPipeline:
    """
    End-to-end pipeline for dataset synthesis

    Workflow:
    1. Load backtest results (if provided)
    2. Populate experience store
    3. Apply judge filtering (if enabled)
    4. Synthesize dataset
    5. Save to output directory
    """

    def __init__(
        self,
        experience_store_dir: Path = Path("data/experiences"),
        config_path: Optional[Path] = None
    ):
        self.experience_store_dir = experience_store_dir
        self.config = self._load_config(config_path)

        # Initialize experience store
        store_config = ExperienceStoreConfig(
            storage_dir=experience_store_dir,
            storage_format=self.config.get("storage", {}).get("storage_format", "jsonl")
        )
        self.store = ExperienceStore(store_config)

        logger.info(f"DatasetSynthesisPipeline initialized")
        logger.info(f"Experience store: {experience_store_dir}")

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load synthesis configuration"""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            # Load default config
            default_config = project_root / "training" / "data_synthesis" / "synthesis_config.yaml"
            if default_config.exists():
                with open(default_config) as f:
                    return yaml.safe_load(f)
        return {}

    def load_backtest_results(
        self,
        backtest_results_path: Path,
        apply_judge_filter: bool = False,
        min_judge_score: Optional[float] = None
    ) -> int:
        """
        Load backtest results into experience store

        Args:
            backtest_results_path: Path to backtest results JSON
            apply_judge_filter: Apply judge filtering during load
            min_judge_score: Minimum judge score if filtering

        Returns:
            Number of experiences added
        """
        logger.info(f"Loading backtest results: {backtest_results_path}")

        if not backtest_results_path.exists():
            raise FileNotFoundError(f"Backtest results not found: {backtest_results_path}")

        # Load results
        with open(backtest_results_path) as f:
            results = json.load(f)

        # Extract experiences from backtest
        experiences = self._extract_experiences_from_backtest(results)

        logger.info(f"Extracted {len(experiences)} experiences from backtest")

        # Add to store
        experience_ids = []
        for exp_data in experiences:
            exp_id = self.store.add_experience(
                signal=exp_data["signal"],
                action=exp_data["action"],
                outcome=exp_data["outcome"],
                reward=exp_data["reward"],
                metadata=exp_data.get("metadata", {})
            )
            experience_ids.append(exp_id)

        logger.info(f"Added {len(experience_ids)} experiences to store")

        # Apply judge filter if requested
        if apply_judge_filter:
            logger.info("Applying judge filter to new experiences...")
            judge_filter = JudgeFilter(experience_store=self.store)
            results = judge_filter.filter_experiences(
                min_score=min_judge_score or 6.0,
                skip_already_judged=True
            )
            logger.info(f"Judge filtering: {results.num_passed}/{results.num_total} passed")

        return len(experience_ids)

    def _extract_experiences_from_backtest(self, backtest_results: Dict) -> List[Dict]:
        """
        Extract experiences from backtest results

        Expected format:
        {
          "signals": [...],
          "trades": [...],
          "metrics": {...}
        }
        """
        experiences = []

        # Get signals and trades
        signals = backtest_results.get("signals", [])
        trades = backtest_results.get("trades", [])

        # Match signals to trades
        for signal in signals:
            # Find corresponding trade
            trade = None
            for t in trades:
                if (t.get("symbol") == signal.get("metadata", {}).get("symbol") and
                    t.get("entry_date") == signal.get("metadata", {}).get("timestamp", "").split("T")[0]):
                    trade = t
                    break

            # If no trade found, create default outcome
            if trade is None:
                outcome = {
                    "pnl": 0.0,
                    "return_pct": 0.0,
                    "duration_days": 0,
                    "exit_reason": "not_executed"
                }
                reward = 0.0
            else:
                outcome = {
                    "pnl": trade.get("pnl", 0.0),
                    "return_pct": trade.get("return_pct", 0.0),
                    "duration_days": trade.get("duration_days", 0),
                    "exit_reason": trade.get("exit_reason", "unknown")
                }
                # Normalize reward to [-1, 1]
                reward = min(max(trade.get("return_pct", 0.0) * 2, -1.0), 1.0)

            # Extract action from signal
            action = {
                "decision": signal.get("signal", "hold"),
                "position_size": signal.get("sizing", {}).get("position_size", 0.0),
                "entry_target": signal.get("metadata", {}).get("quote", {}).get("price"),
                "stop_loss": signal.get("risk", {}).get("stop_loss"),
                "take_profit": signal.get("risk", {}).get("take_profit")
            }

            experiences.append({
                "signal": signal,
                "action": action,
                "outcome": outcome,
                "reward": reward,
                "metadata": {
                    "symbol": signal.get("metadata", {}).get("symbol", "UNKNOWN"),
                    "backtest_id": backtest_results.get("backtest_id"),
                    "timestamp": signal.get("metadata", {}).get("timestamp")
                }
            })

        return experiences

    def synthesize(
        self,
        strategy: str = "judge_approved",
        output_format: str = "chat",
        min_reward: float = 0.0,
        min_judge_score: Optional[float] = None,
        symbol: Optional[str] = None,
        version: str = "1.0.0",
        output_dir: Path = Path("data/datasets/output")
    ) -> Dict[str, Any]:
        """
        Synthesize dataset from experience store

        Args:
            strategy: Synthesis strategy (judge_approved, positive_only, contrastive, full_spectrum)
            output_format: Dataset format (chat, prompt_completion, instruction)
            min_reward: Minimum reward threshold
            min_judge_score: Minimum judge score
            symbol: Filter by symbol
            version: Dataset version
            output_dir: Output directory

        Returns:
            Synthesis results with statistics
        """
        logger.info("Starting dataset synthesis...")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Format: {output_format}")
        logger.info(f"  Min reward: {min_reward}")
        logger.info(f"  Min judge score: {min_judge_score}")

        # Get split ratios from config
        splits_config = self.config.get("splits", {})
        train_split = splits_config.get("train", 0.8)
        val_split = splits_config.get("val", 0.1)
        test_split = splits_config.get("test", 0.1)
        random_seed = splits_config.get("random_seed", 42)

        # Initialize synthesizer
        synthesizer = DatasetSynthesizer(
            experience_store=self.store,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            random_seed=random_seed
        )

        # Synthesize
        dataset = synthesizer.synthesize(
            strategy=strategy,
            output_format=output_format,
            min_reward=min_reward,
            min_judge_score=min_judge_score,
            symbol=symbol,
            version=version
        )

        # Save
        synthesizer.save_dataset(dataset, output_dir)

        # Return results
        return {
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "strategy": dataset.strategy,
            "format": dataset.format,
            "num_examples": dataset.num_examples,
            "num_train": dataset.num_train,
            "num_val": dataset.num_val,
            "num_test": dataset.num_test,
            "avg_reward": dataset.avg_reward,
            "approval_rate": dataset.approval_rate,
            "output_dir": str(output_dir)
        }

    def run_preset(
        self,
        preset_name: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run preset configuration

        Presets:
        - sft_v1: High-quality SFT dataset (judge_approved, chat format)
        - preference_v1: Preference learning (contrastive, instruction format)
        - rl_v1: RL training (full_spectrum, chat format)
        - eval_benchmark: Gold-standard evaluation (high_confidence, chat format)

        Args:
            preset_name: Preset configuration name
            output_dir: Output directory (optional, uses preset default)

        Returns:
            Synthesis results
        """
        presets = self.config.get("presets", {})

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        preset = presets[preset_name]
        logger.info(f"Running preset: {preset_name}")

        # Use preset output dir if not specified
        if output_dir is None:
            output_dir = Path(self.config.get("storage", {}).get("dataset_output_dir", "data/datasets"))
            output_dir = output_dir / f"{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self.synthesize(
            strategy=preset.get("strategy"),
            output_format=preset.get("format"),
            min_reward=preset.get("min_reward", 0.0),
            min_judge_score=preset.get("min_judge_score"),
            version=preset.get("version", "1.0.0"),
            output_dir=output_dir
        )

    def close(self):
        """Close connections"""
        self.store.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset Synthesis CLI - Create training datasets from backtest experiences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configuration
  python scripts/synthesize_dataset.py --preset sft_v1

  # Load backtest and synthesize
  python scripts/synthesize_dataset.py \\
      --backtest-results results/backtest_20240115.json \\
      --strategy judge_approved \\
      --output data/datasets/sft_dataset

  # From existing experience store
  python scripts/synthesize_dataset.py \\
      --experience-store data/experiences \\
      --strategy positive_only \\
      --format chat \\
      --min-reward 0.5

Presets:
  - sft_v1: High-quality SFT dataset (judge-approved, chat format)
  - preference_v1: Preference learning dataset (contrastive pairs)
  - rl_v1: RL training dataset (full spectrum)
  - eval_benchmark: Gold-standard evaluation benchmark
        """
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--backtest-results", type=Path, help="Path to backtest results JSON")
    input_group.add_argument("--experience-store", type=Path, help="Path to existing experience store")
    input_group.add_argument("--preset", type=str, choices=["sft_v1", "preference_v1", "rl_v1", "eval_benchmark"], help="Use preset configuration")

    # Synthesis parameters
    parser.add_argument("--strategy", choices=["judge_approved", "positive_only", "contrastive", "full_spectrum", "high_confidence"], default="judge_approved", help="Synthesis strategy")
    parser.add_argument("--format", choices=["chat", "prompt_completion", "instruction"], default="chat", help="Dataset format")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Minimum reward threshold")
    parser.add_argument("--min-judge-score", type=float, help="Minimum judge score (0-10)")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--version", default="1.0.0", help="Dataset version")

    # Judge filtering
    parser.add_argument("--apply-judge-filter", action="store_true", help="Apply judge filtering to backtest results")

    # Output
    parser.add_argument("--output", type=Path, help="Output directory for dataset")

    # Configuration
    parser.add_argument("--config", type=Path, help="Path to synthesis config YAML")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    # Initialize pipeline
    experience_store_dir = args.experience_store or Path("data/experiences")
    pipeline = DatasetSynthesisPipeline(
        experience_store_dir=experience_store_dir,
        config_path=args.config
    )

    try:
        # Load backtest results if provided
        if args.backtest_results:
            pipeline.load_backtest_results(
                backtest_results_path=args.backtest_results,
                apply_judge_filter=args.apply_judge_filter,
                min_judge_score=args.min_judge_score
            )

        # Run synthesis
        if args.preset:
            # Use preset
            results = pipeline.run_preset(
                preset_name=args.preset,
                output_dir=args.output
            )
        else:
            # Custom synthesis
            output_dir = args.output or Path(f"data/datasets/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            results = pipeline.synthesize(
                strategy=args.strategy,
                output_format=args.format,
                min_reward=args.min_reward,
                min_judge_score=args.min_judge_score,
                symbol=args.symbol,
                version=args.version,
                output_dir=output_dir
            )

        # Print results
        print("\n" + "=" * 60)
        print("✅ DATASET SYNTHESIS COMPLETE")
        print("=" * 60)
        print(f"\nDataset ID: {results['dataset_id']}")
        print(f"Version: {results['version']}")
        print(f"Strategy: {results['strategy']}")
        print(f"Format: {results['format']}")
        print(f"\nExamples: {results['num_examples']}")
        print(f"  - Train: {results['num_train']}")
        print(f"  - Val: {results['num_val']}")
        print(f"  - Test: {results['num_test']}")
        print(f"\nAverage Reward: {results['avg_reward']:.3f}")
        print(f"Approval Rate: {results['approval_rate']:.1%}")
        print(f"\nOutput: {results['output_dir']}")
        print("=" * 60)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
