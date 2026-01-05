#!/usr/bin/env python3
"""
SFT Training Script - Train junior agents (News, Technical, Fundamental) with LoRA/QLoRA

Purpose:
    Unified training script for all three junior agents using supervised fine-tuning.
    Supports LoRA and QLoRA with configurable presets.

Usage:
    # Train news agent with default config
    python scripts/train_agent_sft.py --agent news_agent \
        --dataset data/datasets/sft_v1 \
        --output models/sft/news_agent_v1.0.0

    # Train with quick test preset
    python scripts/train_agent_sft.py --agent technical_agent \
        --dataset data/datasets/sft_v1 \
        --output models/sft/technical_agent_v1.0.0 \
        --preset quick_test

    # Train all agents
    python scripts/train_agent_sft.py --agent all \
        --dataset data/datasets/sft_v1 \
        --output-dir models/sft
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.sft.lora_trainer import LoRATrainer, TrainingResult


def train_single_agent(
    agent_name: str,
    dataset_dir: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    preset: Optional[str] = None,
    version: str = "1.0.0"
) -> TrainingResult:
    """
    Train a single agent

    Args:
        agent_name: Agent to train (news_agent, technical_agent, fundamental_agent)
        dataset_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        output_dir: Output directory for model
        config_path: Path to config YAML
        preset: Training preset (quick_test, production, high_quality)
        version: Model version

    Returns:
        TrainingResult with metrics
    """
    logger.info(f"Starting SFT training for {agent_name}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Preset: {preset or 'default'}")

    # Initialize trainer
    trainer = LoRATrainer(
        agent_name=agent_name,
        config_path=config_path
    )

    # Setup model
    trainer.setup(preset=preset)

    # Prepare datasets
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    logger.info("Loading datasets...")
    train_dataset = trainer.prepare_dataset(train_file, split="train")

    eval_dataset = None
    if val_file.exists():
        eval_dataset = trainer.prepare_dataset(val_file, split="val")
    else:
        logger.warning("No validation data found, skipping evaluation")

    # Train
    result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir
    )

    # Save model
    trainer.save_model(output_dir, version=version)

    # Save training results
    results_file = output_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "agent_name": result.agent_name,
            "version": version,
            "base_model": result.base_model,
            "final_train_loss": result.final_train_loss,
            "final_eval_loss": result.final_eval_loss,
            "training_time_seconds": result.training_time_seconds,
            "num_train_examples": result.num_train_examples,
            "num_eval_examples": result.num_eval_examples,
            "passed_eval_gates": result.passed_eval_gates,
            "timestamp": result.timestamp
        }, f, indent=2)

    logger.info(f"Training results saved: {results_file}")

    return result


def train_all_agents(
    dataset_dir: Path,
    output_base_dir: Path,
    config_path: Optional[Path] = None,
    preset: Optional[str] = None,
    version: str = "1.0.0"
) -> Dict[str, TrainingResult]:
    """
    Train all three agents sequentially

    Args:
        dataset_dir: Base directory containing agent-specific datasets
        output_base_dir: Base output directory
        config_path: Path to config YAML
        preset: Training preset
        version: Model version

    Returns:
        Dict mapping agent_name to TrainingResult
    """
    agents = ["news_agent", "technical_agent", "fundamental_agent"]
    results = {}

    logger.info(f"Training all {len(agents)} agents sequentially...")

    for agent_name in agents:
        # Agent-specific dataset directory (optional)
        agent_dataset_dir = dataset_dir / agent_name
        if not agent_dataset_dir.exists():
            # Fall back to shared dataset
            agent_dataset_dir = dataset_dir

        # Agent output directory
        agent_output_dir = output_base_dir / f"{agent_name}_{version}"

        try:
            result = train_single_agent(
                agent_name=agent_name,
                dataset_dir=agent_dataset_dir,
                output_dir=agent_output_dir,
                config_path=config_path,
                preset=preset,
                version=version
            )
            results[agent_name] = result

            logger.info(f"✅ {agent_name} training complete")
            logger.info(f"   Train loss: {result.final_train_loss:.4f}")
            logger.info(f"   Eval loss: {result.final_eval_loss:.4f}")
            logger.info(f"   Eval gates: {'PASSED' if result.passed_eval_gates else 'FAILED'}")

        except Exception as e:
            logger.error(f"❌ {agent_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            results[agent_name] = None

    # Summary
    successful = sum(1 for r in results.values() if r is not None and r.passed_eval_gates)
    total = len(agents)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {successful}/{total} agents")

    for agent_name, result in results.items():
        if result:
            status = "✅ PASSED" if result.passed_eval_gates else "⚠️  COMPLETED (gates failed)"
            logger.info(f"{agent_name}: {status}")
        else:
            logger.info(f"{agent_name}: ❌ FAILED")

    logger.info(f"{'='*60}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SFT Training for Junior Agents (News, Technical, Fundamental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single agent
  python scripts/train_agent_sft.py \\
      --agent news_agent \\
      --dataset data/datasets/sft_v1 \\
      --output models/sft/news_agent_v1.0.0

  # Train with quick test preset (for development)
  python scripts/train_agent_sft.py \\
      --agent technical_agent \\
      --dataset data/datasets/sft_v1 \\
      --output models/sft/technical_agent_test \\
      --preset quick_test

  # Train all agents
  python scripts/train_agent_sft.py \\
      --agent all \\
      --dataset data/datasets/sft_v1 \\
      --output-dir models/sft

Presets:
  - quick_test: 1 epoch, 100 samples (for development)
  - production: 3 epochs, full dataset (default)
  - high_quality: 5 epochs, optimized hyperparams
        """
    )

    # Agent selection
    parser.add_argument(
        "--agent",
        required=True,
        choices=["news_agent", "technical_agent", "fundamental_agent", "all"],
        help="Agent to train (or 'all' for all agents)"
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset directory containing train.jsonl, val.jsonl"
    )

    # Output
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output",
        type=Path,
        help="Output directory for single agent"
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Base output directory for all agents"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML (default: training/sft/sft_config.yaml)"
    )

    parser.add_argument(
        "--preset",
        choices=["quick_test", "production", "high_quality"],
        help="Training preset"
    )

    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Model version (default: 1.0.0)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    # Create log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"sft_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, level="DEBUG")
    logger.info(f"Logging to {log_file}")

    # Validate dataset
    if not args.dataset.exists():
        logger.error(f"Dataset directory not found: {args.dataset}")
        sys.exit(1)

    # Train
    try:
        if args.agent == "all":
            # Train all agents
            if not args.output_dir:
                logger.error("--output-dir required when training all agents")
                sys.exit(1)

            results = train_all_agents(
                dataset_dir=args.dataset,
                output_base_dir=args.output_dir,
                config_path=args.config,
                preset=args.preset,
                version=args.version
            )

            # Save combined results
            summary_file = args.output_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_file.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_file, "w") as f:
                json.dump({
                    agent: {
                        "final_train_loss": result.final_train_loss if result else None,
                        "final_eval_loss": result.final_eval_loss if result else None,
                        "passed_eval_gates": result.passed_eval_gates if result else False,
                        "training_time_seconds": result.training_time_seconds if result else 0
                    } for agent, result in results.items()
                }, f, indent=2)

            logger.info(f"Training summary saved: {summary_file}")

            # Exit code based on success
            all_passed = all(r and r.passed_eval_gates for r in results.values())
            sys.exit(0 if all_passed else 1)

        else:
            # Train single agent
            if not args.output:
                logger.error("--output required when training single agent")
                sys.exit(1)

            result = train_single_agent(
                agent_name=args.agent,
                dataset_dir=args.dataset,
                output_dir=args.output,
                config_path=args.config,
                preset=args.preset,
                version=args.version
            )

            # Print summary
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE")
            print("="*60)
            print(f"\nAgent: {result.agent_name}")
            print(f"Version: {args.version}")
            print(f"Base Model: {result.base_model}")
            print(f"\nMetrics:")
            print(f"  Final Train Loss: {result.final_train_loss:.4f}")
            print(f"  Final Eval Loss: {result.final_eval_loss:.4f}")
            print(f"  Training Time: {result.training_time_seconds:.1f}s")
            print(f"\nDataset:")
            print(f"  Train Examples: {result.num_train_examples}")
            print(f"  Eval Examples: {result.num_eval_examples}")
            print(f"\nEvaluation Gates: {'✅ PASSED' if result.passed_eval_gates else '❌ FAILED'}")
            print(f"\nModel saved: {args.output}")
            print("="*60)

            # Exit code based on eval gates
            sys.exit(0 if result.passed_eval_gates else 1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
