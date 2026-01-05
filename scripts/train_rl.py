#!/usr/bin/env python3
"""
RL Training Script - Train Senior Strategist with GRPO

Purpose:
    Unified script for RL training using GRPO (Group Relative Policy Optimization).
    Integrates with experience store, supervisor, and regime features.

Usage:
    # Train from SFT checkpoint
    python scripts/train_rl.py \\
        --policy models/sft/strategist_v1.0.0 \\
        --experience-store data/experiences \\
        --output models/rl/strategist_grpo_v1.0.0

    # Quick test
    python scripts/train_rl.py \\
        --policy models/sft/strategist_v1.0.0 \\
        --experience-store data/experiences \\
        --output models/rl/strategist_test \\
        --preset quick_test
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from training.rl.grpo_trainer import GRPOTrainer
    from agents.supervisor_v2 import SupervisorV2
    from agents.regime_features import RegimeFeatureExtractor
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    logger.error(f"RL components not available: {e}")


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RL Training - GRPO for Senior Strategist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from SFT checkpoint
  python scripts/train_rl.py \\
      --policy models/sft/strategist_v1.0.0 \\
      --experience-store data/experiences \\
      --output models/rl/strategist_grpo_v1.0.0 \\
      --iterations 100

  # Quick test (10 iterations)
  python scripts/train_rl.py \\
      --policy models/sft/strategist_v1.0.0 \\
      --experience-store data/experiences \\
      --output models/rl/strategist_test \\
      --preset quick_test
        """
    )

    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to initial policy model (SFT checkpoint)"
    )

    parser.add_argument(
        "--experience-store",
        type=Path,
        required=True,
        help="Path to experience store"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for RL model"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to RL config YAML (default: training/rl/rl_config.yaml)"
    )

    parser.add_argument(
        "--preset",
        choices=["quick_test", "production", "high_quality"],
        help="Training preset"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for training (default: cuda)"
    )

    args = parser.parse_args()

    if not RL_AVAILABLE:
        logger.error("RL training not available - missing dependencies")
        logger.error("Install with: pip install torch transformers peft")
        return 1

    # Print configuration
    print("=" * 60)
    print("RL TRAINING - GRPO")
    print("=" * 60)
    print(f"\nPolicy: {args.policy}")
    print(f"Experience Store: {args.experience_store}")
    print(f"Output: {args.output}")
    print(f"Iterations: {args.iterations}")
    print(f"Preset: {args.preset or 'default'}")
    print(f"Device: {args.device}")
    print()

    # Initialize components
    logger.info("Initializing components...")

    # GRPO Trainer
    trainer = GRPOTrainer(
        policy_path=args.policy,
        config_path=args.config,
        device=args.device
    )

    # Supervisor v2 (for evaluation)
    supervisor = SupervisorV2(config_path=args.config)

    # Regime Features (for context)
    feature_extractor = RegimeFeatureExtractor(config_path=args.config)

    logger.info("✅ Components initialized")

    # Train
    logger.info(f"\nStarting RL training for {args.iterations} iterations...")

    try:
        results = trainer.train(
            experience_store_path=args.experience_store,
            num_iterations=args.iterations,
            output_dir=args.output
        )

        logger.info("\n✅ RL Training complete!")
        logger.info(f"Model saved: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
