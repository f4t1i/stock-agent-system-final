#!/usr/bin/env python3
"""
Run SFT Training Pipeline

Orchestrates the complete SFT training process:
1. Generate synthetic training data
2. Train news, technical, and fundamental agents
3. Save checkpoints and logs
4. Track progress with W&B
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from loguru import logger
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_data import SyntheticDataGenerator
from training.sft.train_news_agent import NewsSFTTrainer
from training.sft.train_technical_agent import TechnicalSFTTrainer
from training.sft.train_fundamental_agent import FundamentalSFTTrainer


class SFTTrainingOrchestrator:
    """
    Orchestrate complete SFT training pipeline.

    Handles:
    - Data generation
    - Sequential agent training
    - Progress tracking
    - Checkpoint management
    """

    def __init__(
        self,
        output_dir: str = "training_runs",
        use_wandb: bool = True,
        provider: str = "openai",
        model: str = "gpt-4o"
    ):
        """
        Initialize orchestrator.

        Args:
            output_dir: Base output directory
            use_wandb: Enable W&B tracking
            provider: LLM provider for data generation
            model: Model for data generation
        """
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        self.provider = provider
        self.model = model

        # Create timestamped run directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"sft_run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="stock-agent-system",
                name=f"sft-pipeline-{self.run_id}",
                config={
                    "run_id": self.run_id,
                    "provider": provider,
                    "model": model
                }
            )

        logger.info(f"SFT Training Orchestrator initialized")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output directory: {self.run_dir}")

    def generate_training_data(
        self,
        num_examples_per_agent: int = 100
    ) -> Dict[str, str]:
        """
        Generate synthetic training data for all agents.

        Args:
            num_examples_per_agent: Number of examples per agent

        Returns:
            Dict mapping agent type to dataset path
        """
        logger.info(f"Generating training data ({num_examples_per_agent} examples per agent)")

        data_dir = self.run_dir / "data"
        data_dir.mkdir(exist_ok=True)

        generator = SyntheticDataGenerator(
            provider=self.provider,
            model=self.model
        )

        agent_types = ['news', 'technical', 'fundamental']
        dataset_paths = {}

        for agent_type in agent_types:
            logger.info(f"Generating data for {agent_type} agent...")

            output_path = data_dir / f"{agent_type}_train.jsonl"

            try:
                examples = generator.generate_examples(
                    agent_type=agent_type,
                    num_examples=num_examples_per_agent,
                    output_path=str(output_path)
                )

                dataset_paths[agent_type] = str(output_path)

                logger.info(f"✓ Generated {len(examples)} examples for {agent_type}")

                if self.use_wandb:
                    wandb.log({
                        f"data_generation/{agent_type}_examples": len(examples)
                    })

            except Exception as e:
                logger.error(f"Error generating data for {agent_type}: {e}")
                raise

        # Save dataset paths
        paths_file = self.run_dir / "dataset_paths.json"
        with open(paths_file, 'w') as f:
            json.dump(dataset_paths, f, indent=2)

        logger.info(f"✓ Training data generation complete")
        logger.info(f"Dataset paths saved to {paths_file}")

        return dataset_paths

    def train_agents(
        self,
        dataset_paths: Dict[str, str],
        max_steps: int = 500
    ):
        """
        Train all agents sequentially.

        Args:
            dataset_paths: Dict mapping agent type to dataset path
            max_steps: Maximum training steps per agent
        """
        logger.info("Starting agent training")

        models_dir = self.run_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Training configurations
        training_configs = {
            'news': {
                'config_path': 'config/sft/news_agent.yaml',
                'trainer_class': NewsSFTTrainer,
                'output_dir': str(models_dir / 'news_agent')
            },
            'technical': {
                'config_path': 'config/sft/technical_agent.yaml',
                'trainer_class': TechnicalSFTTrainer,
                'output_dir': str(models_dir / 'technical_agent')
            },
            'fundamental': {
                'config_path': 'config/sft/fundamental_agent.yaml',
                'trainer_class': FundamentalSFTTrainer,
                'output_dir': str(models_dir / 'fundamental_agent')
            }
        }

        results = {}

        for agent_type, config in training_configs.items():
            if agent_type not in dataset_paths:
                logger.warning(f"No dataset for {agent_type}, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Training {agent_type.upper()} agent")
            logger.info(f"{'='*60}\n")

            try:
                # Update config with dataset path and output dir
                from utils.config_loader import load_config
                agent_config = load_config(config['config_path'])
                agent_config['data']['dataset_path'] = dataset_paths[agent_type]
                agent_config['training']['output_dir'] = config['output_dir']
                agent_config['training']['max_steps'] = max_steps
                agent_config['use_wandb'] = self.use_wandb

                # Save updated config
                config_file = self.run_dir / f"{agent_type}_config.yaml"
                import yaml
                with open(config_file, 'w') as f:
                    yaml.dump(agent_config, f)

                # Initialize and train
                logger.info(f"Initializing {agent_type} trainer...")
                trainer = config['trainer_class'](str(config_file))

                logger.info(f"Starting training for {agent_type}...")
                trainer.train()

                results[agent_type] = {
                    'status': 'success',
                    'output_dir': config['output_dir'],
                    'config_path': str(config_file)
                }

                logger.info(f"✓ {agent_type} training complete")

            except Exception as e:
                logger.error(f"Error training {agent_type}: {e}")
                results[agent_type] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Save results
        results_file = self.run_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("SFT Training Pipeline Complete")
        logger.info(f"{'='*60}\n")
        logger.info(f"Results saved to {results_file}")

        return results

    def run_full_pipeline(
        self,
        num_examples: int = 100,
        max_steps: int = 500
    ):
        """
        Run complete SFT training pipeline.

        Args:
            num_examples: Number of training examples per agent
            max_steps: Maximum training steps per agent
        """
        logger.info("Starting full SFT training pipeline")

        try:
            # Step 1: Generate data
            logger.info("\n[STEP 1/2] Generating training data...")
            dataset_paths = self.generate_training_data(num_examples)

            # Step 2: Train agents
            logger.info("\n[STEP 2/2] Training agents...")
            results = self.train_agents(dataset_paths, max_steps)

            # Summary
            logger.info("\n" + "="*60)
            logger.info("PIPELINE SUMMARY")
            logger.info("="*60)
            logger.info(f"Run ID: {self.run_id}")
            logger.info(f"Output: {self.run_dir}")
            logger.info(f"Training examples: {num_examples} per agent")
            logger.info(f"Training steps: {max_steps} per agent")

            successful = sum(1 for r in results.values() if r['status'] == 'success')
            logger.info(f"\nAgents trained: {successful}/{len(results)}")

            for agent_type, result in results.items():
                status_icon = "✓" if result['status'] == 'success' else "✗"
                logger.info(f"  {status_icon} {agent_type}: {result['status']}")

            logger.info("="*60)

            if self.use_wandb:
                wandb.log({
                    'pipeline/successful_agents': successful,
                    'pipeline/total_agents': len(results)
                })
                wandb.finish()

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run SFT training pipeline for all agents"
    )

    parser.add_argument(
        '--num-examples',
        type=int,
        default=100,
        help='Number of training examples per agent (default: 100)'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Maximum training steps per agent (default: 500)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_runs',
        help='Base output directory (default: training_runs)'
    )

    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic'],
        help='LLM provider for data generation (default: openai)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model for data generation (default: gpt-4o)'
    )

    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B tracking'
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run pipeline
    orchestrator = SFTTrainingOrchestrator(
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        provider=args.provider,
        model=args.model
    )

    orchestrator.run_full_pipeline(
        num_examples=args.num_examples,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
