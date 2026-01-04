#!/usr/bin/env python3
"""
Run RL Training Pipeline

Orchestrates RL training for the strategist agent:
1. Collect trajectories using experience library
2. Train with GRPO or PPO
3. Evaluate improvements
4. Save best models
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

from orchestration.coordinator import SystemCoordinator
from training.data_synthesis.experience_library import ExperienceLibrary
from training.rl.backtester import Backtester


class RLTrainingOrchestrator:
    """
    Orchestrate RL training pipeline for strategist.

    Workflow:
    1. Collect trajectories through backtesting
    2. Store in experience library
    3. Run RL training (GRPO/PPO)
    4. Evaluate model performance
    5. Deploy best model
    """

    def __init__(
        self,
        output_dir: str = "training_runs",
        use_wandb: bool = True,
        algorithm: str = "grpo"
    ):
        """
        Initialize orchestrator.

        Args:
            output_dir: Base output directory
            use_wandb: Enable W&B tracking
            algorithm: RL algorithm (grpo or ppo)
        """
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        self.algorithm = algorithm

        # Create timestamped run directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"rl_{algorithm}_run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experience library
        self.experience_lib = ExperienceLibrary(
            db_path=str(self.run_dir / "experience_library.db")
        )

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="stock-agent-system",
                name=f"rl-{algorithm}-{self.run_id}",
                config={
                    "run_id": self.run_id,
                    "algorithm": algorithm
                }
            )

        logger.info(f"RL Training Orchestrator initialized")
        logger.info(f"Algorithm: {algorithm.upper()}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output directory: {self.run_dir}")

    def collect_trajectories(
        self,
        num_trajectories: int = 500,
        symbols: List[str] = None,
        backtest_period_days: int = 365
    ) -> Dict:
        """
        Collect trajectories through system execution.

        Args:
            num_trajectories: Target number of trajectories
            symbols: List of symbols to trade
            backtest_period_days: Days to backtest

        Returns:
            Collection statistics
        """
        logger.info(f"Collecting {num_trajectories} trajectories")

        if symbols is None:
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'JPM', 'V', 'WMT', 'PG', 'UNH', 'DIS', 'NFLX', 'ADBE'
            ]

        logger.info(f"Using {len(symbols)} symbols: {', '.join(symbols[:5])}...")

        # Initialize coordinator
        coordinator = SystemCoordinator()

        # Calculate date range
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backtest_period_days)

        # Collect trajectories through backtesting
        collected = 0
        successful = 0
        failed = 0

        # Run multiple backtest periods to collect diverse trajectories
        periods = [
            (30, "1mo"),   # Last month
            (90, "3mo"),   # Last quarter
            (180, "6mo"),  # Last 6 months
            (365, "1yr")   # Last year
        ]

        for days, period_name in periods:
            if collected >= num_trajectories:
                break

            logger.info(f"\nCollecting trajectories for {period_name} period...")

            period_start = end_date - timedelta(days=days)

            try:
                # Run backtest
                backtester = Backtester(
                    coordinator=coordinator,
                    start_date=period_start.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    initial_capital=100000
                )

                metrics = backtester.run(symbols)

                # Store trajectories in experience library
                for trade in backtester.trades:
                    # Compute reward based on P&L
                    pnl = trade.get('pnl', 0)
                    reward = self._compute_reward(pnl, metrics)

                    # Store trajectory
                    trajectory_data = {
                        'timestamp': trade['date'].isoformat() if hasattr(trade['date'], 'isoformat') else str(trade['date']),
                        'trajectory': [
                            {
                                'step': 'analysis',
                                'symbol': trade['symbol'],
                                'action': trade['action']
                            }
                        ],
                        'pnl': pnl,
                        'metrics': metrics
                    }

                    try:
                        self.experience_lib.add_trajectory(
                            symbol=trade['symbol'],
                            agent_type='strategist',
                            trajectory_data=trajectory_data,
                            reward=reward,
                            final_decision=trade['action'],
                            market_data={'volatility': metrics.get('volatility', 0.0)}
                        )

                        collected += 1
                        if reward > 0.5:
                            successful += 1
                        else:
                            failed += 1

                    except Exception as e:
                        logger.error(f"Error storing trajectory: {e}")

                logger.info(f"✓ {period_name}: Collected {len(backtester.trades)} trajectories")

                if self.use_wandb:
                    wandb.log({
                        f'collection/{period_name}_trajectories': len(backtester.trades),
                        f'collection/{period_name}_sharpe': metrics.get('sharpe_ratio', 0),
                        'collection/total_collected': collected
                    })

            except Exception as e:
                logger.error(f"Error in {period_name} backtest: {e}")

        # Get statistics
        stats = self.experience_lib.get_statistics()

        logger.info(f"\n{'='*60}")
        logger.info("TRAJECTORY COLLECTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total collected: {collected}")
        logger.info(f"Successful (reward > 0.5): {successful}")
        logger.info(f"Failed (reward <= 0.5): {failed}")
        logger.info(f"Success rate: {successful/collected*100:.1f}%" if collected > 0 else "N/A")
        logger.info(f"{'='*60}\n")

        # Save collection stats
        stats_file = self.run_dir / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'total_collected': collected,
                'successful': successful,
                'failed': failed,
                'library_stats': stats
            }, f, indent=2)

        return {
            'total_collected': collected,
            'successful': successful,
            'failed': failed,
            'stats': stats
        }

    def _compute_reward(self, pnl: float, metrics: Dict) -> float:
        """
        Compute reward from P&L and metrics.

        Args:
            pnl: Profit/loss
            metrics: Performance metrics

        Returns:
            Reward value [0, 1]
        """
        # Normalize P&L to reward
        # Positive P&L -> reward > 0.5
        # Negative P&L -> reward < 0.5

        if pnl > 0:
            reward = 0.5 + min(pnl / 1000, 0.5)  # Cap at 1.0
        else:
            reward = 0.5 + max(pnl / 1000, -0.5)  # Floor at 0.0

        # Adjust by Sharpe ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 0:
            reward = min(reward * (1 + sharpe * 0.1), 1.0)

        return max(0.0, min(reward, 1.0))

    def train_strategist(
        self,
        num_iterations: int = 100
    ):
        """
        Train strategist using collected trajectories.

        Args:
            num_iterations: Number of training iterations
        """
        logger.info(f"Training strategist with {self.algorithm.upper()}")

        if self.algorithm == "grpo":
            from training.rl.train_strategist_grpo import GRPOTrainer

            # Create config
            config = {
                'model': {
                    'sft_checkpoint': 'models/strategist/sft/final',
                    'lora_rank': 8,
                    'lora_alpha': 16,
                    'learning_rate': 1e-5
                },
                'training': {
                    'output_dir': str(self.run_dir / 'models'),
                    'num_iterations': num_iterations,
                    'batch_size': 8,
                    'save_interval': 20
                },
                'use_wandb': self.use_wandb,
                'wandb_project': 'stock-agent-system'
            }

            # Save config
            import yaml
            config_file = self.run_dir / 'grpo_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            # Train
            trainer = GRPOTrainer(str(config_file))
            trainer.train(num_iterations=num_iterations)

        elif self.algorithm == "ppo":
            from training.rl.train_strategist_ppo import PPOTrainer

            # Create config
            config = {
                'model': {
                    'sft_checkpoint': 'models/strategist/sft/final',
                    'lora_rank': 8,
                    'lora_alpha': 16,
                    'learning_rate': 1e-5,
                    'value_lr': 1e-4
                },
                'training': {
                    'output_dir': str(self.run_dir / 'models'),
                    'clip_epsilon': 0.2,
                    'value_coef': 0.5,
                    'entropy_coef': 0.01,
                    'save_interval': 20
                },
                'use_wandb': self.use_wandb,
                'wandb_project': 'stock-agent-system'
            }

            # Save config
            import yaml
            config_file = self.run_dir / 'ppo_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

            # Train
            trainer = PPOTrainer(str(config_file))
            trainer.train(num_iterations=num_iterations)

        logger.info(f"✓ Strategist training complete")

    def run_full_pipeline(
        self,
        num_trajectories: int = 500,
        num_iterations: int = 100,
        symbols: List[str] = None
    ):
        """
        Run complete RL training pipeline.

        Args:
            num_trajectories: Number of trajectories to collect
            num_iterations: Number of training iterations
            symbols: Trading symbols
        """
        logger.info("Starting full RL training pipeline")

        try:
            # Step 1: Collect trajectories
            logger.info("\n[STEP 1/2] Collecting trajectories...")
            collection_stats = self.collect_trajectories(
                num_trajectories=num_trajectories,
                symbols=symbols
            )

            # Step 2: Train strategist
            logger.info("\n[STEP 2/2] Training strategist...")
            self.train_strategist(num_iterations=num_iterations)

            # Summary
            logger.info("\n" + "="*60)
            logger.info("RL PIPELINE SUMMARY")
            logger.info("="*60)
            logger.info(f"Algorithm: {self.algorithm.upper()}")
            logger.info(f"Run ID: {self.run_id}")
            logger.info(f"Output: {self.run_dir}")
            logger.info(f"Trajectories collected: {collection_stats['total_collected']}")
            logger.info(f"Training iterations: {num_iterations}")
            logger.info(f"Success rate: {collection_stats['successful']/collection_stats['total_collected']*100:.1f}%")
            logger.info("="*60)

            if self.use_wandb:
                wandb.finish()

        except Exception as e:
            logger.error(f"RL pipeline failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run RL training pipeline for strategist"
    )

    parser.add_argument(
        '--num-trajectories',
        type=int,
        default=500,
        help='Number of trajectories to collect (default: 500)'
    )

    parser.add_argument(
        '--num-iterations',
        type=int,
        default=100,
        help='Number of training iterations (default: 100)'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='grpo',
        choices=['grpo', 'ppo'],
        help='RL algorithm (default: grpo)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=None,
        help='Trading symbols (default: top 15 stocks)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_runs',
        help='Base output directory (default: training_runs)'
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
    orchestrator = RLTrainingOrchestrator(
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        algorithm=args.algorithm
    )

    orchestrator.run_full_pipeline(
        num_trajectories=args.num_trajectories,
        num_iterations=args.num_iterations,
        symbols=args.symbols
    )


if __name__ == '__main__':
    main()
