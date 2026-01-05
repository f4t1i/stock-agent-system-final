#!/usr/bin/env python3
"""
Continuous Training Pipeline - Medium-Term (1-3 Months)

Implements continuous learning loop for achieving medium-term goals:
- Accumulate 10,000+ trajectories
- Achieve Sharpe > 1.5
- Achieve Win Rate > 55%
- Track and publish benchmarks

This script runs continuously, collecting data, training, and improving models.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.coordinator import SystemCoordinator
from training.data_synthesis.experience_library import ExperienceLibrary
from training.rl.backtester import Backtester
from scripts.run_rl_training import RLTrainingOrchestrator
from scripts.evaluate_models import ModelEvaluator
from scripts.deploy_models import ModelDeployer


class ContinuousTrainingPipeline:
    """
    Continuous training pipeline for medium-term goals.

    Features:
    - Continuous trajectory collection
    - Incremental model updates
    - Performance tracking
    - Automatic deployment when goals met
    - Benchmark publishing
    """

    def __init__(
        self,
        output_dir: str = "continuous_training",
        use_wandb: bool = True,
        target_trajectories: int = 10000,
        target_sharpe: float = 1.5,
        target_win_rate: float = 0.55
    ):
        """
        Initialize continuous training pipeline.

        Args:
            output_dir: Output directory
            use_wandb: Enable W&B tracking
            target_trajectories: Target trajectory count
            target_sharpe: Target Sharpe ratio
            target_win_rate: Target win rate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.target_trajectories = target_trajectories
        self.target_sharpe = target_sharpe
        self.target_win_rate = target_win_rate

        # Initialize experience library
        self.experience_lib = ExperienceLibrary(
            db_path=str(self.output_dir / "experience_library.db")
        )

        # Initialize coordinator
        self.coordinator = SystemCoordinator()

        # State tracking
        self.state_file = self.output_dir / "training_state.json"
        self.state = self._load_state()

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="stock-agent-system",
                name=f"continuous-training-{self.state['run_id']}",
                tags=["continuous", "medium-term"],
                config={
                    "target_trajectories": target_trajectories,
                    "target_sharpe": target_sharpe,
                    "target_win_rate": target_win_rate
                },
                resume="allow",
                id=self.state['run_id']
            )

        logger.info("="*70)
        logger.info("CONTINUOUS TRAINING PIPELINE - MEDIUM-TERM (1-3 MONTHS)")
        logger.info("="*70)
        logger.info(f"Run ID: {self.state['run_id']}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"\nTargets:")
        logger.info(f"  Trajectories: {target_trajectories:,}")
        logger.info(f"  Sharpe Ratio: > {target_sharpe}")
        logger.info(f"  Win Rate: > {target_win_rate*100:.1f}%")
        logger.info("="*70 + "\n")

    def _load_state(self) -> Dict:
        """Load or initialize training state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded existing state: {state['total_trajectories']:,} trajectories")
        else:
            state = {
                'run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'start_time': datetime.now().isoformat(),
                'total_trajectories': 0,
                'training_iterations': 0,
                'best_sharpe': 0.0,
                'best_win_rate': 0.0,
                'deployments': [],
                'benchmarks': []
            }

        return state

    def _save_state(self):
        """Save training state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def collect_trajectories_batch(
        self,
        batch_size: int = 1000,
        symbols: List[str] = None,
        days_per_period: int = 30
    ) -> Dict:
        """
        Collect a batch of trajectories.

        Args:
            batch_size: Target trajectories in this batch
            symbols: Trading symbols
            days_per_period: Days per backtest period

        Returns:
            Collection statistics
        """
        logger.info(f"\nCollecting trajectory batch (target: {batch_size})")

        if symbols is None:
            # Expanded symbol list for diversity
            symbols = [
                # Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
                # Finance
                'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'V', 'MA',
                # Consumer
                'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS',
                # Healthcare
                'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'DHR',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB',
                # Industrial
                'BA', 'CAT', 'GE', 'MMM'
            ]

        collected = 0
        successful = 0
        failed = 0

        # Generate multiple backtest periods for diversity
        end_date = datetime.now()
        periods = []

        # Create rolling windows
        for i in range(0, 365, days_per_period):
            period_end = end_date - timedelta(days=i)
            period_start = period_end - timedelta(days=days_per_period)

            periods.append({
                'name': f'period_{i//days_per_period}',
                'start': period_start.strftime("%Y-%m-%d"),
                'end': period_end.strftime("%Y-%m-%d")
            })

            if len(periods) * len(symbols) >= batch_size:
                break

        logger.info(f"Using {len(symbols)} symbols across {len(periods)} periods")

        for period in periods:
            if collected >= batch_size:
                break

            try:
                # Sample symbols for this period
                import random
                period_symbols = random.sample(symbols, min(10, len(symbols)))

                # Run backtest
                backtester = Backtester(
                    coordinator=self.coordinator,
                    start_date=period['start'],
                    end_date=period['end'],
                    initial_capital=100000
                )

                metrics = backtester.run(period_symbols)

                # Store trajectories
                for trade in backtester.trades:
                    pnl = trade.get('pnl', 0)
                    reward = self._compute_reward(pnl, metrics)

                    trajectory_data = {
                        'timestamp': str(trade.get('date', '')),
                        'period': period['name'],
                        'trajectory': [
                            {
                                'step': 'trade_execution',
                                'symbol': trade['symbol'],
                                'action': trade['action'],
                                'quantity': trade.get('quantity', 0),
                                'price': trade.get('price', 0)
                            }
                        ],
                        'pnl': pnl,
                        'metrics': metrics
                    }

                    self.experience_lib.add_trajectory(
                        symbol=trade['symbol'],
                        agent_type='strategist',
                        trajectory_data=trajectory_data,
                        reward=reward,
                        final_decision=trade['action'],
                        market_data={
                            'volatility': metrics.get('volatility', 0.0),
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0)
                        }
                    )

                    collected += 1
                    if reward > 0.5:
                        successful += 1
                    else:
                        failed += 1

                logger.info(f"  {period['name']}: +{len(backtester.trades)} trajectories")

            except Exception as e:
                logger.error(f"Error in period {period['name']}: {e}")

        stats = {
            'collected': collected,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / collected if collected > 0 else 0
        }

        # Update state
        self.state['total_trajectories'] += collected
        self._save_state()

        if self.use_wandb:
            wandb.log({
                'collection/batch_size': collected,
                'collection/total_trajectories': self.state['total_trajectories'],
                'collection/success_rate': stats['success_rate'],
                'targets/trajectory_progress': self.state['total_trajectories'] / self.target_trajectories
            })

        logger.info(f"\n✓ Batch collection complete: {collected:,} trajectories")
        logger.info(f"  Total accumulated: {self.state['total_trajectories']:,}/{self.target_trajectories:,}")
        logger.info(f"  Progress: {self.state['total_trajectories']/self.target_trajectories*100:.1f}%")

        return stats

    def _compute_reward(self, pnl: float, metrics: Dict) -> float:
        """Compute reward from P&L and metrics"""
        # Base reward from P&L
        if pnl > 0:
            reward = 0.5 + min(pnl / 2000, 0.4)
        else:
            reward = 0.5 + max(pnl / 2000, -0.4)

        # Bonus for high Sharpe
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 1.0:
            reward = min(reward * 1.2, 1.0)

        # Penalty for high drawdown
        drawdown = abs(metrics.get('max_drawdown', 0))
        if drawdown > 0.15:
            reward *= 0.9

        return max(0.0, min(reward, 1.0))

    def train_iteration(
        self,
        num_iterations: int = 50,
        algorithm: str = "grpo"
    ):
        """
        Run training iteration.

        Args:
            num_iterations: Training iterations
            algorithm: RL algorithm
        """
        logger.info(f"\nTraining iteration {self.state['training_iterations'] + 1}")

        # Use RL training orchestrator
        rl_orchestrator = RLTrainingOrchestrator(
            output_dir=str(self.output_dir / "rl_training"),
            use_wandb=self.use_wandb,
            algorithm=algorithm
        )

        # Copy experience library
        import shutil
        dest_lib = Path(rl_orchestrator.run_dir) / "experience_library.db"
        shutil.copy(self.experience_lib.db_path, dest_lib)

        # Train
        try:
            rl_orchestrator.train_strategist(num_iterations=num_iterations)

            self.state['training_iterations'] += 1
            self._save_state()

            logger.info(f"✓ Training iteration complete")

        except Exception as e:
            logger.error(f"Training iteration failed: {e}")

    def evaluate_performance(
        self,
        test_symbols: List[str] = None,
        test_days: int = 90
    ) -> Dict:
        """
        Evaluate current model performance.

        Args:
            test_symbols: Test symbols
            test_days: Test period

        Returns:
            Performance metrics
        """
        logger.info("\nEvaluating model performance...")

        if test_symbols is None:
            test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        # Run backtest on test set
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)

        backtester = Backtester(
            coordinator=self.coordinator,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=100000
        )

        metrics = backtester.run(test_symbols)

        # Extract key metrics
        performance = {
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'total_return': metrics.get('total_return', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'profit_factor': metrics.get('profit_factor', 0.0),
            'evaluated_at': datetime.now().isoformat()
        }

        # Update best metrics
        if performance['sharpe_ratio'] > self.state['best_sharpe']:
            self.state['best_sharpe'] = performance['sharpe_ratio']

        if performance['win_rate'] > self.state['best_win_rate']:
            self.state['best_win_rate'] = performance['win_rate']

        self._save_state()

        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'performance/sharpe_ratio': performance['sharpe_ratio'],
                'performance/win_rate': performance['win_rate'],
                'performance/total_return': performance['total_return'],
                'performance/max_drawdown': performance['max_drawdown'],
                'targets/sharpe_achieved': performance['sharpe_ratio'] >= self.target_sharpe,
                'targets/win_rate_achieved': performance['win_rate'] >= self.target_win_rate
            })

        logger.info(f"\n📊 Performance Metrics:")
        logger.info(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f} (target: {self.target_sharpe})")
        logger.info(f"  Win Rate: {performance['win_rate']*100:.1f}% (target: {self.target_win_rate*100:.1f}%)")
        logger.info(f"  Total Return: {performance['total_return']*100:.1f}%")
        logger.info(f"  Max Drawdown: {performance['max_drawdown']*100:.1f}%")

        # Add to benchmark history
        self.state['benchmarks'].append(performance)
        self._save_state()

        return performance

    def check_goals_achieved(self, performance: Dict) -> bool:
        """Check if all medium-term goals are achieved"""
        trajectories_achieved = self.state['total_trajectories'] >= self.target_trajectories
        sharpe_achieved = performance['sharpe_ratio'] >= self.target_sharpe
        win_rate_achieved = performance['win_rate'] >= self.target_win_rate

        logger.info(f"\n🎯 Goal Status:")
        logger.info(f"  ✓ Trajectories: {self.state['total_trajectories']:,}/{self.target_trajectories:,}" if trajectories_achieved else f"  ⏳ Trajectories: {self.state['total_trajectories']:,}/{self.target_trajectories:,}")
        logger.info(f"  ✓ Sharpe > {self.target_sharpe}: {performance['sharpe_ratio']:.3f}" if sharpe_achieved else f"  ⏳ Sharpe > {self.target_sharpe}: {performance['sharpe_ratio']:.3f}")
        logger.info(f"  ✓ Win Rate > {self.target_win_rate*100:.1f}%: {performance['win_rate']*100:.1f}%" if win_rate_achieved else f"  ⏳ Win Rate > {self.target_win_rate*100:.1f}%: {performance['win_rate']*100:.1f}%")

        all_achieved = trajectories_achieved and sharpe_achieved and win_rate_achieved

        if all_achieved:
            logger.info("\n🎉 ALL MEDIUM-TERM GOALS ACHIEVED! 🎉")

        return all_achieved

    def publish_benchmarks(self):
        """Publish performance benchmarks"""
        logger.info("\nPublishing performance benchmarks...")

        benchmark_file = self.output_dir / "performance_benchmarks.md"

        with open(benchmark_file, 'w') as f:
            f.write("# Stock Agent System - Performance Benchmarks\n\n")
            f.write(f"**Published:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Trajectories:** {self.state['total_trajectories']:,}\n")
            f.write(f"- **Training Iterations:** {self.state['training_iterations']}\n")
            f.write(f"- **Best Sharpe Ratio:** {self.state['best_sharpe']:.3f}\n")
            f.write(f"- **Best Win Rate:** {self.state['best_win_rate']*100:.1f}%\n\n")

            # Latest performance
            if self.state['benchmarks']:
                latest = self.state['benchmarks'][-1]

                f.write("## Latest Performance\n\n")
                f.write(f"**Evaluated:** {latest['evaluated_at']}\n\n")
                f.write(f"| Metric | Value | Target | Status |\n")
                f.write(f"|--------|-------|--------|--------|\n")
                f.write(f"| Sharpe Ratio | {latest['sharpe_ratio']:.3f} | {self.target_sharpe} | {'✅' if latest['sharpe_ratio'] >= self.target_sharpe else '⏳'} |\n")
                f.write(f"| Win Rate | {latest['win_rate']*100:.1f}% | {self.target_win_rate*100:.1f}% | {'✅' if latest['win_rate'] >= self.target_win_rate else '⏳'} |\n")
                f.write(f"| Total Return | {latest['total_return']*100:.1f}% | - | - |\n")
                f.write(f"| Max Drawdown | {latest['max_drawdown']*100:.1f}% | <15% | {'✅' if abs(latest['max_drawdown']) < 0.15 else '⚠️'} |\n")
                f.write(f"| Profit Factor | {latest.get('profit_factor', 0):.2f} | >1.5 | {'✅' if latest.get('profit_factor', 0) > 1.5 else '⏳'} |\n\n")

            # Historical performance
            if len(self.state['benchmarks']) > 1:
                f.write("## Performance History\n\n")
                f.write("| Date | Sharpe | Win Rate | Return | Drawdown |\n")
                f.write("|------|--------|----------|--------|----------|\n")

                for bm in self.state['benchmarks'][-10:]:  # Last 10
                    date = datetime.fromisoformat(bm['evaluated_at']).strftime('%Y-%m-%d')
                    f.write(f"| {date} | {bm['sharpe_ratio']:.3f} | {bm['win_rate']*100:.1f}% | {bm['total_return']*100:.1f}% | {bm['max_drawdown']*100:.1f}% |\n")

        logger.info(f"✓ Benchmarks published to {benchmark_file}")

        # Also save JSON
        json_file = self.output_dir / "benchmarks.json"
        with open(json_file, 'w') as f:
            json.dump(self.state['benchmarks'], f, indent=2)

        logger.info(f"✓ JSON benchmarks saved to {json_file}")

    def run_continuous_loop(
        self,
        collection_batch_size: int = 1000,
        training_interval: int = 5,
        evaluation_interval: int = 10,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous training loop.

        Args:
            collection_batch_size: Trajectories per collection batch
            training_interval: Train every N collection batches
            evaluation_interval: Evaluate every N collection batches
            max_iterations: Maximum iterations (None for unlimited)
        """
        logger.info("\n🔄 Starting continuous training loop...\n")

        iteration = 0

        while True:
            iteration += 1

            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {iteration}")
            logger.info(f"{'='*70}\n")

            # 1. Collect trajectories
            try:
                self.collect_trajectories_batch(batch_size=collection_batch_size)
            except Exception as e:
                logger.error(f"Collection failed: {e}")

            # 2. Train periodically
            if iteration % training_interval == 0:
                try:
                    self.train_iteration()
                except Exception as e:
                    logger.error(f"Training failed: {e}")

            # 3. Evaluate periodically
            if iteration % evaluation_interval == 0:
                try:
                    performance = self.evaluate_performance()

                    # Check if goals achieved
                    if self.check_goals_achieved(performance):
                        logger.info("\n🎯 Medium-term goals achieved!")
                        logger.info("Publishing final benchmarks...")
                        self.publish_benchmarks()

                        if self.use_wandb:
                            wandb.log({"goals/all_achieved": True})

                        logger.info("\n✅ Continuous training complete!")
                        break

                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")

            # 4. Publish benchmarks periodically
            if iteration % (evaluation_interval * 2) == 0:
                self.publish_benchmarks()

            # Check max iterations
            if max_iterations and iteration >= max_iterations:
                logger.info(f"\nReached maximum iterations ({max_iterations})")
                break

            # Progress update
            logger.info(f"\n📊 Overall Progress:")
            logger.info(f"  Trajectories: {self.state['total_trajectories']:,}/{self.target_trajectories:,} ({self.state['total_trajectories']/self.target_trajectories*100:.1f}%)")
            logger.info(f"  Best Sharpe: {self.state['best_sharpe']:.3f}/{self.target_sharpe}")
            logger.info(f"  Best Win Rate: {self.state['best_win_rate']*100:.1f}%/{self.target_win_rate*100:.1f}%")

        if self.use_wandb:
            wandb.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run continuous training pipeline for medium-term goals"
    )

    parser.add_argument('--target-trajectories', type=int, default=10000,
                       help='Target trajectory count (default: 10000)')
    parser.add_argument('--target-sharpe', type=float, default=1.5,
                       help='Target Sharpe ratio (default: 1.5)')
    parser.add_argument('--target-win-rate', type=float, default=0.55,
                       help='Target win rate (default: 0.55)')

    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Trajectories per collection batch (default: 1000)')
    parser.add_argument('--training-interval', type=int, default=5,
                       help='Train every N batches (default: 5)')
    parser.add_argument('--evaluation-interval', type=int, default=10,
                       help='Evaluate every N batches (default: 10)')

    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum iterations (default: unlimited)')
    parser.add_argument('--output-dir', type=str, default='continuous_training',
                       help='Output directory')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B tracking')

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run continuous training
    pipeline = ContinuousTrainingPipeline(
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        target_trajectories=args.target_trajectories,
        target_sharpe=args.target_sharpe,
        target_win_rate=args.target_win_rate
    )

    pipeline.run_continuous_loop(
        collection_batch_size=args.batch_size,
        training_interval=args.training_interval,
        evaluation_interval=args.evaluation_interval,
        max_iterations=args.max_iterations
    )


if __name__ == '__main__':
    main()
