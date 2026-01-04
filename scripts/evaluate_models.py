#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates trained models against baseline and previous versions:
1. Run backtesting on test set
2. Compare with baseline performance
3. Generate evaluation report
4. Recommend deployment
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.coordinator import SystemCoordinator
from training.rl.backtester import Backtester
from utils.metrics import calculate_portfolio_metrics


class ModelEvaluator:
    """
    Evaluate model performance and compare with baselines.

    Evaluation metrics:
    - Sharpe Ratio
    - Sortino Ratio
    - Max Drawdown
    - Win Rate
    - Profit Factor
    - Total Return
    """

    def __init__(
        self,
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize evaluator.

        Args:
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"eval_{self.run_id}"
        self.run_dir.mkdir(exist_ok=True)

        logger.info(f"Model Evaluator initialized")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output: {self.run_dir}")

    def evaluate_model(
        self,
        model_path: str,
        test_symbols: List[str],
        test_start_date: str,
        test_end_date: str,
        model_name: str = "model"
    ) -> Dict:
        """
        Evaluate a single model.

        Args:
            model_path: Path to model directory
            test_symbols: Symbols for testing
            test_start_date: Test start date
            test_end_date: Test end date
            model_name: Model name for reporting

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Test period: {test_start_date} to {test_end_date}")
        logger.info(f"Test symbols: {', '.join(test_symbols)}")

        # Initialize coordinator with model
        # In production, load the specific model
        coordinator = SystemCoordinator()

        # Run backtest
        backtester = Backtester(
            coordinator=coordinator,
            start_date=test_start_date,
            end_date=test_end_date,
            initial_capital=100000
        )

        metrics = backtester.run(test_symbols)

        # Add model info
        metrics['model_name'] = model_name
        metrics['model_path'] = model_path
        metrics['test_symbols'] = test_symbols
        metrics['evaluated_at'] = datetime.now().isoformat()

        logger.info(f"✓ {model_name} evaluation complete")

        return metrics

    def compare_models(
        self,
        evaluations: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple model evaluations.

        Args:
            evaluations: List of evaluation results

        Returns:
            Comparison DataFrame
        """
        logger.info(f"Comparing {len(evaluations)} models")

        # Key metrics to compare
        comparison_metrics = [
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'total_return',
            'num_trades'
        ]

        comparison_data = []

        for eval_result in evaluations:
            row = {'model': eval_result['model_name']}

            for metric in comparison_metrics:
                row[metric] = eval_result.get(metric, 0.0)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        return df

    def generate_report(
        self,
        comparison_df: pd.DataFrame,
        evaluations: List[Dict]
    ):
        """
        Generate evaluation report.

        Args:
            comparison_df: Comparison DataFrame
            evaluations: Full evaluation results
        """
        logger.info("Generating evaluation report")

        report_file = self.run_dir / "evaluation_report.md"

        with open(report_file, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Summary table
            f.write("## Performance Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")

            # Best model
            best_sharpe_idx = comparison_df['sharpe_ratio'].idxmax()
            best_model = comparison_df.iloc[best_sharpe_idx]

            f.write("## Recommended Model\n\n")
            f.write(f"**{best_model['model']}** (Best Sharpe Ratio: {best_model['sharpe_ratio']:.3f})\n\n")

            # Key metrics
            f.write("### Key Metrics\n\n")
            f.write(f"- **Sharpe Ratio:** {best_model['sharpe_ratio']:.3f}\n")
            f.write(f"- **Sortino Ratio:** {best_model['sortino_ratio']:.3f}\n")
            f.write(f"- **Max Drawdown:** {best_model['max_drawdown']:.2%}\n")
            f.write(f"- **Win Rate:** {best_model['win_rate']:.2%}\n")
            f.write(f"- **Total Return:** {best_model['total_return']:.2%}\n")
            f.write(f"- **Profit Factor:** {best_model['profit_factor']:.3f}\n\n")

            # Detailed results
            f.write("---\n\n## Detailed Results\n\n")

            for eval_result in evaluations:
                f.write(f"### {eval_result['model_name']}\n\n")
                f.write(f"- **Model Path:** `{eval_result['model_path']}`\n")
                f.write(f"- **Test Period:** {eval_result['start_date']} to {eval_result['end_date']}\n")
                f.write(f"- **Initial Capital:** ${eval_result['initial_capital']:,.2f}\n")
                f.write(f"- **Final Value:** ${eval_result['final_value']:,.2f}\n")
                f.write(f"- **Total Return:** {eval_result.get('total_return', 0):.2%}\n")
                f.write(f"- **Sharpe Ratio:** {eval_result.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"- **Max Drawdown:** {eval_result.get('max_drawdown', 0):.2%}\n")
                f.write(f"- **Win Rate:** {eval_result.get('win_rate', 0):.2%}\n\n")

        logger.info(f"✓ Report saved to {report_file}")

        # Also save as JSON
        json_file = self.run_dir / "evaluation_results.json"
        with open(json_file, 'w') as f:
            json.dump({
                'evaluations': evaluations,
                'comparison': comparison_df.to_dict(orient='records'),
                'best_model': best_model.to_dict()
            }, f, indent=2)

        logger.info(f"✓ JSON results saved to {json_file}")

    def plot_comparison(
        self,
        comparison_df: pd.DataFrame
    ):
        """
        Plot performance comparison.

        Args:
            comparison_df: Comparison DataFrame
        """
        logger.info("Generating comparison plots")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        metrics = [
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('max_drawdown', 'Max Drawdown (%)'),
            ('win_rate', 'Win Rate (%)'),
            ('total_return', 'Total Return (%)'),
            ('profit_factor', 'Profit Factor')
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            values = comparison_df[metric].values
            models = comparison_df['model'].values

            # Convert percentages
            if metric in ['max_drawdown', 'win_rate', 'total_return']:
                values = values * 100

            ax.bar(models, values, color='steelblue')
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        plot_file = self.run_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Plots saved to {plot_file}")

    def evaluate_pipeline(
        self,
        model_configs: List[Dict],
        test_symbols: List[str] = None,
        test_days: int = 90
    ) -> Dict:
        """
        Evaluate multiple models.

        Args:
            model_configs: List of model configurations
            test_symbols: Test symbols
            test_days: Number of test days

        Returns:
            Evaluation summary
        """
        if test_symbols is None:
            test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        # Calculate test period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)

        test_start = start_date.strftime("%Y-%m-%d")
        test_end = end_date.strftime("%Y-%m-%d")

        logger.info(f"\n{'='*60}")
        logger.info("MODEL EVALUATION PIPELINE")
        logger.info(f"{'='*60}")
        logger.info(f"Test period: {test_start} to {test_end}")
        logger.info(f"Test symbols: {', '.join(test_symbols)}")
        logger.info(f"Models to evaluate: {len(model_configs)}")
        logger.info(f"{'='*60}\n")

        # Evaluate each model
        evaluations = []

        for config in model_configs:
            try:
                metrics = self.evaluate_model(
                    model_path=config['model_path'],
                    test_symbols=test_symbols,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    model_name=config['name']
                )

                evaluations.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {config['name']}: {e}")

        # Compare models
        comparison_df = self.compare_models(evaluations)

        # Generate report
        self.generate_report(comparison_df, evaluations)

        # Plot comparison
        self.plot_comparison(comparison_df)

        # Summary
        best_idx = comparison_df['sharpe_ratio'].idxmax()
        best_model = comparison_df.iloc[best_idx]

        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Models evaluated: {len(evaluations)}")
        logger.info(f"Best model: {best_model['model']}")
        logger.info(f"Best Sharpe Ratio: {best_model['sharpe_ratio']:.3f}")
        logger.info(f"Results saved to: {self.run_dir}")
        logger.info(f"{'='*60}\n")

        return {
            'evaluations': evaluations,
            'comparison': comparison_df.to_dict(orient='records'),
            'best_model': best_model.to_dict(),
            'output_dir': str(self.run_dir)
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models"
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Model paths to evaluate (format: name:path)'
    )

    parser.add_argument(
        '--test-symbols',
        type=str,
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        help='Test symbols'
    )

    parser.add_argument(
        '--test-days',
        type=int,
        default=90,
        help='Number of test days (default: 90)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory'
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Parse model configs
    model_configs = []
    for model_spec in args.models:
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
        else:
            name = Path(model_spec).name
            path = model_spec

        model_configs.append({
            'name': name,
            'model_path': path
        })

    # Run evaluation
    evaluator = ModelEvaluator(output_dir=args.output_dir)

    evaluator.evaluate_pipeline(
        model_configs=model_configs,
        test_symbols=args.test_symbols,
        test_days=args.test_days
    )


if __name__ == '__main__':
    main()
