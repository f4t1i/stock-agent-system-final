#!/usr/bin/env python
"""
Run Backtest Script - One-click backtest execution

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbols AAPL,MSFT --start 2023-01-01 --end 2023-12-31
    make backtest
    make backtest-quick
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.backtester_v2 import BacktesterV2, BacktestConfig
from loguru import logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run backtest with Signal Contract validation and institutional-grade controls'
    )

    # Universe
    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL,MSFT,GOOGL',
        help='Comma-separated list of stock symbols (default: AAPL,MSFT,GOOGL)'
    )

    # Date range
    parser.add_argument(
        '--start',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date YYYY-MM-DD (default: 1 year ago)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date YYYY-MM-DD (default: today)'
    )

    # Capital & costs
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital (default: 100000)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage in basis points (default: 5 bps)'
    )

    # Controls
    parser.add_argument(
        '--validate-signals',
        action='store_true',
        help='Enable signal contract validation (requires coordinator)'
    )

    parser.add_argument(
        '--disable-survivorship-guard',
        action='store_true',
        help='Disable survivorship bias guard'
    )

    parser.add_argument(
        '--disable-corporate-actions',
        action='store_true',
        help='Disable corporate actions handling'
    )

    parser.add_argument(
        '--no-fail-fast',
        action='store_true',
        help='Continue on missing data (instead of failing fast)'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='backtest_results',
        help='Output directory (default: backtest_results)'
    )

    parser.add_argument(
        '--no-save-trades',
        action='store_true',
        help='Do not save trades to file'
    )

    parser.add_argument(
        '--no-save-signals',
        action='store_true',
        help='Do not save signals to file'
    )

    return parser.parse_args()


def main():
    """Main backtest execution"""
    args = parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Create config
    config = BacktestConfig(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage_bps=args.slippage,
        enable_survivorship_bias_guard=not args.disable_survivorship_guard,
        enable_corporate_actions=not args.disable_corporate_actions,
        fail_fast_on_missing_data=not args.no_fail_fast,
        validate_signals=args.validate_signals,
        random_seed=args.seed,
        output_dir=args.output_dir,
        save_trades=not args.no_save_trades,
        save_signals=not args.no_save_signals
    )

    # Log configuration
    logger.info("="*60)
    logger.info("BACKTEST CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Commission: {args.commission:.3%}")
    logger.info(f"Slippage: {args.slippage} bps")
    logger.info(f"Survivorship Bias Guard: {'✅ Enabled' if config.enable_survivorship_bias_guard else '❌ Disabled'}")
    logger.info(f"Corporate Actions: {'✅ Enabled' if config.enable_corporate_actions else '❌ Disabled'}")
    logger.info(f"Signal Validation: {'✅ Enabled' if config.validate_signals else '❌ Disabled'}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info("="*60)

    # Create backtester (no coordinator for now - backtester will use simple strategy)
    backtester = BacktesterV2(config=config, coordinator=None)

    # Run backtest
    try:
        results = backtester.run()

        # Check for errors
        if 'error' in results:
            logger.error(f"❌ Backtest failed: {results['error']}")
            return 1

        # Print summary
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60 + "\n")

        # Key metrics
        print(f"Initial Capital:    ${results['initial_capital']:>12,.2f}")
        print(f"Final Value:        ${results['final_value']:>12,.2f}")
        print(f"Total Return:       {results['total_return']:>12.2%}")
        print(f"Sharpe Ratio:       {results.get('sharpe_ratio', 0):>12.2f}")
        print(f"Max Drawdown:       {results.get('max_drawdown', 0):>12.2%}")
        print(f"Win Rate:           {results.get('win_rate', 0):>12.2%}")
        print(f"\nTotal Trades:       {results['num_trades']:>12,}")
        print(f"Trading Days:       {results['trading_days']:>12,}")

        if results.get('delisted_symbols'):
            print(f"\n⚠️  Delisted Symbols: {', '.join(results['delisted_symbols'])}")

        if results.get('failed_signals', 0) > 0:
            print(f"\n❌ Failed Signals: {results['failed_signals']}")

        print("\n" + "="*60)
        print(f"\n📁 Results saved to: {args.output_dir}/")
        print("="*60 + "\n")

        # Success
        logger.success("✅ Backtest completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
