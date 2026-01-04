#!/usr/bin/env python3
"""
Main Entry Point für das Self-Improving Stock Analysis Multi-Agent System
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from dotenv import load_dotenv

from orchestration.coordinator import SystemCoordinator
from utils.config_loader import load_config
from utils.logging_setup import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Self-Improving Stock Analysis Multi-Agent System"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch", "backtest", "serve"],
        default="interactive",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        help="Stock symbol for interactive mode (e.g., AAPL)"
    )
    
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="File containing list of symbols for batch mode"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backtesting (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backtesting (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/system.yaml"),
        help="Path to system configuration file"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (serve mode)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results"
    )
    
    return parser.parse_args()


def run_interactive_mode(coordinator: SystemCoordinator, symbol: str):
    """Run interactive analysis for a single symbol"""
    logger.info(f"Starting interactive analysis for {symbol}")
    
    try:
        result = coordinator.analyze_symbol(symbol)
        
        # Print results
        print("\n" + "="*80)
        print(f"Analysis for {symbol}")
        print("="*80)
        print(f"\nRecommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nReasoning:")
        print(result['reasoning'])
        
        if 'technical_analysis' in result:
            print(f"\nTechnical Indicators:")
            for indicator, value in result['technical_analysis'].items():
                print(f"  {indicator}: {value}")
        
        if 'sentiment' in result:
            print(f"\nSentiment Score: {result['sentiment']:.2f}")
        
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        raise


def run_batch_mode(coordinator: SystemCoordinator, symbols_file: Path, output: Optional[Path]):
    """Run batch analysis for multiple symbols"""
    logger.info(f"Starting batch analysis from {symbols_file}")
    
    # Load symbols
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Analyzing {len(symbols)} symbols")
    
    results = []
    for symbol in symbols:
        try:
            logger.info(f"Analyzing {symbol}")
            result = coordinator.analyze_symbol(symbol)
            results.append({
                'symbol': symbol,
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning']
            })
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'error': str(e)
            })
    
    # Save results
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output}")
    
    return results


def run_backtest_mode(
    coordinator: SystemCoordinator,
    start_date: str,
    end_date: str,
    symbols_file: Optional[Path]
):
    """Run backtest over historical period"""
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    from training.rl.backtester import Backtester
    
    backtester = Backtester(coordinator, start_date, end_date)
    
    if symbols_file:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        # Default watchlist
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    results = backtester.run(symbols)
    
    # Print summary
    print("\n" + "="*80)
    print("Backtest Results")
    print("="*80)
    print(f"\nTotal Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print("="*80 + "\n")
    
    return results


def run_serve_mode(coordinator: SystemCoordinator, port: int):
    """Start API server"""
    logger.info(f"Starting API server on port {port}")
    
    from api.server import create_app
    import uvicorn
    
    app = create_app(coordinator)
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("Initializing Stock Analysis Multi-Agent System")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize system coordinator
        coordinator = SystemCoordinator(config)
        
        # Execute based on mode
        if args.mode == "interactive":
            if not args.symbol:
                logger.error("Symbol required for interactive mode")
                sys.exit(1)
            run_interactive_mode(coordinator, args.symbol)
            
        elif args.mode == "batch":
            if not args.symbols_file:
                logger.error("Symbols file required for batch mode")
                sys.exit(1)
            run_batch_mode(coordinator, args.symbols_file, args.output)
            
        elif args.mode == "backtest":
            if not (args.start_date and args.end_date):
                logger.error("Start and end dates required for backtest mode")
                sys.exit(1)
            run_backtest_mode(
                coordinator,
                args.start_date,
                args.end_date,
                args.symbols_file
            )
            
        elif args.mode == "serve":
            run_serve_mode(coordinator, args.port)
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
        sys.exit(1)


if __name__ == "__main__":
    main()
