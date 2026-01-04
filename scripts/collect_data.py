#!/usr/bin/env python3
"""
Data Collection Script - Download historical market data
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from loguru import logger


def collect_historical_data(
    symbols: list,
    days: int,
    output_dir: str
):
    """
    Collect historical price data for symbols

    Args:
        symbols: List of stock symbols
        days: Number of days to look back
        output_dir: Output directory for parquet files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Collecting data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

    for symbol in symbols:
        try:
            logger.info(f"Downloading {symbol}...")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Save as parquet
            output_file = output_path / f"{symbol}.parquet"
            df.to_parquet(output_file)

            logger.info(f"Saved {len(df)} rows for {symbol} to {output_file}")

        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")

    logger.info("Data collection complete")


def main():
    parser = argparse.ArgumentParser(
        description="Collect historical market data"
    )

    parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to look back (default: 365)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    collect_historical_data(symbols, args.days, args.output)


if __name__ == "__main__":
    main()
