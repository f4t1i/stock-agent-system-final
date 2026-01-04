#!/usr/bin/env python3
"""
News Collection Script - Fetch and store news articles
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from loguru import logger

from utils.news_fetcher import NewsFetcher


def collect_news(
    symbols: list,
    days: int,
    output_file: str
):
    """
    Collect news articles for symbols

    Args:
        symbols: List of stock symbols
        days: Number of days to look back
        output_file: Output JSON file
    """
    logger.info(f"Collecting news for {len(symbols)} symbols")

    fetcher = NewsFetcher()

    all_news = {}

    for symbol in symbols:
        logger.info(f"Fetching news for {symbol}...")

        try:
            articles = fetcher.get_news(symbol, days=days)

            # Deduplicate
            articles = fetcher.deduplicate_articles(articles)

            all_news[symbol] = articles

            logger.info(f"Found {len(articles)} articles for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            all_news[symbol] = []

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_news, f, indent=2)

    logger.info(f"News saved to {output_path}")

    # Print summary
    total_articles = sum(len(articles) for articles in all_news.values())
    logger.info(f"Total articles collected: {total_articles}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect news articles"
    )

    parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Comma-separated list of symbols'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to look back (default: 30)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/news.json',
        help='Output JSON file'
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    collect_news(symbols, args.days, args.output)


if __name__ == "__main__":
    main()
