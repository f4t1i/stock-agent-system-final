#!/usr/bin/env python3
"""
OpenBB Integration - Comprehensive Market Data Provider

Integrates OpenBB Platform for:
- Real-time and historical market data
- Technical indicators and analysis
- Fundamental financial data
- Economic data
- News and sentiment
- Advanced charting

OpenBB provides professional-grade financial data and analytics.
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
    logger.info("OpenBB Platform loaded successfully")
except ImportError:
    OPENBB_AVAILABLE = False
    logger.warning("OpenBB not available. Install with: pip install openbb")


class OpenBBProvider:
    """
    Comprehensive data provider using OpenBB Platform.

    Provides unified access to:
    - Market data (stocks, options, futures, crypto, forex)
    - Technical analysis (indicators, patterns, signals)
    - Fundamental data (financials, ratios, estimates)
    - News and sentiment
    - Economic indicators
    - Portfolio analytics
    """

    def __init__(self):
        """Initialize OpenBB provider"""
        if not OPENBB_AVAILABLE:
            raise ImportError("OpenBB not installed. Install with: pip install openbb")

        self.obb = obb
        logger.info("OpenBB Provider initialized")

    # ========== Market Data ==========

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., AAPL)

        Returns:
            Quote data including price, volume, change, etc.
        """
        try:
            quote = self.obb.equity.price.quote(symbol=symbol)

            if quote and hasattr(quote, 'results'):
                data = quote.results[0] if quote.results else None

                if data:
                    return {
                        'symbol': symbol,
                        'price': data.last_price if hasattr(data, 'last_price') else data.price,
                        'open': data.open,
                        'high': data.high,
                        'low': data.low,
                        'volume': data.volume,
                        'change': data.change if hasattr(data, 'change') else None,
                        'change_percent': data.change_percent if hasattr(data, 'change_percent') else None,
                        'timestamp': datetime.now().isoformat()
                    }

            logger.warning(f"No quote data for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Default to last year if no dates specified
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")

            if not start_date:
                start = datetime.now() - timedelta(days=365)
                start_date = start.strftime("%Y-%m-%d")

            data = self.obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )

            if data and hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
                return df

            logger.warning(f"No historical data for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    # ========== Technical Analysis ==========

    def get_technical_indicators(
        self,
        symbol: str,
        period: int = 30
    ) -> Optional[Dict]:
        """
        Calculate comprehensive technical indicators.

        Args:
            symbol: Stock symbol
            period: Lookback period in days

        Returns:
            Dict with various technical indicators
        """
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period * 2)  # Extra for calculation

            df = self.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )

            if df is None or df.empty:
                return None

            indicators = {}

            # Moving Averages
            indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1]
            indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD
            macd = indicators['ema_12'] - indicators['ema_26']
            signal = df['close'].ewm(span=9).mean().iloc[-1]
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = macd - signal

            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]

            # Current price
            indicators['current_price'] = df['close'].iloc[-1]

            # Volume
            indicators['volume'] = df['volume'].iloc[-1]
            indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean().iloc[-1]

            # Trend direction
            if indicators['current_price'] > indicators['sma_20']:
                indicators['trend'] = 'bullish'
            elif indicators['current_price'] < indicators['sma_20']:
                indicators['trend'] = 'bearish'
            else:
                indicators['trend'] = 'neutral'

            return indicators

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return None

    def get_support_resistance(
        self,
        symbol: str,
        period: int = 90
    ) -> Optional[Dict]:
        """
        Identify support and resistance levels.

        Args:
            symbol: Stock symbol
            period: Lookback period

        Returns:
            Dict with support and resistance levels
        """
        try:
            df = self.get_historical_data(
                symbol=symbol,
                start_date=(datetime.now() - timedelta(days=period)).strftime("%Y-%m-%d")
            )

            if df is None or df.empty:
                return None

            # Find local maxima (resistance) and minima (support)
            highs = df['high'].rolling(window=5, center=True).max()
            lows = df['low'].rolling(window=5, center=True).min()

            resistance_levels = df[df['high'] == highs]['high'].sort_values(ascending=False).head(3).tolist()
            support_levels = df[df['low'] == lows]['low'].sort_values().head(3).tolist()

            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'current_price': df['close'].iloc[-1]
            }

        except Exception as e:
            logger.error(f"Error finding support/resistance for {symbol}: {e}")
            return None

    # ========== Fundamental Data ==========

    def get_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive fundamental data.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with financial ratios and metrics
        """
        try:
            # Get company profile
            profile = self.obb.equity.profile(symbol=symbol)

            # Get key metrics
            metrics = self.obb.equity.fundamental.metrics(symbol=symbol)

            # Get financial ratios
            ratios = self.obb.equity.fundamental.ratios(symbol=symbol)

            fundamental_data = {}

            # Extract profile data
            if profile and hasattr(profile, 'results') and profile.results:
                prof_data = profile.results[0]
                fundamental_data['company_name'] = getattr(prof_data, 'name', symbol)
                fundamental_data['sector'] = getattr(prof_data, 'sector', None)
                fundamental_data['industry'] = getattr(prof_data, 'industry', None)
                fundamental_data['market_cap'] = getattr(prof_data, 'market_cap', None)

            # Extract metrics
            if metrics and hasattr(metrics, 'results') and metrics.results:
                metric_data = metrics.results[0]
                fundamental_data['pe_ratio'] = getattr(metric_data, 'pe_ratio', None)
                fundamental_data['forward_pe'] = getattr(metric_data, 'forward_pe', None)
                fundamental_data['peg_ratio'] = getattr(metric_data, 'peg_ratio', None)
                fundamental_data['price_to_book'] = getattr(metric_data, 'price_to_book', None)
                fundamental_data['price_to_sales'] = getattr(metric_data, 'price_to_sales', None)
                fundamental_data['dividend_yield'] = getattr(metric_data, 'dividend_yield', None)

            # Extract ratios
            if ratios and hasattr(ratios, 'results') and ratios.results:
                ratio_data = ratios.results[0]
                fundamental_data['roe'] = getattr(ratio_data, 'return_on_equity', None)
                fundamental_data['roa'] = getattr(ratio_data, 'return_on_assets', None)
                fundamental_data['debt_to_equity'] = getattr(ratio_data, 'debt_to_equity', None)
                fundamental_data['current_ratio'] = getattr(ratio_data, 'current_ratio', None)
                fundamental_data['quick_ratio'] = getattr(ratio_data, 'quick_ratio', None)

            return fundamental_data if fundamental_data else None

        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return None

    def get_financial_statements(
        self,
        symbol: str,
        statement_type: str = "income",
        period: str = "annual",
        limit: int = 5
    ) -> Optional[pd.DataFrame]:
        """
        Get financial statements.

        Args:
            symbol: Stock symbol
            statement_type: Type (income, balance, cash)
            period: Period (annual, quarter)
            limit: Number of periods

        Returns:
            DataFrame with financial data
        """
        try:
            if statement_type == "income":
                data = self.obb.equity.fundamental.income(
                    symbol=symbol,
                    period=period,
                    limit=limit
                )
            elif statement_type == "balance":
                data = self.obb.equity.fundamental.balance(
                    symbol=symbol,
                    period=period,
                    limit=limit
                )
            elif statement_type == "cash":
                data = self.obb.equity.fundamental.cash(
                    symbol=symbol,
                    period=period,
                    limit=limit
                )
            else:
                logger.error(f"Unknown statement type: {statement_type}")
                return None

            if data and hasattr(data, 'to_dataframe'):
                return data.to_dataframe()

            return None

        except Exception as e:
            logger.error(f"Error getting {statement_type} statement for {symbol}: {e}")
            return None

    # ========== News & Sentiment ==========

    def get_news(
        self,
        symbol: str,
        limit: int = 10
    ) -> Optional[List[Dict]]:
        """
        Get recent news for a symbol.

        Args:
            symbol: Stock symbol
            limit: Number of articles

        Returns:
            List of news articles
        """
        try:
            news = self.obb.news.company(
                symbol=symbol,
                limit=limit
            )

            if news and hasattr(news, 'results'):
                articles = []

                for item in news.results:
                    articles.append({
                        'title': getattr(item, 'title', ''),
                        'text': getattr(item, 'text', getattr(item, 'description', '')),
                        'url': getattr(item, 'url', ''),
                        'source': getattr(item, 'source', ''),
                        'published_date': getattr(item, 'date', getattr(item, 'published_date', '')),
                        'symbols': getattr(item, 'symbols', [symbol])
                    })

                return articles

            return []

        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []

    # ========== Screening & Discovery ==========

    def screen_stocks(
        self,
        preset: str = "top_gainers",
        limit: int = 20
    ) -> Optional[List[Dict]]:
        """
        Screen stocks based on criteria.

        Args:
            preset: Preset screen (top_gainers, top_losers, most_active, etc.)
            limit: Number of results

        Returns:
            List of stock data
        """
        try:
            if preset == "top_gainers":
                data = self.obb.equity.discovery.gainers(limit=limit)
            elif preset == "top_losers":
                data = self.obb.equity.discovery.losers(limit=limit)
            elif preset == "most_active":
                data = self.obb.equity.discovery.active(limit=limit)
            else:
                logger.error(f"Unknown preset: {preset}")
                return None

            if data and hasattr(data, 'results'):
                return [
                    {
                        'symbol': item.symbol,
                        'name': getattr(item, 'name', ''),
                        'price': getattr(item, 'price', None),
                        'change': getattr(item, 'change', None),
                        'change_percent': getattr(item, 'change_percent', None),
                        'volume': getattr(item, 'volume', None)
                    }
                    for item in data.results
                ]

            return []

        except Exception as e:
            logger.error(f"Error screening stocks with {preset}: {e}")
            return []

    # ========== Options Data ==========

    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get options chain for a symbol.

        Args:
            symbol: Stock symbol
            expiration: Expiration date (YYYY-MM-DD), None for nearest

        Returns:
            DataFrame with options data
        """
        try:
            chains = self.obb.derivatives.options.chains(
                symbol=symbol,
                date=expiration
            )

            if chains and hasattr(chains, 'to_dataframe'):
                return chains.to_dataframe()

            return None

        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return None

    # ========== Economic Data ==========

    def get_economic_calendar(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get economic calendar events.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of economic events
        """
        try:
            if not start_date:
                start_date = datetime.now().strftime("%Y-%m-%d")

            if not end_date:
                end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

            calendar = self.obb.economy.calendar(
                start_date=start_date,
                end_date=end_date
            )

            if calendar and hasattr(calendar, 'results'):
                return [
                    {
                        'date': getattr(item, 'date', ''),
                        'country': getattr(item, 'country', ''),
                        'event': getattr(item, 'event', ''),
                        'importance': getattr(item, 'importance', ''),
                        'actual': getattr(item, 'actual', None),
                        'forecast': getattr(item, 'forecast', None),
                        'previous': getattr(item, 'previous', None)
                    }
                    for item in calendar.results
                ]

            return []

        except Exception as e:
            logger.error(f"Error getting economic calendar: {e}")
            return []


# Singleton instance
_openbb_provider = None

def get_openbb_provider() -> OpenBBProvider:
    """Get or create OpenBB provider instance"""
    global _openbb_provider

    if _openbb_provider is None:
        _openbb_provider = OpenBBProvider()

    return _openbb_provider
