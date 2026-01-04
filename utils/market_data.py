"""
Market Data Fetcher - yfinance wrapper for market data
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


class MarketDataFetcher:
    """
    Market Data Fetcher using yfinance.

    Provides historical and real-time market data.
    """

    def __init__(self):
        """Initialize Market Data Fetcher"""
        pass

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical market data

        Args:
            symbol: Stock symbol
            period: Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)
            interval: Interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            return df

        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_realtime(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote data

        Args:
            symbol: Stock symbol

        Returns:
            Dict with current price, volume, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'open': info.get('regularMarketOpen', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error fetching realtime data for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Alias for get_realtime"""
        return self.get_realtime(symbol)

    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """
        Get company information

        Args:
            symbol: Stock symbol

        Returns:
            Dict with company info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'description': info.get('longBusinessSummary', ''),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'exchange': info.get('exchange', '')
            }

        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return None

    def calculate_volatility(
        self,
        symbol: str,
        period: str = "1mo",
        annualized: bool = True
    ) -> Optional[float]:
        """
        Calculate historical volatility

        Args:
            symbol: Stock symbol
            period: Period for calculation
            annualized: Annualize the volatility (default True)

        Returns:
            Volatility as float
        """
        try:
            df = self.get_historical(symbol, period=period, interval="1d")

            if df is None or len(df) < 2:
                return None

            # Calculate daily returns
            returns = df['Close'].pct_change().dropna()

            # Calculate volatility (std dev)
            volatility = returns.std()

            # Annualize if requested (assume 252 trading days)
            if annualized:
                volatility = volatility * (252 ** 0.5)

            return float(volatility)

        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return None

    def batch_get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to quote data
        """
        results = {}

        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                results[symbol] = quote

        return results


if __name__ == "__main__":
    fetcher = MarketDataFetcher()

    # Get realtime quote
    quote = fetcher.get_quote("AAPL")
    if quote:
        print(f"Current price of AAPL: ${quote['price']:.2f}")

    # Get historical data
    df = fetcher.get_historical("AAPL", period="1mo")
    if df is not None:
        print(f"\nHistorical data: {len(df)} days")
        print(df.tail())

    # Calculate volatility
    vol = fetcher.calculate_volatility("AAPL", period="3mo")
    if vol:
        print(f"\nAnnualized volatility: {vol:.2%}")
