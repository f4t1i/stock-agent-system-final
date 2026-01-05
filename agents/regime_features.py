#!/usr/bin/env python3
"""
Regime Features - Market Regime Detection and Feature Extraction

Purpose:
    Extract market regime features (volatility, trend, sentiment) to enable
    context-aware agent routing. The supervisor uses these features to select
    the most appropriate agent for current market conditions.

Features:
    - Volatility calculation (historical, realized, implied)
    - Trend detection (SMA crossovers, trend strength)
    - Market regime classification (bull/bear × high/low vol)
    - Sentiment analysis integration hooks
    - Rolling window statistics

Usage:
    extractor = RegimeFeatureExtractor(config_path="training/rl/rl_config.yaml")

    # Extract features from market data
    features = extractor.extract(
        symbol="AAPL",
        price_data=price_df,  # OHLCV data
        sentiment_data=sentiment_dict  # Optional
    )

    # Get current regime
    regime = features["regime"]  # e.g., "bull_low_vol"

    # Use for routing
    supervisor.select_agent(symbol="AAPL", context=features)
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class TrendDirection(Enum):
    """Trend direction"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class VolatilityLevel(Enum):
    """Volatility level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RegimeFeatures:
    """Market regime features"""
    symbol: str

    # Volatility
    volatility: float  # Annualized volatility
    volatility_level: VolatilityLevel

    # Trend
    trend_direction: TrendDirection
    trend_strength: float  # 0-1 scale
    short_ma: float
    long_ma: float

    # Regime classification
    regime: str  # e.g., "bull_low_vol"

    # Sentiment (optional)
    sentiment_score: Optional[float] = None

    # Additional context
    context: Optional[Dict] = None


class RegimeFeatureExtractor:
    """
    Extract market regime features for contextual agent routing
    """

    def __init__(
        self,
        config_path: Optional[Path] = None
    ):
        """
        Initialize feature extractor

        Args:
            config_path: Path to RL config YAML
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "training" / "rl" / "rl_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.config = config.get("regime_features", {})

        # Volatility parameters
        vol_config = self.config.get("volatility", {})
        self.vol_window = vol_config.get("window", 20)
        self.vol_thresholds = vol_config.get("thresholds", {
            "low": 0.15,
            "medium": 0.30,
            "high": 0.30
        })

        # Trend parameters
        trend_config = self.config.get("trend", {})
        self.short_ma_period = trend_config.get("short_ma", 20)
        self.long_ma_period = trend_config.get("long_ma", 50)
        self.trend_strength_threshold = trend_config.get("trend_strength_threshold", 0.02)

        # Regime definitions
        self.regimes = self.config.get("regimes", [])

        logger.info(f"Regime Feature Extractor initialized")
        logger.info(f"  Volatility window: {self.vol_window} days")
        logger.info(f"  MA periods: {self.short_ma_period}/{self.long_ma_period}")

    def extract(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        sentiment_data: Optional[Dict] = None
    ) -> RegimeFeatures:
        """
        Extract regime features from market data

        Args:
            symbol: Stock symbol
            price_data: OHLCV DataFrame with columns: open, high, low, close, volume
            sentiment_data: Optional sentiment scores

        Returns:
            RegimeFeatures
        """
        # Calculate volatility
        volatility, vol_level = self._calculate_volatility(price_data)

        # Detect trend
        trend_dir, trend_strength, short_ma, long_ma = self._detect_trend(price_data)

        # Classify regime
        regime = self._classify_regime(trend_dir, vol_level)

        # Extract sentiment
        sentiment_score = None
        if sentiment_data:
            sentiment_score = sentiment_data.get("composite_score")

        features = RegimeFeatures(
            symbol=symbol,
            volatility=volatility,
            volatility_level=vol_level,
            trend_direction=trend_dir,
            trend_strength=trend_strength,
            short_ma=short_ma,
            long_ma=long_ma,
            regime=regime,
            sentiment_score=sentiment_score,
            context={
                "price_change_1d": self._price_change(price_data, 1),
                "price_change_5d": self._price_change(price_data, 5),
                "price_change_20d": self._price_change(price_data, 20),
                "volume_trend": self._volume_trend(price_data)
            }
        )

        logger.debug(
            f"{symbol}: regime={regime}, "
            f"vol={volatility:.2%}, "
            f"trend={trend_dir.value}, "
            f"strength={trend_strength:.2f}"
        )

        return features

    def _calculate_volatility(
        self,
        price_data: pd.DataFrame
    ) -> Tuple[float, VolatilityLevel]:
        """
        Calculate annualized volatility

        Args:
            price_data: OHLCV DataFrame

        Returns:
            (volatility, volatility_level)
        """
        # Calculate daily returns
        returns = price_data["close"].pct_change().dropna()

        # Rolling volatility
        vol = returns.tail(self.vol_window).std() * np.sqrt(252)  # Annualized

        # Classify volatility level
        if vol < self.vol_thresholds["low"]:
            vol_level = VolatilityLevel.LOW
        elif vol < self.vol_thresholds["medium"]:
            vol_level = VolatilityLevel.MEDIUM
        else:
            vol_level = VolatilityLevel.HIGH

        return vol, vol_level

    def _detect_trend(
        self,
        price_data: pd.DataFrame
    ) -> Tuple[TrendDirection, float, float, float]:
        """
        Detect trend using moving averages

        Args:
            price_data: OHLCV DataFrame

        Returns:
            (trend_direction, trend_strength, short_ma, long_ma)
        """
        # Calculate moving averages
        short_ma = price_data["close"].rolling(window=self.short_ma_period).mean().iloc[-1]
        long_ma = price_data["close"].rolling(window=self.long_ma_period).mean().iloc[-1]

        # Percent difference
        ma_diff = (short_ma - long_ma) / long_ma

        # Trend direction
        if ma_diff > self.trend_strength_threshold:
            trend_dir = TrendDirection.UP
        elif ma_diff < -self.trend_strength_threshold:
            trend_dir = TrendDirection.DOWN
        else:
            trend_dir = TrendDirection.SIDEWAYS

        # Trend strength (normalized)
        trend_strength = min(abs(ma_diff) / 0.10, 1.0)  # Cap at 10% = strength 1.0

        return trend_dir, trend_strength, short_ma, long_ma

    def _classify_regime(
        self,
        trend_dir: TrendDirection,
        vol_level: VolatilityLevel
    ) -> str:
        """
        Classify market regime

        Args:
            trend_dir: Trend direction
            vol_level: Volatility level

        Returns:
            Regime name (e.g., "bull_low_vol")
        """
        # Map trend + volatility to regime
        trend_str = trend_dir.value
        vol_str = vol_level.value

        # Simple classification
        if trend_dir == TrendDirection.UP:
            if vol_level == VolatilityLevel.LOW:
                return "bull_low_vol"
            else:
                return "bull_high_vol"
        elif trend_dir == TrendDirection.DOWN:
            if vol_level == VolatilityLevel.LOW:
                return "bear_low_vol"
            else:
                return "bear_high_vol"
        else:  # SIDEWAYS
            if vol_level == VolatilityLevel.LOW:
                return "sideways_low_vol"
            else:
                return "sideways_high_vol"

    def _price_change(
        self,
        price_data: pd.DataFrame,
        periods: int
    ) -> float:
        """Calculate price change over N periods"""
        if len(price_data) < periods + 1:
            return 0.0

        current_price = price_data["close"].iloc[-1]
        past_price = price_data["close"].iloc[-periods-1]

        return (current_price - past_price) / past_price

    def _volume_trend(
        self,
        price_data: pd.DataFrame,
        window: int = 20
    ) -> str:
        """
        Detect volume trend

        Args:
            price_data: OHLCV DataFrame
            window: Lookback window

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(price_data) < window * 2:
            return "stable"

        # Recent vs historical volume
        recent_vol = price_data["volume"].tail(window).mean()
        historical_vol = price_data["volume"].tail(window * 2).head(window).mean()

        if historical_vol == 0:
            return "stable"

        change = (recent_vol - historical_vol) / historical_vol

        if change > 0.20:
            return "increasing"
        elif change < -0.20:
            return "decreasing"
        else:
            return "stable"


def main():
    """Demo and testing"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regime Features - Market Regime Detection"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to RL config YAML"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic data"
    )

    args = parser.parse_args()

    if args.demo:
        # Generate synthetic price data
        print("\nGenerating synthetic price data...")
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Bull market with increasing volatility
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02 + 0.001))

        price_data = pd.DataFrame({
            "date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1_000_000, 10_000_000, size=len(dates))
        })

        # Initialize extractor
        extractor = RegimeFeatureExtractor(config_path=args.config)

        # Extract features for different time periods
        print("\nExtracting regime features...\n")

        test_periods = [
            ("Q1 2023", price_data.iloc[:90]),
            ("Q2 2023", price_data.iloc[90:180]),
            ("Q3 2023", price_data.iloc[180:270]),
            ("Q4 2023", price_data.iloc[270:])
        ]

        for period_name, period_data in test_periods:
            if len(period_data) < 50:
                continue

            features = extractor.extract(
                symbol="DEMO",
                price_data=period_data
            )

            print(f"{period_name}:")
            print(f"  Regime: {features.regime}")
            print(f"  Volatility: {features.volatility:.2%} ({features.volatility_level.value})")
            print(f"  Trend: {features.trend_direction.value} (strength={features.trend_strength:.2f})")
            print(f"  MA(20/50): {features.short_ma:.2f} / {features.long_ma:.2f}")
            if features.context:
                print(f"  Price changes: 1d={features.context['price_change_1d']:.2%}, "
                      f"5d={features.context['price_change_5d']:.2%}, "
                      f"20d={features.context['price_change_20d']:.2%}")
            print()

    else:
        print("Use --demo to run demonstration")


if __name__ == "__main__":
    main()
