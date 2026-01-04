"""
Market Regime Detection

Detects market regimes to enable adaptive agent routing:
1. Volatility Regime (low, medium, high)
2. Trend Regime (bull, bear, sideways)
3. News Impact Regime (low, high)
4. Overall Regime Classification

Used by Supervisor to adapt agent selection strategy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Trend regime classification"""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


class NewsImpactRegime(Enum):
    """News impact regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RegimeDetection:
    """Regime detection result"""
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    news_impact_regime: NewsImpactRegime
    overall_regime: str
    confidence: float
    recommended_agents: List[str]
    reasoning: str


class MarketRegimeDetector:
    """
    Detect market regimes from market data
    
    Uses multiple indicators:
    - Volatility (ATR, historical volatility)
    - Trend (moving averages, momentum)
    - News impact (sentiment, volume)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Thresholds for regime classification
        self.volatility_thresholds = {
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75
        }
        
        self.trend_thresholds = {
            'strong_bull': 0.6,
            'bull': 0.2,
            'bear': -0.2,
            'strong_bear': -0.6
        }
        
        self.news_impact_thresholds = {
            'low': 0.3,
            'high': 0.7
        }
    
    def detect_regime(self, market_data: Dict) -> RegimeDetection:
        """
        Detect current market regime
        
        Args:
            market_data: Dictionary with market indicators
        
        Returns:
            RegimeDetection
        """
        # Extract indicators
        volatility = market_data.get('volatility', 0.5)
        trend_strength = market_data.get('trend_strength', 0.0)
        news_impact = market_data.get('news_impact', 0.5)
        
        # Classify regimes
        vol_regime = self._classify_volatility(volatility)
        trend_regime = self._classify_trend(trend_strength)
        news_regime = self._classify_news_impact(news_impact)
        
        # Determine overall regime
        overall_regime = self._determine_overall_regime(
            vol_regime, trend_regime, news_regime
        )
        
        # Recommend agents based on regime
        recommended_agents, reasoning = self._recommend_agents(
            vol_regime, trend_regime, news_regime
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(market_data)
        
        return RegimeDetection(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            news_impact_regime=news_regime,
            overall_regime=overall_regime,
            confidence=confidence,
            recommended_agents=recommended_agents,
            reasoning=reasoning
        )
    
    def _classify_volatility(self, volatility: float) -> VolatilityRegime:
        """Classify volatility regime"""
        if volatility < self.volatility_thresholds['low']:
            return VolatilityRegime.LOW
        elif volatility < self.volatility_thresholds['medium']:
            return VolatilityRegime.MEDIUM
        elif volatility < self.volatility_thresholds['high']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _classify_trend(self, trend_strength: float) -> TrendRegime:
        """Classify trend regime"""
        if trend_strength > self.trend_thresholds['strong_bull']:
            return TrendRegime.STRONG_BULL
        elif trend_strength > self.trend_thresholds['bull']:
            return TrendRegime.BULL
        elif trend_strength > self.trend_thresholds['bear']:
            return TrendRegime.SIDEWAYS
        elif trend_strength > self.trend_thresholds['strong_bear']:
            return TrendRegime.BEAR
        else:
            return TrendRegime.STRONG_BEAR
    
    def _classify_news_impact(self, news_impact: float) -> NewsImpactRegime:
        """Classify news impact regime"""
        if news_impact < self.news_impact_thresholds['low']:
            return NewsImpactRegime.LOW
        elif news_impact < self.news_impact_thresholds['high']:
            return NewsImpactRegime.MEDIUM
        else:
            return NewsImpactRegime.HIGH
    
    def _determine_overall_regime(
        self,
        vol_regime: VolatilityRegime,
        trend_regime: TrendRegime,
        news_regime: NewsImpactRegime
    ) -> str:
        """Determine overall market regime"""
        
        # Crash scenario
        if vol_regime == VolatilityRegime.EXTREME and news_regime == NewsImpactRegime.HIGH:
            return "crash"
        
        # High volatility scenarios
        if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            if news_regime == NewsImpactRegime.HIGH:
                return "news_driven_volatility"
            else:
                return "high_volatility"
        
        # Trending scenarios
        if trend_regime in [TrendRegime.STRONG_BULL, TrendRegime.STRONG_BEAR]:
            if vol_regime == VolatilityRegime.LOW:
                return "strong_trend_low_vol"
            else:
                return "strong_trend_high_vol"
        
        # Bull/Bear markets
        if trend_regime == TrendRegime.BULL:
            return "bull_market"
        elif trend_regime == TrendRegime.BEAR:
            return "bear_market"
        
        # Sideways/Range-bound
        if trend_regime == TrendRegime.SIDEWAYS:
            if vol_regime == VolatilityRegime.LOW:
                return "range_bound_low_vol"
            else:
                return "range_bound_high_vol"
        
        return "neutral"
    
    def _recommend_agents(
        self,
        vol_regime: VolatilityRegime,
        trend_regime: TrendRegime,
        news_regime: NewsImpactRegime
    ) -> Tuple[List[str], str]:
        """
        Recommend which agents to prioritize based on regime
        
        Returns:
            (recommended_agents, reasoning)
        """
        
        # High news impact → prioritize news agent
        if news_regime == NewsImpactRegime.HIGH:
            return (
                ['news', 'fundamental'],
                "High news impact detected. News sentiment is driving market movements."
            )
        
        # Extreme volatility → prioritize news agent
        if vol_regime == VolatilityRegime.EXTREME:
            return (
                ['news', 'fundamental'],
                "Extreme volatility detected. Market is likely driven by news/events."
            )
        
        # Strong trend + low volatility → prioritize technical agent
        if trend_regime in [TrendRegime.STRONG_BULL, TrendRegime.STRONG_BEAR] and \
           vol_regime == VolatilityRegime.LOW:
            return (
                ['technical', 'fundamental'],
                "Strong trend with low volatility. Technical indicators are reliable."
            )
        
        # High volatility + low news → technical may be unreliable
        if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME] and \
           news_regime == NewsImpactRegime.LOW:
            return (
                ['fundamental', 'news'],
                "High volatility without clear news driver. Focus on fundamentals."
            )
        
        # Sideways market → technical + fundamental
        if trend_regime == TrendRegime.SIDEWAYS:
            return (
                ['technical', 'fundamental'],
                "Range-bound market. Technical levels and fundamentals are key."
            )
        
        # Default: all agents
        return (
            ['news', 'technical', 'fundamental'],
            "Balanced market conditions. All agents provide value."
        )
    
    def _calculate_confidence(self, market_data: Dict) -> float:
        """
        Calculate confidence in regime detection
        
        Higher confidence when:
        - Clear signals (extreme values)
        - Consistent indicators
        - Sufficient data
        """
        volatility = market_data.get('volatility', 0.5)
        trend_strength = abs(market_data.get('trend_strength', 0.0))
        news_impact = market_data.get('news_impact', 0.5)
        
        # Confidence increases with extreme values (clear signals)
        vol_confidence = min(volatility / 0.8, 1.0)  # High vol → high confidence
        trend_confidence = min(trend_strength / 0.8, 1.0)  # Strong trend → high confidence
        news_confidence = min(news_impact / 0.8, 1.0)  # High news impact → high confidence
        
        # Average confidence
        confidence = (vol_confidence + trend_confidence + news_confidence) / 3.0
        
        # Boost confidence if multiple indicators agree
        if volatility > 0.7 and news_impact > 0.7:
            confidence = min(confidence * 1.2, 1.0)
        
        if trend_strength > 0.7 and volatility < 0.3:
            confidence = min(confidence * 1.2, 1.0)
        
        return confidence


def detect_regime(market_data: Dict) -> Dict:
    """
    Convenience function for regime detection
    
    Args:
        market_data: Dictionary with market indicators
    
    Returns:
        Dictionary with regime information
    """
    detector = MarketRegimeDetector()
    result = detector.detect_regime(market_data)
    
    return {
        'volatility_regime': result.volatility_regime.value,
        'trend_regime': result.trend_regime.value,
        'news_impact_regime': result.news_impact_regime.value,
        'overall_regime': result.overall_regime,
        'confidence': result.confidence,
        'recommended_agents': result.recommended_agents,
        'reasoning': result.reasoning
    }


class AdaptiveAgentRouter:
    """
    Adaptive agent router that uses regime detection
    
    Integrates with Supervisor to provide regime-aware routing
    """
    
    def __init__(self, supervisor, regime_detector: Optional[MarketRegimeDetector] = None):
        self.supervisor = supervisor
        self.regime_detector = regime_detector or MarketRegimeDetector()
        
        # Track regime history
        self.regime_history = []
        self.max_history = 100
    
    def route(
        self,
        market_data: Dict,
        portfolio_state: Dict,
        use_regime_detection: bool = True
    ) -> Dict:
        """
        Route to agents with regime awareness
        
        Args:
            market_data: Market context
            portfolio_state: Portfolio state
            use_regime_detection: Whether to use regime detection
        
        Returns:
            Routing decision with regime information
        """
        # Detect regime
        if use_regime_detection:
            regime = self.regime_detector.detect_regime(market_data)
            
            # Track regime history
            self.regime_history.append(regime.overall_regime)
            if len(self.regime_history) > self.max_history:
                self.regime_history.pop(0)
            
            # Check for regime change
            regime_changed = self._detect_regime_change()
            
            if regime_changed:
                logger.info(f"Regime change detected: {self.regime_history[-2]} → {regime.overall_regime}")
                # Notify supervisor to increase exploration
                if hasattr(self.supervisor, '_handle_regime_change'):
                    self.supervisor._handle_regime_change()
        else:
            regime = None
            regime_changed = False
        
        # Get supervisor routing
        selected_agents, uncertainties = self.supervisor.select_agents(
            context=market_data,
            explore=True
        )
        
        # Combine with regime recommendations
        if regime and use_regime_detection:
            # Weight supervisor selection with regime recommendations
            recommended = set(regime.recommended_agents)
            selected = set(selected_agents)
            
            # If supervisor selected an agent not recommended by regime,
            # add top regime-recommended agent
            if not selected.intersection(recommended):
                selected_agents.append(regime.recommended_agents[0])
        
        return {
            'active_agents': selected_agents,
            'uncertainties': uncertainties,
            'regime': regime.__dict__ if regime else None,
            'regime_changed': regime_changed
        }
    
    def _detect_regime_change(self) -> bool:
        """Detect if regime has changed recently"""
        if len(self.regime_history) < 2:
            return False
        
        # Check if regime changed in last step
        return self.regime_history[-1] != self.regime_history[-2]


if __name__ == '__main__':
    # Test regime detection
    
    # Test 1: Crash scenario
    crash_data = {
        'volatility': 0.95,
        'trend_strength': -0.8,
        'news_impact': 0.9
    }
    
    result = detect_regime(crash_data)
    print("Crash Scenario:")
    print(f"  Overall Regime: {result['overall_regime']}")
    print(f"  Recommended Agents: {result['recommended_agents']}")
    print(f"  Reasoning: {result['reasoning']}")
    print()
    
    # Test 2: Bull market
    bull_data = {
        'volatility': 0.2,
        'trend_strength': 0.7,
        'news_impact': 0.3
    }
    
    result = detect_regime(bull_data)
    print("Bull Market:")
    print(f"  Overall Regime: {result['overall_regime']}")
    print(f"  Recommended Agents: {result['recommended_agents']}")
    print(f"  Reasoning: {result['reasoning']}")
    print()
    
    # Test 3: High volatility
    high_vol_data = {
        'volatility': 0.85,
        'trend_strength': 0.0,
        'news_impact': 0.5
    }
    
    result = detect_regime(high_vol_data)
    print("High Volatility:")
    print(f"  Overall Regime: {result['overall_regime']}")
    print(f"  Recommended Agents: {result['recommended_agents']}")
    print(f"  Reasoning: {result['reasoning']}")
