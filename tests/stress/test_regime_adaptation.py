"""
Stress Tests for Non-Stationary Market Environments

Tests Supervisor's ability to adapt to regime changes:
1. Bull Market → Crash (volatility spike)
2. Low Volatility → High Volatility
3. Trending Market → Range-Bound Market
4. News-Driven → Technical-Driven

Key Question: Can the Supervisor quickly detect regime changes and
re-prioritize agents accordingly?
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from agents.supervisor.supervisor_agent import SupervisorAgent
from utils.config_loader import load_config


@dataclass
class MarketRegime:
    """Market regime definition"""
    name: str
    volatility: float  # 0-1 (0=low, 1=high)
    trend_strength: float  # -1 to 1 (-1=strong down, 0=sideways, 1=strong up)
    news_impact: float  # 0-1 (0=low, 1=high)
    technical_reliability: float  # 0-1 (0=unreliable, 1=reliable)
    optimal_agent: str  # Which agent should perform best


# Define market regimes
BULL_MARKET = MarketRegime(
    name="Bull Market",
    volatility=0.2,
    trend_strength=0.8,
    news_impact=0.3,
    technical_reliability=0.9,
    optimal_agent="technical"
)

CRASH = MarketRegime(
    name="Crash",
    volatility=0.95,
    trend_strength=-0.9,
    news_impact=0.95,
    technical_reliability=0.2,
    optimal_agent="news"
)

HIGH_VOLATILITY = MarketRegime(
    name="High Volatility",
    volatility=0.85,
    trend_strength=0.0,
    news_impact=0.7,
    technical_reliability=0.4,
    optimal_agent="news"
)

LOW_VOLATILITY = MarketRegime(
    name="Low Volatility",
    volatility=0.15,
    trend_strength=0.3,
    news_impact=0.2,
    technical_reliability=0.85,
    optimal_agent="technical"
)

RANGE_BOUND = MarketRegime(
    name="Range Bound",
    volatility=0.3,
    trend_strength=0.0,
    news_impact=0.4,
    technical_reliability=0.7,
    optimal_agent="technical"
)

TRENDING = MarketRegime(
    name="Trending",
    volatility=0.4,
    trend_strength=0.7,
    news_impact=0.3,
    technical_reliability=0.85,
    optimal_agent="technical"
)

NEWS_DRIVEN = MarketRegime(
    name="News Driven",
    volatility=0.6,
    trend_strength=0.2,
    news_impact=0.9,
    technical_reliability=0.5,
    optimal_agent="news"
)


class MarketSimulator:
    """Simulate market data for different regimes"""
    
    def __init__(self, regime: MarketRegime):
        self.regime = regime
    
    def generate_market_data(self, symbol: str = "AAPL") -> Dict[str, Any]:
        """Generate synthetic market data matching regime characteristics"""
        
        # Base price
        base_price = 150.0
        
        # Generate price based on regime
        volatility_component = np.random.normal(0, self.regime.volatility * 10)
        trend_component = self.regime.trend_strength * 5
        price = base_price + trend_component + volatility_component
        
        # Generate technical indicators
        rsi = self._generate_rsi()
        macd = self._generate_macd()
        
        # Generate news sentiment
        news_sentiment = self._generate_news_sentiment()
        
        return {
            'symbol': symbol,
            'price': price,
            'volatility': self.regime.volatility,
            'rsi': rsi,
            'macd': macd,
            'news_sentiment': news_sentiment,
            'regime': self.regime.name,
            'technical_reliability': self.regime.technical_reliability,
            'news_impact': self.regime.news_impact
        }
    
    def _generate_rsi(self) -> float:
        """Generate RSI based on regime"""
        if self.regime.trend_strength > 0.5:
            # Bull market: RSI tends to be higher
            return np.clip(np.random.normal(65, 10), 0, 100)
        elif self.regime.trend_strength < -0.5:
            # Bear market: RSI tends to be lower
            return np.clip(np.random.normal(35, 10), 0, 100)
        else:
            # Neutral: RSI around 50
            return np.clip(np.random.normal(50, 15), 0, 100)
    
    def _generate_macd(self) -> float:
        """Generate MACD based on regime"""
        # MACD follows trend strength
        return self.regime.trend_strength * 2 + np.random.normal(0, 0.5)
    
    def _generate_news_sentiment(self) -> float:
        """Generate news sentiment based on regime"""
        # News sentiment correlates with news impact
        base_sentiment = self.regime.trend_strength
        noise = np.random.normal(0, 0.3)
        sentiment = base_sentiment + noise * (1 - self.regime.news_impact)
        return np.clip(sentiment, -2, 2)
    
    def generate_agent_performance(self, agent_type: str) -> float:
        """
        Generate expected agent performance in this regime
        
        Returns reward (0-1)
        """
        if agent_type == 'news':
            # News agent performs well when news impact is high
            base_performance = self.regime.news_impact
            noise = np.random.normal(0, 0.1)
            return np.clip(base_performance + noise, 0, 1)
        
        elif agent_type == 'technical':
            # Technical agent performs well when technical indicators are reliable
            base_performance = self.regime.technical_reliability
            noise = np.random.normal(0, 0.1)
            return np.clip(base_performance + noise, 0, 1)
        
        elif agent_type == 'fundamental':
            # Fundamental agent is more stable across regimes
            base_performance = 0.6
            noise = np.random.normal(0, 0.15)
            return np.clip(base_performance + noise, 0, 1)
        
        else:
            return 0.5


class TestRegimeAdaptation:
    """Test Supervisor adaptation to regime changes"""
    
    @pytest.fixture
    def supervisor(self):
        """Create Supervisor instance"""
        config = {
            'supervisor': {
                'model': 'neural_ucb',
                'exploration_factor': 2.0,
                'learning_rate': 0.01,
                'context_dim': 10
            }
        }
        return SupervisorAgent(config=config)
    
    # ============================================================
    # 1. SINGLE REGIME TESTS
    # ============================================================
    
    def test_bull_market_routing(self, supervisor):
        """Test: In bull market, technical agent should be preferred"""
        simulator = MarketSimulator(BULL_MARKET)
        
        # Run multiple iterations
        technical_selections = 0
        iterations = 20
        
        for _ in range(iterations):
            market_data = simulator.generate_market_data()
            
            # Simulate agent performances
            agent_performances = {
                'news': simulator.generate_agent_performance('news'),
                'technical': simulator.generate_agent_performance('technical'),
                'fundamental': simulator.generate_agent_performance('fundamental')
            }
            
            # Get routing decision
            routing = supervisor.route(
                market_data=market_data,
                portfolio_state={'cash': 10000}
            )
            
            # Update supervisor with feedback
            for agent, perf in agent_performances.items():
                if agent in routing['active_agents']:
                    supervisor.update(
                        context=market_data,
                        agent=agent,
                        reward=perf
                    )
            
            # Count technical selections
            if 'technical' in routing['active_agents']:
                technical_selections += 1
        
        # In bull market, technical should be selected frequently (>60%)
        technical_rate = technical_selections / iterations
        assert technical_rate > 0.6, f"Technical agent selected {technical_rate:.2%} in bull market (should be >60%)"
    
    def test_crash_routing(self, supervisor):
        """Test: In crash, news agent should be preferred"""
        simulator = MarketSimulator(CRASH)
        
        news_selections = 0
        iterations = 20
        
        for _ in range(iterations):
            market_data = simulator.generate_market_data()
            
            agent_performances = {
                'news': simulator.generate_agent_performance('news'),
                'technical': simulator.generate_agent_performance('technical'),
                'fundamental': simulator.generate_agent_performance('fundamental')
            }
            
            routing = supervisor.route(
                market_data=market_data,
                portfolio_state={'cash': 10000}
            )
            
            for agent, perf in agent_performances.items():
                if agent in routing['active_agents']:
                    supervisor.update(
                        context=market_data,
                        agent=agent,
                        reward=perf
                    )
            
            if 'news' in routing['active_agents']:
                news_selections += 1
        
        news_rate = news_selections / iterations
        assert news_rate > 0.6, f"News agent selected {news_rate:.2%} in crash (should be >60%)"
    
    # ============================================================
    # 2. REGIME CHANGE TESTS (CRITICAL!)
    # ============================================================
    
    def test_bull_to_crash_adaptation(self, supervisor):
        """
        Test: Supervisor adapts when market suddenly crashes
        
        Scenario:
        - Phase 1: Bull market (20 iterations) → Technical agent performs well
        - Phase 2: Sudden crash (20 iterations) → News agent should take over
        
        Success Criteria:
        - Phase 1: Technical selected >60%
        - Phase 2: News selected >60% (after adaptation period)
        """
        # Phase 1: Bull Market
        bull_simulator = MarketSimulator(BULL_MARKET)
        phase1_technical_rate = self._run_regime_phase(
            supervisor, bull_simulator, iterations=20, target_agent='technical'
        )
        
        assert phase1_technical_rate > 0.6, f"Phase 1: Technical {phase1_technical_rate:.2%} (should be >60%)"
        
        # Phase 2: Sudden Crash
        crash_simulator = MarketSimulator(CRASH)
        phase2_news_rate = self._run_regime_phase(
            supervisor, crash_simulator, iterations=20, target_agent='news'
        )
        
        assert phase2_news_rate > 0.6, f"Phase 2: News {phase2_news_rate:.2%} after crash (should be >60%)"
    
    def test_low_to_high_volatility_adaptation(self, supervisor):
        """Test: Supervisor adapts to volatility spike"""
        # Phase 1: Low Volatility
        low_vol_simulator = MarketSimulator(LOW_VOLATILITY)
        phase1_technical_rate = self._run_regime_phase(
            supervisor, low_vol_simulator, iterations=15, target_agent='technical'
        )
        
        # Phase 2: High Volatility
        high_vol_simulator = MarketSimulator(HIGH_VOLATILITY)
        phase2_news_rate = self._run_regime_phase(
            supervisor, high_vol_simulator, iterations=15, target_agent='news'
        )
        
        # Check adaptation
        assert phase1_technical_rate > 0.5, f"Low vol: Technical {phase1_technical_rate:.2%}"
        assert phase2_news_rate > 0.5, f"High vol: News {phase2_news_rate:.2%}"
    
    def test_trending_to_range_bound_adaptation(self, supervisor):
        """Test: Supervisor adapts from trending to range-bound market"""
        # Phase 1: Trending Market
        trending_simulator = MarketSimulator(TRENDING)
        phase1_technical_rate = self._run_regime_phase(
            supervisor, trending_simulator, iterations=15, target_agent='technical'
        )
        
        # Phase 2: Range Bound
        range_simulator = MarketSimulator(RANGE_BOUND)
        phase2_technical_rate = self._run_regime_phase(
            supervisor, range_simulator, iterations=15, target_agent='technical'
        )
        
        # Both should prefer technical, but range-bound might be less confident
        assert phase1_technical_rate > 0.6, f"Trending: Technical {phase1_technical_rate:.2%}"
        assert phase2_technical_rate > 0.5, f"Range-bound: Technical {phase2_technical_rate:.2%}"
    
    # ============================================================
    # 3. ADAPTATION SPEED TESTS
    # ============================================================
    
    def test_adaptation_speed(self, supervisor):
        """
        Test: How quickly does Supervisor adapt after regime change?
        
        Measures number of iterations needed to switch to optimal agent
        """
        # Phase 1: Bull Market (establish baseline)
        bull_simulator = MarketSimulator(BULL_MARKET)
        self._run_regime_phase(supervisor, bull_simulator, iterations=20, target_agent='technical')
        
        # Phase 2: Sudden Crash - measure adaptation speed
        crash_simulator = MarketSimulator(CRASH)
        
        news_selections = []
        for i in range(20):
            market_data = crash_simulator.generate_market_data()
            
            agent_performances = {
                'news': crash_simulator.generate_agent_performance('news'),
                'technical': crash_simulator.generate_agent_performance('technical'),
                'fundamental': crash_simulator.generate_agent_performance('fundamental')
            }
            
            routing = supervisor.route(
                market_data=market_data,
                portfolio_state={'cash': 10000}
            )
            
            for agent, perf in agent_performances.items():
                if agent in routing['active_agents']:
                    supervisor.update(
                        context=market_data,
                        agent=agent,
                        reward=perf
                    )
            
            news_selected = 'news' in routing['active_agents']
            news_selections.append(news_selected)
        
        # Calculate adaptation point (when news selection rate exceeds 50%)
        adaptation_point = None
        window_size = 5
        for i in range(window_size, len(news_selections)):
            window = news_selections[i-window_size:i]
            if sum(window) / window_size > 0.5:
                adaptation_point = i
                break
        
        # Should adapt within 10 iterations
        assert adaptation_point is not None, "Supervisor failed to adapt to regime change"
        assert adaptation_point <= 10, f"Adaptation took {adaptation_point} iterations (should be ≤10)"
    
    # ============================================================
    # 4. UNCERTAINTY QUANTIFICATION TESTS
    # ============================================================
    
    def test_uncertainty_increases_after_regime_change(self, supervisor):
        """
        Test: Uncertainty should spike after regime change
        
        This triggers more exploration
        """
        # Phase 1: Bull Market (low uncertainty)
        bull_simulator = MarketSimulator(BULL_MARKET)
        self._run_regime_phase(supervisor, bull_simulator, iterations=20, target_agent='technical')
        
        # Get uncertainty before regime change
        market_data = bull_simulator.generate_market_data()
        routing_before = supervisor.route(market_data=market_data, portfolio_state={'cash': 10000})
        uncertainty_before = routing_before.get('uncertainty', {})
        
        # Phase 2: Sudden Crash
        crash_simulator = MarketSimulator(CRASH)
        market_data_crash = crash_simulator.generate_market_data()
        
        # First routing after crash should have higher uncertainty
        routing_after = supervisor.route(market_data=market_data_crash, portfolio_state={'cash': 10000})
        uncertainty_after = routing_after.get('uncertainty', {})
        
        # Uncertainty should increase (if implemented)
        # This is a check for Neural-UCB uncertainty quantification
        if uncertainty_before and uncertainty_after:
            avg_uncertainty_before = np.mean(list(uncertainty_before.values()))
            avg_uncertainty_after = np.mean(list(uncertainty_after.values()))
            
            # Uncertainty should increase after regime change
            assert avg_uncertainty_after >= avg_uncertainty_before, \
                f"Uncertainty should increase after regime change: {avg_uncertainty_before:.3f} → {avg_uncertainty_after:.3f}"
    
    # ============================================================
    # 5. EXPLORATION VS EXPLOITATION TESTS
    # ============================================================
    
    def test_exploration_in_new_regime(self, supervisor):
        """
        Test: Supervisor should explore more in new/uncertain regimes
        
        Measures diversity of agent selections
        """
        # Phase 1: Stable regime (should exploit)
        bull_simulator = MarketSimulator(BULL_MARKET)
        self._run_regime_phase(supervisor, bull_simulator, iterations=20, target_agent='technical')
        
        stable_selections = self._measure_agent_diversity(supervisor, bull_simulator, iterations=10)
        
        # Phase 2: New regime (should explore)
        crash_simulator = MarketSimulator(CRASH)
        new_selections = self._measure_agent_diversity(crash_simulator, crash_simulator, iterations=10)
        
        # Diversity should be higher in new regime
        stable_diversity = len(set(stable_selections))
        new_diversity = len(set(new_selections))
        
        # In new regime, should try more different agents
        assert new_diversity >= stable_diversity, \
            f"Should explore more in new regime: {stable_diversity} vs {new_diversity} agents tried"
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _run_regime_phase(
        self,
        supervisor,
        simulator: MarketSimulator,
        iterations: int,
        target_agent: str
    ) -> float:
        """
        Run a regime phase and return selection rate of target agent
        """
        target_selections = 0
        
        for _ in range(iterations):
            market_data = simulator.generate_market_data()
            
            agent_performances = {
                'news': simulator.generate_agent_performance('news'),
                'technical': simulator.generate_agent_performance('technical'),
                'fundamental': simulator.generate_agent_performance('fundamental')
            }
            
            routing = supervisor.route(
                market_data=market_data,
                portfolio_state={'cash': 10000}
            )
            
            for agent, perf in agent_performances.items():
                if agent in routing['active_agents']:
                    supervisor.update(
                        context=market_data,
                        agent=agent,
                        reward=perf
                    )
            
            if target_agent in routing['active_agents']:
                target_selections += 1
        
        return target_selections / iterations
    
    def _measure_agent_diversity(
        self,
        supervisor,
        simulator: MarketSimulator,
        iterations: int
    ) -> List[str]:
        """Measure which agents are selected"""
        selections = []
        
        for _ in range(iterations):
            market_data = simulator.generate_market_data()
            routing = supervisor.route(
                market_data=market_data,
                portfolio_state={'cash': 10000}
            )
            
            for agent in routing['active_agents']:
                selections.append(agent)
        
        return selections


class TestRegimeDetection:
    """Test regime detection capabilities"""
    
    def test_detect_volatility_regime(self):
        """Test: Can system detect volatility regime?"""
        from utils.market_regime_detector import detect_regime
        
        # Low volatility data
        low_vol_data = {
            'volatility': 0.15,
            'trend_strength': 0.3,
            'news_impact': 0.2
        }
        
        regime_low = detect_regime(low_vol_data)
        assert regime_low['volatility_regime'] == 'low', "Should detect low volatility"
        
        # High volatility data
        high_vol_data = {
            'volatility': 0.85,
            'trend_strength': 0.0,
            'news_impact': 0.7
        }
        
        regime_high = detect_regime(high_vol_data)
        assert regime_high['volatility_regime'] == 'high', "Should detect high volatility"
    
    def test_detect_trend_regime(self):
        """Test: Can system detect trend regime?"""
        from utils.market_regime_detector import detect_regime
        
        # Trending data
        trending_data = {
            'volatility': 0.4,
            'trend_strength': 0.7,
            'news_impact': 0.3
        }
        
        regime = detect_regime(trending_data)
        assert regime['trend_regime'] in ['trending', 'bull'], "Should detect trending market"
        
        # Range-bound data
        range_data = {
            'volatility': 0.3,
            'trend_strength': 0.0,
            'news_impact': 0.4
        }
        
        regime = detect_regime(range_data)
        assert regime['trend_regime'] in ['range_bound', 'sideways'], "Should detect range-bound market"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
