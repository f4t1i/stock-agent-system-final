"""
Reward Calculator

Calculates rewards for agent decisions based on actual trading outcomes.
Integrates with backtester to get real performance metrics.

Key Features:
1. Multiple reward components (return, risk, consistency)
2. Shaped rewards for RL training
3. Backtesting integration
4. Risk-adjusted metrics (Sharpe, Sortino)
5. Confidence calibration rewards

Based on:
- Modern portfolio theory
- Risk-adjusted return metrics
- RL reward shaping best practices
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class TradingOutcome:
    """Outcome of a trading decision"""
    symbol: str
    recommendation: str  # buy, sell, hold
    entry_price: float
    exit_price: Optional[float] = None
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Actual outcomes
    actual_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    holding_period: Optional[int] = None  # days
    
    # Agent predictions
    predicted_confidence: float = 0.5
    predicted_return: Optional[float] = None


@dataclass
class RewardComponents:
    """Individual reward components"""
    return_reward: float
    risk_reward: float
    confidence_reward: float
    consistency_reward: float
    total_reward: float


class RewardCalculator:
    """
    Reward Calculator
    
    Calculates shaped rewards for agent decisions based on outcomes.
    """
    
    def __init__(
        self,
        return_weight: float = 1.0,
        risk_weight: float = 0.3,
        confidence_weight: float = 0.2,
        consistency_weight: float = 0.1,
        risk_free_rate: float = 0.02  # 2% annual
    ):
        """
        Initialize reward calculator
        
        Args:
            return_weight: Weight for return component
            risk_weight: Weight for risk component
            confidence_weight: Weight for confidence calibration
            consistency_weight: Weight for consistency
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.confidence_weight = confidence_weight
        self.consistency_weight = consistency_weight
        self.risk_free_rate = risk_free_rate
    
    def calculate_reward(
        self,
        outcome: TradingOutcome,
        benchmark_return: Optional[float] = None
    ) -> RewardComponents:
        """
        Calculate reward for a trading outcome
        
        Args:
            outcome: Trading outcome
            benchmark_return: Benchmark return (e.g., S&P 500)
        
        Returns:
            RewardComponents with breakdown
        """
        # 1. Return reward
        return_reward = self._calculate_return_reward(
            outcome,
            benchmark_return
        )
        
        # 2. Risk reward (penalize high risk)
        risk_reward = self._calculate_risk_reward(outcome)
        
        # 3. Confidence calibration reward
        confidence_reward = self._calculate_confidence_reward(outcome)
        
        # 4. Consistency reward (placeholder, needs history)
        consistency_reward = 0.0
        
        # Total reward
        total_reward = (
            self.return_weight * return_reward +
            self.risk_weight * risk_reward +
            self.confidence_weight * confidence_reward +
            self.consistency_weight * consistency_reward
        )
        
        return RewardComponents(
            return_reward=return_reward,
            risk_reward=risk_reward,
            confidence_reward=confidence_reward,
            consistency_reward=consistency_reward,
            total_reward=total_reward
        )
    
    def _calculate_return_reward(
        self,
        outcome: TradingOutcome,
        benchmark_return: Optional[float]
    ) -> float:
        """
        Calculate return-based reward
        
        Reward = actual_return (or alpha if benchmark provided)
        
        Args:
            outcome: Trading outcome
            benchmark_return: Benchmark return
        
        Returns:
            Return reward
        """
        if outcome.actual_return is None:
            return 0.0
        
        # Use alpha (excess return) if benchmark provided
        if benchmark_return is not None:
            return outcome.actual_return - benchmark_return
        else:
            return outcome.actual_return
    
    def _calculate_risk_reward(self, outcome: TradingOutcome) -> float:
        """
        Calculate risk-based reward (penalty)
        
        Penalizes:
        - High position sizes
        - Large drawdowns
        - Missing stop-loss
        
        Args:
            outcome: Trading outcome
        
        Returns:
            Risk reward (usually negative)
        """
        penalty = 0.0
        
        # Penalize large position sizes
        if outcome.position_size > 0.15:  # > 15% of portfolio
            penalty -= (outcome.position_size - 0.15) * 2.0
        
        # Penalize large drawdowns
        if outcome.max_drawdown is not None:
            if outcome.max_drawdown < -0.1:  # > 10% drawdown
                penalty -= abs(outcome.max_drawdown) * 2.0
        
        # Penalize missing stop-loss
        if outcome.recommendation in ['buy', 'sell'] and outcome.stop_loss is None:
            penalty -= 0.1
        
        return penalty
    
    def _calculate_confidence_reward(self, outcome: TradingOutcome) -> float:
        """
        Calculate confidence calibration reward
        
        Rewards well-calibrated confidence:
        - High confidence + correct = positive
        - High confidence + wrong = negative
        - Low confidence + correct = small positive
        - Low confidence + wrong = small negative
        
        Args:
            outcome: Trading outcome
        
        Returns:
            Confidence reward
        """
        if outcome.actual_return is None:
            return 0.0
        
        # Determine if prediction was correct
        correct = (
            (outcome.recommendation == 'buy' and outcome.actual_return > 0) or
            (outcome.recommendation == 'sell' and outcome.actual_return < 0) or
            (outcome.recommendation == 'hold' and abs(outcome.actual_return) < 0.02)
        )
        
        # Reward/penalty based on confidence and correctness
        confidence = outcome.predicted_confidence
        
        if correct:
            # Reward high confidence when correct
            reward = confidence
        else:
            # Penalize high confidence when wrong
            reward = -confidence
        
        return reward
    
    def calculate_batch_rewards(
        self,
        outcomes: List[TradingOutcome],
        benchmark_returns: Optional[List[float]] = None
    ) -> Tuple[List[RewardComponents], Dict]:
        """
        Calculate rewards for a batch of outcomes
        
        Also calculates aggregate metrics (Sharpe, Sortino, etc.)
        
        Args:
            outcomes: List of trading outcomes
            benchmark_returns: List of benchmark returns
        
        Returns:
            (List of RewardComponents, aggregate metrics dict)
        """
        # Calculate individual rewards
        rewards = []
        for i, outcome in enumerate(outcomes):
            benchmark = benchmark_returns[i] if benchmark_returns else None
            reward = self.calculate_reward(outcome, benchmark)
            rewards.append(reward)
        
        # Calculate aggregate metrics
        returns = [o.actual_return for o in outcomes if o.actual_return is not None]
        
        if returns:
            metrics = {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe_ratio': self._calculate_sharpe(returns),
                'sortino_ratio': self._calculate_sortino(returns),
                'max_drawdown': min([o.max_drawdown for o in outcomes if o.max_drawdown is not None], default=0),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'mean_reward': np.mean([r.total_reward for r in rewards])
            }
        else:
            metrics = {}
        
        return rewards, metrics
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio
        
        Sharpe = (mean_return - risk_free_rate) / std_return
        
        Args:
            returns: List of returns
        
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        sharpe = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino(self, returns: List[float]) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility)
        
        Sortino = (mean_return - risk_free_rate) / downside_std
        
        Args:
            returns: List of returns
        
        Returns:
            Sortino ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')  # No downside
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        sortino = (mean_return - self.risk_free_rate / 252) / downside_std * np.sqrt(252)
        
        return sortino
    
    def calculate_consistency_reward(
        self,
        outcomes: List[TradingOutcome],
        window: int = 20
    ) -> float:
        """
        Calculate consistency reward based on recent performance
        
        Rewards consistent performance (low variance in returns)
        
        Args:
            outcomes: List of trading outcomes (recent)
            window: Window size for consistency calculation
        
        Returns:
            Consistency reward
        """
        if len(outcomes) < window:
            return 0.0
        
        # Get recent returns
        recent_returns = [
            o.actual_return for o in outcomes[-window:]
            if o.actual_return is not None
        ]
        
        if len(recent_returns) < window // 2:
            return 0.0
        
        # Calculate coefficient of variation (normalized std)
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if mean_return == 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        
        # Reward low CV (consistent performance)
        # CV < 0.5 is good, CV > 2.0 is bad
        if cv < 0.5:
            reward = 0.5
        elif cv > 2.0:
            reward = -0.5
        else:
            reward = 0.5 - (cv - 0.5) / 1.5 * 1.0
        
        return reward


class BacktestRewardCalculator(RewardCalculator):
    """
    Reward Calculator with Backtesting Integration
    
    Extends RewardCalculator to work with backtester.
    """
    
    def __init__(self, backtester, **kwargs):
        """
        Initialize with backtester
        
        Args:
            backtester: Backtester instance
            **kwargs: RewardCalculator arguments
        """
        super().__init__(**kwargs)
        self.backtester = backtester
    
    def calculate_rewards_from_backtest(
        self,
        decisions: List[Dict],
        start_date: str,
        end_date: str
    ) -> Tuple[List[RewardComponents], Dict]:
        """
        Calculate rewards by running backtest
        
        Args:
            decisions: List of agent decisions
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            (List of RewardComponents, backtest metrics)
        """
        # Run backtest
        backtest_results = self.backtester.run_backtest(
            decisions=decisions,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert backtest results to outcomes
        outcomes = []
        for trade in backtest_results['trades']:
            outcome = TradingOutcome(
                symbol=trade['symbol'],
                recommendation=trade['action'],
                entry_price=trade['entry_price'],
                exit_price=trade['exit_price'],
                position_size=trade['position_size'],
                stop_loss=trade.get('stop_loss'),
                take_profit=trade.get('take_profit'),
                actual_return=trade['return'],
                max_drawdown=trade.get('max_drawdown'),
                holding_period=trade.get('holding_period'),
                predicted_confidence=trade.get('confidence', 0.5)
            )
            outcomes.append(outcome)
        
        # Calculate rewards
        rewards, metrics = self.calculate_batch_rewards(outcomes)
        
        # Add backtest metrics
        metrics.update(backtest_results['metrics'])
        
        return rewards, metrics


if __name__ == '__main__':
    # Test
    calculator = RewardCalculator(
        return_weight=1.0,
        risk_weight=0.3,
        confidence_weight=0.2
    )
    
    # Example outcome
    outcome = TradingOutcome(
        symbol='AAPL',
        recommendation='buy',
        entry_price=150.0,
        exit_price=165.0,
        position_size=0.1,
        stop_loss=145.0,
        take_profit=170.0,
        actual_return=0.10,  # 10% return
        max_drawdown=-0.03,  # 3% max drawdown
        holding_period=30,
        predicted_confidence=0.85
    )
    
    reward = calculator.calculate_reward(outcome)
    
    print(f"Return Reward: {reward.return_reward:.3f}")
    print(f"Risk Reward: {reward.risk_reward:.3f}")
    print(f"Confidence Reward: {reward.confidence_reward:.3f}")
    print(f"Total Reward: {reward.total_reward:.3f}")
