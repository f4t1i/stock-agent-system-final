#!/usr/bin/env python3
"""
Supervisor v2 - Contextual Bandit Agent Routing

Purpose:
    Route trading decisions to the best-performing agent based on market context.
    Uses contextual multi-armed bandit algorithms to learn which agent performs
    best in different market regimes.

Algorithm:
    1. Extract market regime features (volatility, trend, sentiment)
    2. Select agent using bandit algorithm (Thompson Sampling, UCB, ε-greedy)
    3. Execute agent's decision
    4. Observe reward (trading outcome)
    5. Update agent selection probabilities

Features:
    - Multiple bandit algorithms (Thompson Sampling, UCB, ε-greedy)
    - Regime-aware context
    - Per-agent performance tracking
    - Symbol-specific and regime-specific routing
    - Exploration-exploitation balance

Usage:
    supervisor = SupervisorV2(config_path="training/rl/rl_config.yaml")

    # Route decision
    agent_name, confidence = supervisor.select_agent(
        symbol="AAPL",
        market_data=market_data,
        regime_features=regime_features
    )

    # Execute agent decision
    signal = agents[agent_name].analyze(market_data)

    # Update with reward
    supervisor.update(
        agent_name=agent_name,
        reward=trade_outcome.reward,
        context=regime_features
    )
"""

import json
import yaml
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger


class BanditAlgorithm(Enum):
    """Bandit algorithm types"""
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class AgentStats:
    """Statistics for an agent"""
    agent_name: str
    num_selections: int = 0
    num_successes: int = 0
    num_failures: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    success_rate: float = 0.0

    # Thompson Sampling parameters
    alpha: float = 1.0  # Beta prior alpha (successes)
    beta: float = 1.0   # Beta prior beta (failures)


@dataclass
class RoutingDecision:
    """Agent routing decision"""
    agent_name: str
    confidence: float
    algorithm: str
    context: Optional[Dict] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SupervisorV2:
    """
    Contextual Bandit Supervisor for Agent Routing

    Learns to route decisions to the best agent based on market context.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        tracking_db: Optional[Path] = None
    ):
        """
        Initialize supervisor

        Args:
            config_path: Path to RL config YAML
            tracking_db: Path to tracking database
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "training" / "rl" / "rl_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.config = config.get("contextual_bandit", {})

        # Parse algorithm
        algorithm_name = self.config.get("algorithm", "thompson_sampling")
        self.algorithm = BanditAlgorithm(algorithm_name)

        # Initialize agent stats
        self.agents: Dict[str, AgentStats] = {}
        for agent_config in self.config.get("agents", []):
            agent_name = agent_config["name"]
            self.agents[agent_name] = AgentStats(
                agent_name=agent_name,
                alpha=self.config.get("prior_alpha", 1.0),
                beta=self.config.get("prior_beta", 1.0)
            )

        # Epsilon-greedy parameters
        self.epsilon = self.config.get("epsilon", 0.1)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)

        # UCB parameters
        self.ucb_c = self.config.get("ucb_c", 2.0)

        # Tracking database
        if tracking_db is None:
            tracking_db = Path("models/rl/supervisor_tracking.db")

        self.tracking_db = Path(tracking_db)
        self.tracking_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        logger.info(f"Supervisor v2 initialized")
        logger.info(f"  Algorithm: {self.algorithm.value}")
        logger.info(f"  Agents: {list(self.agents.keys())}")

    def _init_db(self):
        """Initialize tracking database"""
        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_history (
                routing_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                symbol TEXT,
                regime TEXT,
                confidence REAL,
                reward REAL,
                success BOOLEAN,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                agent_name TEXT PRIMARY KEY,
                num_selections INTEGER,
                num_successes INTEGER,
                num_failures INTEGER,
                total_reward REAL,
                avg_reward REAL,
                success_rate REAL,
                last_updated TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_agent
            ON routing_history(agent_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_regime
            ON routing_history(regime)
        """)

        conn.commit()
        conn.close()

    def select_agent(
        self,
        symbol: str,
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """
        Select agent using bandit algorithm

        Args:
            symbol: Stock symbol
            context: Market context/regime features

        Returns:
            RoutingDecision with agent name and confidence
        """
        if self.algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
            return self._thompson_sampling(symbol, context)
        elif self.algorithm == BanditAlgorithm.UCB:
            return self._ucb(symbol, context)
        elif self.algorithm == BanditAlgorithm.EPSILON_GREEDY:
            return self._epsilon_greedy(symbol, context)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _thompson_sampling(
        self,
        symbol: str,
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """Thompson Sampling agent selection"""
        # Sample from Beta distribution for each agent
        samples = {}
        for agent_name, stats in self.agents.items():
            # Beta distribution with updated parameters
            sample = np.random.beta(stats.alpha, stats.beta)
            samples[agent_name] = sample

        # Select agent with highest sample
        selected_agent = max(samples, key=samples.get)
        confidence = samples[selected_agent]

        return RoutingDecision(
            agent_name=selected_agent,
            confidence=confidence,
            algorithm="thompson_sampling",
            context=context
        )

    def _ucb(
        self,
        symbol: str,
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """Upper Confidence Bound agent selection"""
        total_selections = sum(stats.num_selections for stats in self.agents.values())

        if total_selections == 0:
            # Random selection on first round
            agent_name = np.random.choice(list(self.agents.keys()))
            return RoutingDecision(
                agent_name=agent_name,
                confidence=1.0 / len(self.agents),
                algorithm="ucb",
                context=context
            )

        # Compute UCB for each agent
        ucb_values = {}
        for agent_name, stats in self.agents.items():
            if stats.num_selections == 0:
                # High UCB for unselected agents
                ucb_values[agent_name] = float("inf")
            else:
                # UCB formula: avg_reward + c * sqrt(ln(N) / n_i)
                exploration_term = self.ucb_c * np.sqrt(
                    np.log(total_selections) / stats.num_selections
                )
                ucb_values[agent_name] = stats.avg_reward + exploration_term

        # Select agent with highest UCB
        selected_agent = max(ucb_values, key=ucb_values.get)
        confidence = ucb_values[selected_agent]

        return RoutingDecision(
            agent_name=selected_agent,
            confidence=confidence,
            algorithm="ucb",
            context=context
        )

    def _epsilon_greedy(
        self,
        symbol: str,
        context: Optional[Dict] = None
    ) -> RoutingDecision:
        """Epsilon-greedy agent selection"""
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            # Random exploration
            agent_name = np.random.choice(list(self.agents.keys()))
            confidence = self.epsilon
        else:
            # Exploit best agent
            best_agent = max(
                self.agents.values(),
                key=lambda s: s.avg_reward
            )
            agent_name = best_agent.agent_name
            confidence = 1.0 - self.epsilon

        return RoutingDecision(
            agent_name=agent_name,
            confidence=confidence,
            algorithm="epsilon_greedy",
            context=context
        )

    def update(
        self,
        agent_name: str,
        reward: float,
        symbol: Optional[str] = None,
        context: Optional[Dict] = None
    ):
        """
        Update agent statistics with observed reward

        Args:
            agent_name: Selected agent
            reward: Observed reward (e.g., trade P&L)
            symbol: Stock symbol
            context: Market context
        """
        if agent_name not in self.agents:
            logger.warning(f"Unknown agent: {agent_name}")
            return

        stats = self.agents[agent_name]

        # Update counts
        stats.num_selections += 1

        # Binary success/failure (reward > 0)
        success = reward > 0
        if success:
            stats.num_successes += 1
            stats.alpha += 1  # Thompson Sampling update
        else:
            stats.num_failures += 1
            stats.beta += 1  # Thompson Sampling update

        # Update reward stats
        stats.total_reward += reward
        stats.avg_reward = stats.total_reward / stats.num_selections
        stats.success_rate = stats.num_successes / stats.num_selections

        # Decay epsilon (for epsilon-greedy)
        if self.algorithm == BanditAlgorithm.EPSILON_GREEDY:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

        # Store in database
        self._store_routing_update(
            agent_name=agent_name,
            symbol=symbol,
            reward=reward,
            success=success,
            context=context
        )

        logger.debug(
            f"Updated {agent_name}: "
            f"selections={stats.num_selections}, "
            f"avg_reward={stats.avg_reward:.4f}, "
            f"success_rate={stats.success_rate:.2%}"
        )

    def _store_routing_update(
        self,
        agent_name: str,
        symbol: Optional[str],
        reward: float,
        success: bool,
        context: Optional[Dict]
    ):
        """Store routing update in database"""
        routing_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        regime = context.get("regime", "unknown") if context else "unknown"

        conn = sqlite3.connect(self.tracking_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO routing_history (
                routing_id, agent_name, symbol, regime,
                confidence, reward, success, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            routing_id,
            agent_name,
            symbol,
            regime,
            0.0,  # Confidence not stored for updates
            reward,
            1 if success else 0,
            datetime.now().isoformat()
        ))

        # Update agent performance summary
        stats = self.agents[agent_name]
        cursor.execute("""
            INSERT OR REPLACE INTO agent_performance (
                agent_name, num_selections, num_successes, num_failures,
                total_reward, avg_reward, success_rate, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent_name,
            stats.num_selections,
            stats.num_successes,
            stats.num_failures,
            stats.total_reward,
            stats.avg_reward,
            stats.success_rate,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_agent_stats(self) -> Dict[str, AgentStats]:
        """Get current agent statistics"""
        return self.agents.copy()

    def get_performance_summary(self) -> str:
        """Generate performance summary report"""
        lines = []
        lines.append("=" * 60)
        lines.append("SUPERVISOR V2 - AGENT PERFORMANCE")
        lines.append("=" * 60)
        lines.append("")

        # Sort agents by avg reward
        sorted_agents = sorted(
            self.agents.values(),
            key=lambda s: s.avg_reward,
            reverse=True
        )

        for stats in sorted_agents:
            lines.append(f"{stats.agent_name}:")
            lines.append(f"  Selections: {stats.num_selections}")
            lines.append(f"  Success Rate: {stats.success_rate:.2%}")
            lines.append(f"  Avg Reward: {stats.avg_reward:.4f}")
            lines.append(f"  Total Reward: {stats.total_reward:.4f}")
            if self.algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
                lines.append(f"  Thompson α={stats.alpha:.1f}, β={stats.beta:.1f}")
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"Algorithm: {self.algorithm.value}")
        if self.algorithm == BanditAlgorithm.EPSILON_GREEDY:
            lines.append(f"Epsilon: {self.epsilon:.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """CLI interface for supervisor"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supervisor v2 - Contextual Bandit Agent Routing"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to RL config YAML"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show agent statistics"
    )

    args = parser.parse_args()

    # Initialize supervisor
    supervisor = SupervisorV2(config_path=args.config)

    if args.stats:
        print(supervisor.get_performance_summary())
    else:
        # Demo mode
        print("\nDemo: Agent selection")
        for i in range(5):
            decision = supervisor.select_agent(
                symbol="AAPL",
                context={"regime": "bull_low_vol"}
            )
            print(f"Round {i+1}: {decision.agent_name} (confidence={decision.confidence:.4f})")

            # Simulate reward
            reward = np.random.randn() * 0.5
            supervisor.update(
                agent_name=decision.agent_name,
                reward=reward,
                symbol="AAPL",
                context={"regime": "bull_low_vol"}
            )

        print("\n" + supervisor.get_performance_summary())


if __name__ == "__main__":
    main()
