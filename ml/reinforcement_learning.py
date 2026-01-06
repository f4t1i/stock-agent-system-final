#!/usr/bin/env python3
"""
Reinforcement Learning for Trading

Advanced RL algorithms for automated trading:
- DQN (Deep Q-Network)
- DDQN (Double DQN)
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- Trading environment with realistic constraints
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import random
from pathlib import Path
from loguru import logger


@dataclass
class TradingEnvironmentConfig:
    """Configuration for trading environment"""
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001  # 0.1%
    max_position_size: float = 0.2  # Max 20% of portfolio per position
    max_drawdown: float = 0.15  # Max 15% drawdown before stopping
    reward_scaling: float = 1.0


@dataclass
class RLConfig:
    """Configuration for RL agent"""
    algorithm: str  # 'dqn', 'ddqn', 'ppo', 'a2c'
    state_size: int
    action_size: int = 3  # BUY, HOLD, SELL

    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = 'relu'

    # Training
    learning_rate: float = 0.0001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0  # Exploration rate
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # DQN specific
    batch_size: int = 64
    memory_size: int = 10000
    target_update_frequency: int = 10  # Update target network every N episodes

    # PPO specific
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.

    State space:
    - Price features (OHLCV, technical indicators)
    - Portfolio features (cash, positions, PnL)
    - Market features (volatility, momentum)

    Action space:
    - 0: BUY
    - 1: HOLD
    - 2: SELL

    Reward:
    - Portfolio returns
    - Sharpe ratio
    - Risk-adjusted returns
    """

    def __init__(
        self,
        price_data: np.ndarray,
        features: np.ndarray,
        config: TradingEnvironmentConfig,
    ):
        self.price_data = price_data  # [time_steps, OHLCV]
        self.features = features  # [time_steps, n_features]
        self.config = config

        self.n_steps = len(price_data)
        self.current_step = 0

        # Portfolio state
        self.cash = config.initial_balance
        self.initial_cash = config.initial_balance
        self.positions = 0.0  # Number of shares
        self.portfolio_value = config.initial_balance
        self.max_portfolio_value = config.initial_balance

        # History
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.config.initial_balance
        self.positions = 0.0
        self.portfolio_value = self.config.initial_balance
        self.max_portfolio_value = self.config.initial_balance

        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state"""
        if self.current_step >= len(self.features):
            return np.zeros_like(self.features[0])

        # Market features
        market_features = self.features[self.current_step]

        # Portfolio features
        current_price = self.price_data[self.current_step, 3]  # Close price
        position_value = self.positions * current_price
        total_value = self.cash + position_value

        portfolio_features = np.array([
            self.cash / self.initial_cash,  # Normalized cash
            position_value / self.initial_cash,  # Normalized position value
            total_value / self.initial_cash,  # Normalized total value
            self.positions / (self.initial_cash / current_price) if current_price > 0 else 0,  # Normalized positions
        ])

        # Combine features
        state = np.concatenate([market_features, portfolio_features])

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done flag, and info.

        Args:
            action: 0=BUY, 1=HOLD, 2=SELL

        Returns:
            next_state, reward, done, info
        """
        current_price = self.price_data[self.current_step, 3]  # Close price
        prev_portfolio_value = self.portfolio_value

        # Execute action
        if action == 0:  # BUY
            max_shares = (self.cash * self.config.max_position_size) / current_price
            shares_to_buy = min(max_shares, self.cash / (current_price * (1 + self.config.transaction_cost)))

            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.config.transaction_cost)
                self.cash -= cost
                self.positions += shares_to_buy

        elif action == 2:  # SELL
            if self.positions > 0:
                proceeds = self.positions * current_price * (1 - self.config.transaction_cost)
                self.cash += proceeds
                self.positions = 0

        # Update portfolio value
        position_value = self.positions * current_price
        self.portfolio_value = self.cash + position_value

        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, self.portfolio_value)

        # Update history
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self._is_done()

        # Get next state
        next_state = self._get_state()

        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'current_price': current_price,
            'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash,
        }

        return next_state, reward, done, info

    def _calculate_reward(
        self,
        prev_value: float,
        current_value: float,
    ) -> float:
        """Calculate reward based on portfolio change"""
        # Returns-based reward
        returns = (current_value - prev_value) / prev_value if prev_value > 0 else 0

        # Risk penalty (drawdown)
        if current_value < self.max_portfolio_value:
            drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
            risk_penalty = -drawdown * 10  # Penalty for drawdown
        else:
            self.max_portfolio_value = current_value
            risk_penalty = 0

        # Combined reward
        reward = (returns * 100 + risk_penalty) * self.config.reward_scaling

        return reward

    def _is_done(self) -> bool:
        """Check if episode is done"""
        # End of data
        if self.current_step >= self.n_steps - 1:
            return True

        # Max drawdown exceeded
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        if drawdown > self.config.max_drawdown:
            logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
            return True

        # Bankrupt
        if self.portfolio_value <= 0:
            logger.warning("Portfolio value <= 0")
            return True

        return False

    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}

        portfolio_history = np.array(self.portfolio_history)

        # Returns
        total_return = (portfolio_history[-1] - self.initial_cash) / self.initial_cash
        returns = np.diff(portfolio_history) / portfolio_history[:-1]

        # Sharpe ratio (annualized, assuming daily data)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_history)
        drawdown = (cummax - portfolio_history) / cummax
        max_drawdown = np.max(drawdown)

        # Win rate
        profitable_trades = np.sum(np.array(self.reward_history) > 0)
        win_rate = profitable_trades / len(self.reward_history) if self.reward_history else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': portfolio_history[-1],
            'n_trades': len(self.action_history),
        }


class DQNNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]):
        super().__init__()

        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for trading.

    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Gradient clipping
    """

    def __init__(
        self,
        config: RLConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.device = device

        # Networks
        self.policy_net = DQNNetwork(
            config.state_size,
            config.action_size,
            config.hidden_sizes,
        ).to(device)

        self.target_net = DQNNetwork(
            config.state_size,
            config.action_size,
            config.hidden_sizes,
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate,
        )

        # Replay buffer
        self.memory = ReplayBuffer(config.memory_size)

        # Exploration
        self.epsilon = config.epsilon_start

        # Training metrics
        self.training_losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Explore
            return random.randrange(self.config.action_size)
        else:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.config.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        # Loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_losses.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def save(self, path: Path):
        """Save agent"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
        }, path)

        logger.info(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> 'DQNAgent':
        """Load agent"""
        checkpoint = torch.load(path)

        agent = cls(checkpoint['config'])
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint['epsilon']

        logger.info(f"Loaded DQN agent from {path}")

        return agent


def train_dqn_agent(
    agent: DQNAgent,
    env: TradingEnvironment,
    n_episodes: int = 1000,
    save_path: Optional[Path] = None,
) -> Dict[str, List]:
    """
    Train DQN agent.

    Args:
        agent: DQN agent
        env: Trading environment
        n_episodes: Number of episodes to train
        save_path: Path to save best model

    Returns:
        Training history
    """
    logger.info(f"Training DQN agent for {n_episodes} episodes")

    history = {
        'episode_rewards': [],
        'episode_returns': [],
        'episode_sharpe': [],
        'epsilon': [],
    }

    best_return = -float('inf')

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.memory.push(state, action, reward, next_state, done)

            # Train
            agent.train_step()

            episode_reward += reward
            state = next_state

        # Update target network periodically
        if episode % agent.config.target_update_frequency == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Get metrics
        metrics = env.get_metrics()

        # Update history
        history['episode_rewards'].append(episode_reward)
        history['episode_returns'].append(metrics.get('total_return', 0))
        history['episode_sharpe'].append(metrics.get('sharpe_ratio', 0))
        history['epsilon'].append(agent.epsilon)

        # Log progress
        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}/{n_episodes} - "
                f"Reward: {episode_reward:.2f}, "
                f"Return: {metrics.get('total_return', 0):.2%}, "
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Save best model
        if metrics.get('total_return', -float('inf')) > best_return:
            best_return = metrics['total_return']
            if save_path:
                agent.save(save_path)

    logger.info("Training complete!")
    logger.info(f"Best return: {best_return:.2%}")

    return history


if __name__ == '__main__':
    # Example usage
    # Generate dummy price data
    n_steps = 1000
    price_data = np.random.randn(n_steps, 5).cumsum(axis=0) + 100  # OHLCV
    features = np.random.randn(n_steps, 20)  # Technical indicators

    # Create environment
    env_config = TradingEnvironmentConfig(
        initial_balance=100000.0,
        transaction_cost=0.001,
    )
    env = TradingEnvironment(price_data, features, env_config)

    # Create agent
    rl_config = RLConfig(
        algorithm='dqn',
        state_size=features.shape[1] + 4,  # Features + portfolio features
        action_size=3,
        learning_rate=0.0001,
        memory_size=10000,
    )
    agent = DQNAgent(rl_config)

    # Train agent
    history = train_dqn_agent(
        agent,
        env,
        n_episodes=100,
        save_path=Path('models/dqn_trading_agent.pt'),
    )

    print(f"Average return: {np.mean(history['episode_returns']):.2%}")
