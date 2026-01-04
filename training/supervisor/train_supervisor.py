"""
Supervisor Training - Contextual Bandit (Neural-UCB) Training

Trains the supervisor agent using contextual bandit approach with:
- Neural network for context embedding
- UCB exploration strategy
- Online learning from feedback
- Replay buffer management
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger
import wandb

from agents.supervisor.supervisor_agent import SupervisorAgent
from utils.config_loader import load_config
from utils.logging_setup import setup_logging


class ReplayBuffer:
    """
    Replay buffer for storing supervisor decisions and rewards.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: Maximum buffer size
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(
        self,
        context: Dict,
        action: int,
        reward: float,
        next_context: Dict
    ):
        """Add experience to buffer"""
        self.buffer.append({
            'context': context,
            'action': action,
            'reward': reward,
            'next_context': next_context,
            'timestamp': datetime.now().isoformat()
        })
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class SupervisorTrainer:
    """
    Trainer for Supervisor Agent using Contextual Bandit approach.
    
    Uses Neural-UCB algorithm:
    1. Neural network encodes context
    2. UCB exploration for action selection
    3. Online learning from rewards
    4. Replay buffer for stability
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to config file
        """
        self.config = load_config(config_path)
        
        # Initialize wandb
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'stock-agent-system'),
                name=f"supervisor-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Initialize supervisor
        self.supervisor = SupervisorAgent(
            model_path=self.config['model']['model_path'],
            config=self.config['model']
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=self.config['training'].get('buffer_size', 10000)
        )
        
        # Training metrics
        self.metrics = {
            'total_steps': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'exploration_rate': self.config['model'].get('exploration_factor', 0.5)
        }
        
        logger.info("Supervisor Trainer initialized")
    
    def train_online(
        self,
        num_episodes: int = 1000,
        symbols: List[str] = None,
        save_interval: int = 100
    ):
        """
        Online training loop.
        
        Args:
            num_episodes: Number of training episodes
            symbols: List of symbols to train on
            save_interval: Save checkpoint every N episodes
        """
        logger.info(f"Starting online training for {num_episodes} episodes")
        
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        for episode in range(num_episodes):
            # Sample random symbol
            symbol = np.random.choice(symbols)
            
            # Create mock context (in production, use real market data)
            context = self._create_mock_context(symbol)
            
            # Get supervisor decision
            routing_decision = self.supervisor.route(context)
            action = self._decision_to_action(routing_decision)
            
            # Simulate environment step and get reward
            reward = self._simulate_step(symbol, routing_decision)
            
            # Add to replay buffer
            next_context = self._create_mock_context(symbol)
            self.replay_buffer.add(context, action, reward, next_context)
            
            # Update metrics
            self.metrics['total_steps'] += 1
            self.metrics['total_reward'] += reward
            self.metrics['avg_reward'] = self.metrics['total_reward'] / self.metrics['total_steps']
            
            # Train from replay buffer
            if len(self.replay_buffer) >= self.config['training'].get('batch_size', 32):
                loss = self._train_step()
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'episode': episode,
                        'reward': reward,
                        'avg_reward': self.metrics['avg_reward'],
                        'loss': loss,
                        'buffer_size': len(self.replay_buffer)
                    })
            
            # Log progress
            if (episode + 1) % 100 == 0:
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {self.metrics['avg_reward']:.3f} | "
                    f"Buffer: {len(self.replay_buffer)}"
                )
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)
        
        # Save final model
        self._save_final_model()
        
        logger.info("Training complete!")
        
        if self.config.get('use_wandb', True):
            wandb.finish()
    
    def _train_step(self) -> float:
        """
        Single training step using replay buffer.
        
        Returns:
            Training loss
        """
        batch_size = self.config['training'].get('batch_size', 32)
        batch = self.replay_buffer.sample(batch_size)
        
        # In a full implementation, this would:
        # 1. Extract context features
        # 2. Compute predicted rewards for each action
        # 3. Update neural network weights
        # 4. Update UCB parameters
        
        # Placeholder loss calculation
        loss = 0.0
        
        # Update supervisor's internal state
        for experience in batch:
            self.supervisor.update(
                experience['context'],
                experience['action'],
                experience['reward']
            )
            loss += abs(experience['reward'])
        
        return loss / len(batch)
    
    def _create_mock_context(self, symbol: str) -> Dict:
        """
        Create mock context for training.
        
        In production, this would fetch real market data.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Context dictionary
        """
        return {
            'symbol': symbol,
            'market_data': {
                'price': np.random.uniform(100, 300),
                'volume': np.random.uniform(1e6, 1e8),
                'volatility': np.random.uniform(0.1, 0.5)
            },
            'portfolio_state': {
                'cash': 100000,
                'positions': {},
                'total_value': 100000
            }
        }
    
    def _decision_to_action(self, routing_decision: Dict) -> int:
        """
        Convert routing decision to action index.
        
        Args:
            routing_decision: Supervisor routing decision
        
        Returns:
            Action index (0-6 for 7 strategies)
        """
        strategy = routing_decision.get('strategy', 'all_agents')
        
        strategy_map = {
            'news_only': 0,
            'technical_only': 1,
            'fundamental_only': 2,
            'news_technical': 3,
            'technical_fundamental': 4,
            'news_fundamental': 5,
            'all_agents': 6
        }
        
        return strategy_map.get(strategy, 6)
    
    def _simulate_step(self, symbol: str, routing_decision: Dict) -> float:
        """
        Simulate environment step and compute reward.
        
        In production, this would:
        1. Execute the routing decision
        2. Run selected agents
        3. Make trading decision
        4. Observe actual outcome
        5. Compute reward based on performance
        
        Args:
            symbol: Stock symbol
            routing_decision: Routing decision
        
        Returns:
            Reward value
        """
        # Mock reward based on strategy
        # In production, use actual trading performance
        
        strategy = routing_decision.get('strategy', 'all_agents')
        
        # Simulate different strategies having different success rates
        base_reward = np.random.normal(0.5, 0.2)
        
        # Bonus for using all agents (more information)
        if strategy == 'all_agents':
            base_reward += 0.1
        
        # Clip to [0, 1]
        reward = np.clip(base_reward, 0.0, 1.0)
        
        return reward
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config['training']['output_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        
        # Save supervisor state
        self.supervisor.save(str(checkpoint_path))
        
        # Save training metrics
        metrics_path = checkpoint_dir / f'metrics_episode_{episode}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final trained model"""
        output_dir = Path(self.config['training']['output_dir']) / 'final'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save supervisor
        self.supervisor.save(str(output_dir / 'supervisor_model.pt'))
        
        # Save final metrics
        with open(output_dir / 'final_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Final model saved: {output_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Supervisor Agent")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/supervisor/neural_ucb.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=None,
        help='List of symbols to train on'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Train
    trainer = SupervisorTrainer(args.config)
    trainer.train_online(
        num_episodes=args.episodes,
        symbols=args.symbols
    )


if __name__ == '__main__':
    main()
