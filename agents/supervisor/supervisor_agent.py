"""
Supervisor Agent - Routing via Contextual Bandits (NeuralUCB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

from ..base_agent import BaseAgent


class NeuralUCBNetwork(nn.Module):
    """Neural network für UCB-basiertes Routing"""
    
    def __init__(self, context_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Für Uncertainty Estimation
        self.num_actions = num_actions
        
    def forward(self, context):
        return self.network(context)


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent für intelligentes Routing zu Junior-Agenten.
    
    Nutzt NeuralUCB für kontextabhängige Agentenauswahl.
    """
    
    # Verfügbare Routing-Strategien
    ROUTING_STRATEGIES = {
        'news_only': {'news': True, 'technical': False, 'fundamental': False},
        'technical_only': {'news': False, 'technical': True, 'fundamental': False},
        'fundamental_only': {'news': False, 'technical': False, 'fundamental': True},
        'news_technical': {'news': True, 'technical': True, 'fundamental': False},
        'news_fundamental': {'news': True, 'technical': False, 'fundamental': True},
        'technical_fundamental': {'news': False, 'technical': True, 'fundamental': True},
        'all_agents': {'news': True, 'technical': True, 'fundamental': True}
    }
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.context_dim = config.get('context_dim', 16)
        self.num_actions = len(self.ROUTING_STRATEGIES)
        
        # Neural UCB Model
        self.model = NeuralUCBNetwork(
            self.context_dim,
            self.num_actions,
            hidden_dim=config.get('hidden_dim', 128)
        )
        
        if config.get('model_path'):
            self.model.load_state_dict(torch.load(config['model_path']))
        
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer für Online-Learning
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        # Replay Buffer
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 10000))
        
        # Exploration parameter
        self.exploration_factor = config.get('exploration_factor', 1.0)
        self.exploration_decay = config.get('exploration_decay', 0.995)
        self.min_exploration = config.get('min_exploration', 0.1)
        
        # Uncertainty tracking
        self.action_counts = np.zeros(self.num_actions)
        self.action_rewards = np.zeros(self.num_actions)
        
        # Latency tracking (für Effizienz-Bonus)
        self.avg_latencies = {}
        
    def select_routing_strategy(
        self,
        context: Dict,
        explore: bool = True
    ) -> Tuple[str, Dict, float]:
        """
        Wähle Routing-Strategie basierend auf Kontext
        
        Args:
            context: Markt- und Query-Kontext
            explore: Ob Exploration erlaubt ist
            
        Returns:
            (strategy_name, strategy_config, confidence)
        """
        # Encode context zu Feature-Vector
        context_vector = self._encode_context(context)
        
        # Neural UCB
        with torch.no_grad():
            q_values = self.model(context_vector)
            
            if explore:
                # UCB: Q(a) + β * sqrt(log(t) / n(a))
                total_steps = self.action_counts.sum()
                ucb_bonus = self.exploration_factor * np.sqrt(
                    np.log(max(total_steps, 1)) / (self.action_counts + 1)
                )
                
                ucb_values = q_values.cpu().numpy() + ucb_bonus
                action_idx = np.argmax(ucb_values)
                confidence = float(q_values[action_idx])
            else:
                # Pure exploitation
                action_idx = torch.argmax(q_values).item()
                confidence = float(q_values[action_idx])
        
        # Map action index to strategy
        strategy_names = list(self.ROUTING_STRATEGIES.keys())
        strategy_name = strategy_names[action_idx]
        strategy_config = self.ROUTING_STRATEGIES[strategy_name]
        
        # Update counts
        self.action_counts[action_idx] += 1
        
        return strategy_name, strategy_config, confidence
    
    def _encode_context(self, context: Dict) -> torch.Tensor:
        """
        Encode Marktkontext zu Feature-Vector
        
        Context sollte enthalten:
        - market_regime: 'bull', 'bear', 'sideways'
        - volatility: VIX-Wert
        - query_type: 'technical', 'fundamental', 'news', 'mixed'
        - time_horizon: 'intraday', 'short', 'medium', 'long'
        - information_density: News-Volumen
        """
        
        features = []
        
        # Market regime (one-hot)
        regime_map = {'bull': [1, 0, 0], 'bear': [0, 1, 0], 'sideways': [0, 0, 1]}
        regime = context.get('market_regime', 'sideways')
        features.extend(regime_map.get(regime, [0, 0, 1]))
        
        # Volatility (normalized)
        vix = context.get('volatility', 20.0)
        features.append(min(vix / 50.0, 1.0))  # Normalize zu [0, 1]
        
        # Query type (one-hot)
        query_map = {
            'technical': [1, 0, 0, 0],
            'fundamental': [0, 1, 0, 0],
            'news': [0, 0, 1, 0],
            'mixed': [0, 0, 0, 1]
        }
        query_type = context.get('query_type', 'mixed')
        features.extend(query_map.get(query_type, [0, 0, 0, 1]))
        
        # Time horizon (one-hot)
        horizon_map = {
            'intraday': [1, 0, 0, 0],
            'short': [0, 1, 0, 0],
            'medium': [0, 0, 1, 0],
            'long': [0, 0, 0, 1]
        }
        horizon = context.get('time_horizon', 'short')
        features.extend(horizon_map.get(horizon, [0, 1, 0, 0]))
        
        # Information density (normalized)
        info_density = context.get('information_density', 10)
        features.append(min(info_density / 100.0, 1.0))
        
        # Trading session
        session_map = {
            'pre_market': [1, 0, 0, 0],
            'market_hours': [0, 1, 0, 0],
            'after_hours': [0, 0, 1, 0],
            'closed': [0, 0, 0, 1]
        }
        session = context.get('trading_session', 'market_hours')
        features.extend(session_map.get(session, [0, 1, 0, 0]))
        
        # Pad oder truncate zu context_dim
        while len(features) < self.context_dim:
            features.append(0.0)
        features = features[:self.context_dim]
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)
    
    def update(self, context: Dict, action: int, reward: float):
        """
        Online-Update des Modells basierend auf Feedback
        
        Args:
            context: Der Kontext bei der Entscheidung
            action: Die gewählte Aktion (Strategie-Index)
            reward: Die erhaltene Belohnung
        """
        # Store in replay buffer
        self.replay_buffer.append({
            'context': context,
            'action': action,
            'reward': reward
        })
        
        # Update reward statistics
        self.action_rewards[action] += reward
        
        # Mini-batch update
        if len(self.replay_buffer) >= 32:
            self._train_step()
        
        # Decay exploration
        self.exploration_factor = max(
            self.min_exploration,
            self.exploration_factor * self.exploration_decay
        )
    
    def _train_step(self, batch_size: int = 32):
        """Training step mit Replay-Buffer"""
        
        # Sample batch
        indices = np.random.choice(
            len(self.replay_buffer),
            size=min(batch_size, len(self.replay_buffer)),
            replace=False
        )
        
        batch = [self.replay_buffer[i] for i in indices]
        
        # Prepare tensors
        contexts = torch.stack([
            self._encode_context(sample['context']) for sample in batch
        ])
        actions = torch.tensor([sample['action'] for sample in batch], dtype=torch.long)
        rewards = torch.tensor([sample['reward'] for sample in batch], dtype=torch.float32)
        
        contexts = contexts.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        # Forward pass
        q_values = self.model(contexts)
        
        # Get Q-values for taken actions
        q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Loss: MSE zwischen predicted Q und actual reward
        loss = F.mse_loss(q_values_selected, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def get_statistics(self) -> Dict:
        """Statistiken über Routing-Entscheidungen"""
        
        strategy_names = list(self.ROUTING_STRATEGIES.keys())
        
        stats = {
            'total_decisions': int(self.action_counts.sum()),
            'exploration_factor': self.exploration_factor,
            'strategy_usage': {}
        }
        
        for i, name in enumerate(strategy_names):
            count = int(self.action_counts[i])
            total_reward = float(self.action_rewards[i])
            avg_reward = total_reward / max(count, 1)
            
            stats['strategy_usage'][name] = {
                'count': count,
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'usage_pct': count / max(stats['total_decisions'], 1) * 100
            }
        
        return stats
    
    def save(self, path: str):
        """Save model und state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_counts': self.action_counts,
            'action_rewards': self.action_rewards,
            'exploration_factor': self.exploration_factor
        }, path)
    
    def load(self, path: str):
        """Load model und state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_counts = checkpoint['action_counts']
        self.action_rewards = checkpoint['action_rewards']
        self.exploration_factor = checkpoint['exploration_factor']


# Beispiel-Nutzung
if __name__ == "__main__":
    config = {
        'context_dim': 16,
        'hidden_dim': 128,
        'learning_rate': 1e-3,
        'exploration_factor': 1.0
    }
    
    supervisor = SupervisorAgent(config)
    
    # Beispiel-Kontext
    context = {
        'market_regime': 'bull',
        'volatility': 18.5,
        'query_type': 'mixed',
        'time_horizon': 'short',
        'information_density': 45,
        'trading_session': 'market_hours'
    }
    
    strategy_name, strategy_config, confidence = supervisor.select_routing_strategy(context)
    
    print(f"Selected Strategy: {strategy_name}")
    print(f"Config: {strategy_config}")
    print(f"Confidence: {confidence:.3f}")
    
    # Simuliere Feedback
    reward = 0.8  # Positive Belohnung
    action_idx = list(supervisor.ROUTING_STRATEGIES.keys()).index(strategy_name)
    supervisor.update(context, action_idx, reward)
    
    # Statistiken
    print("\nStatistics:")
    print(json.dumps(supervisor.get_statistics(), indent=2))
