"""
Enhanced Neural-UCB Supervisor with Uncertainty-Aware Exploration

Improvements for Non-Stationary Environments:
1. Explicit Uncertainty Quantification (Epistemic + Aleatoric)
2. Adaptive Exploration based on Regime Detection
3. Sliding Window for Reward Estimation (handles non-stationarity)
4. Change Point Detection for Regime Shifts
5. Fast Adaptation via Meta-Learning principles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from loguru import logger


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for an action"""
    epistemic: float  # Model uncertainty (lack of knowledge)
    aleatoric: float  # Data uncertainty (inherent randomness)
    total: float  # Combined uncertainty


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer for Uncertainty Estimation"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Mean and log variance for weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Mean and log variance for bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -5)  # Small initial variance
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_logvar, -5)
    
    def forward(self, x, sample=True):
        if sample:
            # Sample weights from distribution
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean weights (no sampling)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """KL divergence from prior (standard normal)"""
        kl_weight = 0.5 * (
            self.weight_mu.pow(2) + 
            self.weight_logvar.exp() - 
            self.weight_logvar - 1
        ).sum()
        
        kl_bias = 0.5 * (
            self.bias_mu.pow(2) + 
            self.bias_logvar.exp() - 
            self.bias_logvar - 1
        ).sum()
        
        return kl_weight + kl_bias


class EnhancedNeuralUCBNetwork(nn.Module):
    """
    Enhanced Neural Network with:
    - Bayesian layers for epistemic uncertainty
    - Dropout for aleatoric uncertainty
    - Ensemble predictions
    """
    
    def __init__(
        self,
        context_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        use_bayesian: bool = True
    ):
        super().__init__()
        
        self.use_bayesian = use_bayesian
        
        if use_bayesian:
            # Bayesian network for uncertainty quantification
            self.fc1 = BayesianLinear(context_dim, hidden_dim)
            self.fc2 = BayesianLinear(hidden_dim, hidden_dim // 2)
            self.fc3 = BayesianLinear(hidden_dim // 2, num_actions)
        else:
            # Standard network with dropout
            self.fc1 = nn.Linear(context_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, num_actions)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        
        self.num_actions = num_actions
    
    def forward(self, x, sample=True, num_samples=1):
        """
        Forward pass with optional sampling for uncertainty
        
        Args:
            x: Input context
            sample: Whether to sample (for Bayesian layers)
            num_samples: Number of samples for uncertainty estimation
        
        Returns:
            If num_samples=1: (mean_output,)
            If num_samples>1: (mean_output, std_output)
        """
        if num_samples == 1:
            # Single forward pass
            if self.use_bayesian:
                x = F.relu(self.fc1(x, sample=sample))
            else:
                x = F.relu(self.fc1(x))
            
            x = self.dropout1(x)
            
            if self.use_bayesian:
                x = F.relu(self.fc2(x, sample=sample))
            else:
                x = F.relu(self.fc2(x))
            
            x = self.dropout2(x)
            
            if self.use_bayesian:
                x = self.fc3(x, sample=sample)
            else:
                x = self.fc3(x)
            
            return x
        else:
            # Multiple forward passes for uncertainty estimation
            outputs = []
            for _ in range(num_samples):
                if self.use_bayesian:
                    h = F.relu(self.fc1(x, sample=True))
                else:
                    h = F.relu(self.fc1(x))
                
                h = self.dropout1(h)
                
                if self.use_bayesian:
                    h = F.relu(self.fc2(h, sample=True))
                else:
                    h = F.relu(self.fc2(h))
                
                h = self.dropout2(h)
                
                if self.use_bayesian:
                    h = self.fc3(h, sample=True)
                else:
                    h = self.fc3(h)
                
                outputs.append(h)
            
            outputs = torch.stack(outputs)
            mean = outputs.mean(dim=0)
            std = outputs.std(dim=0)
            
            return mean, std
    
    def kl_divergence(self):
        """Total KL divergence (for Bayesian layers)"""
        if not self.use_bayesian:
            return 0.0
        
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()


class ChangePointDetector:
    """Detect regime changes via cumulative sum (CUSUM) algorithm"""
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.mean_estimate = 0.0
        self.n_samples = 0
    
    def update(self, value: float) -> bool:
        """
        Update with new value and detect change point
        
        Returns:
            True if change point detected
        """
        # Update mean estimate
        self.n_samples += 1
        delta = value - self.mean_estimate
        self.mean_estimate += delta / self.n_samples
        
        # Update CUSUM
        deviation = value - self.mean_estimate
        self.cusum_pos = max(0, self.cusum_pos + deviation - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - deviation - self.drift)
        
        # Check for change point
        if self.cusum_pos > self.threshold or self.cusum_neg > self.threshold:
            # Reset
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            return True
        
        return False
    
    def reset(self):
        """Reset detector"""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.mean_estimate = 0.0
        self.n_samples = 0


class EnhancedSupervisorAgent:
    """
    Enhanced Supervisor with:
    1. Uncertainty-aware exploration
    2. Regime change detection
    3. Adaptive learning rates
    4. Sliding window for non-stationarity
    """
    
    def __init__(self, config: Dict):
        self.context_dim = config.get('context_dim', 16)
        self.num_agents = config.get('num_agents', 3)  # news, technical, fundamental
        
        # Enhanced Neural-UCB Model
        self.model = EnhancedNeuralUCBNetwork(
            context_dim=self.context_dim,
            num_actions=self.num_agents,
            hidden_dim=config.get('hidden_dim', 128),
            use_bayesian=config.get('use_bayesian', True)
        )
        
        self.device = config.get('device', 'cpu')
        self.model.to(self.device)
        
        # Optimizer with adaptive learning rate
        self.base_lr = config.get('learning_rate', 1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)
        
        # Sliding window for reward tracking (handles non-stationarity)
        self.window_size = config.get('window_size', 100)
        self.reward_windows = {i: deque(maxlen=self.window_size) for i in range(self.num_agents)}
        
        # Exploration parameters
        self.base_exploration = config.get('exploration_factor', 2.0)
        self.exploration_factor = self.base_exploration
        self.min_exploration = config.get('min_exploration', 0.5)
        
        # Change point detection
        self.change_detectors = {i: ChangePointDetector() for i in range(self.num_agents)}
        self.regime_change_detected = False
        
        # Action counts (for UCB)
        self.action_counts = np.zeros(self.num_agents)
        self.total_steps = 0
        
        # Uncertainty tracking
        self.uncertainty_history = deque(maxlen=100)
        
        # Agent names
        self.agent_names = ['news', 'technical', 'fundamental']
    
    def select_agents(
        self,
        context: Dict,
        explore: bool = True
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select agents based on context with uncertainty-aware exploration
        
        Args:
            context: Market context
            explore: Whether to explore (vs pure exploitation)
        
        Returns:
            (selected_agents, uncertainties)
        """
        # Convert context to tensor
        context_tensor = self._context_to_tensor(context)
        
        # Get predictions with uncertainty
        with torch.no_grad():
            mean_rewards, std_rewards = self.model(
                context_tensor,
                sample=True,
                num_samples=10  # Monte Carlo samples for uncertainty
            )
        
        mean_rewards = mean_rewards.cpu().numpy().flatten()
        std_rewards = std_rewards.cpu().numpy().flatten()
        
        # Calculate UCB scores
        ucb_scores = self._calculate_ucb_scores(mean_rewards, std_rewards)
        
        # Select agent(s) based on UCB scores
        if explore:
            # Select top agent(s) with exploration
            selected_idx = np.argmax(ucb_scores)
        else:
            # Pure exploitation
            selected_idx = np.argmax(mean_rewards)
        
        selected_agents = [self.agent_names[selected_idx]]
        
        # Calculate uncertainties
        uncertainties = {
            self.agent_names[i]: {
                'epistemic': float(std_rewards[i]),
                'total': float(std_rewards[i])
            }
            for i in range(self.num_agents)
        }
        
        # Track uncertainty
        avg_uncertainty = np.mean(std_rewards)
        self.uncertainty_history.append(avg_uncertainty)
        
        return selected_agents, uncertainties
    
    def _calculate_ucb_scores(
        self,
        mean_rewards: np.ndarray,
        std_rewards: np.ndarray
    ) -> np.ndarray:
        """
        Calculate UCB scores with uncertainty bonus
        
        UCB = mean_reward + exploration_factor * (uncertainty + count_bonus)
        """
        # Count-based bonus (classic UCB)
        count_bonus = np.sqrt(
            2 * np.log(self.total_steps + 1) / (self.action_counts + 1)
        )
        
        # Uncertainty-based bonus (Neural-UCB)
        uncertainty_bonus = std_rewards
        
        # Combined UCB score
        ucb_scores = mean_rewards + self.exploration_factor * (uncertainty_bonus + count_bonus)
        
        return ucb_scores
    
    def update(
        self,
        context: Dict,
        agent: str,
        reward: float
    ):
        """
        Update model with observed reward
        
        Includes:
        - Sliding window updates
        - Change point detection
        - Adaptive exploration
        """
        agent_idx = self.agent_names.index(agent)
        
        # Update sliding window
        self.reward_windows[agent_idx].append(reward)
        
        # Update action counts
        self.action_counts[agent_idx] += 1
        self.total_steps += 1
        
        # Change point detection
        change_detected = self.change_detectors[agent_idx].update(reward)
        
        if change_detected:
            logger.info(f"Regime change detected for {agent}!")
            self.regime_change_detected = True
            self._handle_regime_change()
        
        # Train model
        self._train_step(context, agent_idx, reward)
        
        # Decay exploration (unless regime change detected)
        if not self.regime_change_detected:
            self.exploration_factor = max(
                self.min_exploration,
                self.exploration_factor * 0.995
            )
    
    def _handle_regime_change(self):
        """Handle detected regime change"""
        # Increase exploration
        self.exploration_factor = self.base_exploration
        
        # Increase learning rate temporarily
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * 2.0
        
        # Clear old reward windows (focus on recent data)
        for window in self.reward_windows.values():
            if len(window) > self.window_size // 2:
                # Keep only recent half
                recent = list(window)[self.window_size // 2:]
                window.clear()
                window.extend(recent)
        
        logger.info("Adapted to regime change: increased exploration and learning rate")
        
        # Reset flag after handling
        self.regime_change_detected = False
    
    def _train_step(self, context: Dict, agent_idx: int, reward: float):
        """Single training step"""
        context_tensor = self._context_to_tensor(context)
        
        # Forward pass
        predicted_rewards = self.model(context_tensor, sample=True)
        
        # Loss: MSE on observed reward
        target = torch.zeros_like(predicted_rewards)
        target[0, agent_idx] = reward
        
        loss = F.mse_loss(predicted_rewards, target)
        
        # Add KL divergence for Bayesian layers
        if self.model.use_bayesian:
            kl_loss = self.model.kl_divergence() / len(self.reward_windows[agent_idx])
            loss += 0.01 * kl_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
    
    def _context_to_tensor(self, context: Dict) -> torch.Tensor:
        """Convert context dict to tensor"""
        # Extract features from context
        features = [
            context.get('volatility', 0.5),
            context.get('trend_strength', 0.0),
            context.get('news_impact', 0.5),
            context.get('rsi', 50.0) / 100.0,
            context.get('macd', 0.0),
            context.get('price', 150.0) / 150.0,
            # Add more features as needed...
        ]
        
        # Pad to context_dim
        while len(features) < self.context_dim:
            features.append(0.0)
        
        features = features[:self.context_dim]
        
        return torch.tensor([features], dtype=torch.float32).to(self.device)
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics for monitoring"""
        stats = {}
        
        for i, agent_name in enumerate(self.agent_names):
            window = self.reward_windows[i]
            if len(window) > 0:
                stats[agent_name] = {
                    'mean_reward': np.mean(window),
                    'std_reward': np.std(window),
                    'count': self.action_counts[i],
                    'recent_rewards': list(window)[-10:]
                }
        
        stats['exploration_factor'] = self.exploration_factor
        stats['total_steps'] = self.total_steps
        stats['avg_uncertainty'] = np.mean(self.uncertainty_history) if self.uncertainty_history else 0.0
        
        return stats


if __name__ == '__main__':
    # Test
    config = {
        'context_dim': 16,
        'num_agents': 3,
        'hidden_dim': 128,
        'use_bayesian': True,
        'learning_rate': 1e-3,
        'exploration_factor': 2.0
    }
    
    supervisor = EnhancedSupervisorAgent(config)
    
    # Test selection
    context = {
        'volatility': 0.8,
        'trend_strength': -0.5,
        'news_impact': 0.9,
        'rsi': 35.0,
        'macd': -1.5,
        'price': 145.0
    }
    
    selected, uncertainties = supervisor.select_agents(context, explore=True)
    print(f"Selected: {selected}")
    print(f"Uncertainties: {uncertainties}")
    
    # Test update
    supervisor.update(context, selected[0], reward=0.8)
    
    print(f"Statistics: {supervisor.get_agent_statistics()}")
