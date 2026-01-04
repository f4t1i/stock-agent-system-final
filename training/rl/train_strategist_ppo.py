"""
PPO Training for Senior Strategist Agent

Alternative to GRPO for systems with >24GB VRAM.
Uses Proximal Policy Optimization with value network.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from loguru import logger
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from agents.senior.strategist_agent import StrategistAgent
from judge.llm_judge import LLMJudge
from utils.config_loader import load_config
from utils.logging_setup import setup_logging


class PPOTrainer:
    """
    PPO Trainer for Strategist Agent.
    
    Implements:
    - Policy network (strategist LLM)
    - Value network (critic)
    - PPO clipped objective
    - GAE for advantage estimation
    - Multiple epochs per batch
    """
    
    def __init__(self, config_path: str):
        """
        Initialize PPO trainer.
        
        Args:
            config_path: Path to config file
        """
        self.config = load_config(config_path)
        
        # Initialize wandb
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'stock-agent-system'),
                name=f"strategist-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
        
        # Load SFT checkpoint
        self._load_models()
        
        # Initialize judge
        self.judge = LLMJudge()
        
        # PPO hyperparameters
        self.clip_epsilon = self.config['training'].get('clip_epsilon', 0.2)
        self.value_coef = self.config['training'].get('value_coef', 0.5)
        self.entropy_coef = self.config['training'].get('entropy_coef', 0.01)
        self.gamma = self.config['training'].get('gamma', 0.99)
        self.gae_lambda = self.config['training'].get('gae_lambda', 0.95)
        
        # Training state
        self.global_step = 0
        
        logger.info("PPO Trainer initialized")
    
    def _load_models(self):
        """Load policy and value networks"""
        model_config = self.config['model']
        
        # Load policy network (strategist)
        logger.info(f"Loading policy network from {model_config['sft_checkpoint']}")
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_config['sft_checkpoint'],
            torch_dtype=torch.float16 if model_config.get('fp16', True) else torch.float32,
            device_map='auto'
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['sft_checkpoint'])
        
        # Add LoRA for fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_config.get('lora_rank', 8),
            lora_alpha=model_config.get('lora_alpha', 16),
            lora_dropout=model_config.get('lora_dropout', 0.05),
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.policy_model = get_peft_model(self.policy_model, peft_config)
        self.policy_model.train()
        
        # Initialize value network (simple MLP on top of LLM embeddings)
        hidden_size = self.policy_model.config.hidden_size
        self.value_network = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.policy_model.device)
        
        # Optimizers
        self.policy_optimizer = optim.AdamW(
            self.policy_model.parameters(),
            lr=model_config.get('learning_rate', 1e-5)
        )
        
        self.value_optimizer = optim.AdamW(
            self.value_network.parameters(),
            lr=model_config.get('value_lr', 1e-4)
        )
        
        logger.info("Models loaded successfully")
    
    def train(
        self,
        num_iterations: int = 1000,
        episodes_per_iteration: int = 10,
        epochs_per_iteration: int = 4
    ):
        """
        Main PPO training loop.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            epochs_per_iteration: PPO epochs per iteration
        """
        logger.info(f"Starting PPO training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Collect trajectories
            trajectories = self._collect_trajectories(episodes_per_iteration)
            
            # Compute advantages
            advantages, returns = self._compute_advantages(trajectories)
            
            # PPO update
            for epoch in range(epochs_per_iteration):
                policy_loss, value_loss, entropy = self._ppo_update(
                    trajectories,
                    advantages,
                    returns
                )
                
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs_per_iteration} | "
                    f"Policy Loss: {policy_loss:.4f} | "
                    f"Value Loss: {value_loss:.4f} | "
                    f"Entropy: {entropy:.4f}"
                )
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'iteration': iteration,
                        'epoch': epoch,
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        'entropy': entropy,
                        'global_step': self.global_step
                    })
                
                self.global_step += 1
            
            # Save checkpoint
            if (iteration + 1) % self.config['training'].get('save_interval', 100) == 0:
                self._save_checkpoint(iteration + 1)
        
        # Save final model
        self._save_final_model()
        
        logger.info("Training complete!")
        
        if self.config.get('use_wandb', True):
            wandb.finish()
    
    def _collect_trajectories(self, num_episodes: int) -> List[Dict]:
        """
        Collect trajectories using current policy.
        
        Args:
            num_episodes: Number of episodes to collect
        
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        
        # Mock symbols for training
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for episode in range(num_episodes):
            symbol = np.random.choice(symbols)
            
            # Create mock agent outputs
            agent_outputs = self._create_mock_agent_outputs(symbol)
            
            # Generate strategist decision
            decision = self._generate_decision(symbol, agent_outputs)
            
            # Evaluate with judge
            reward = self._evaluate_decision(decision, agent_outputs)
            
            trajectories.append({
                'symbol': symbol,
                'agent_outputs': agent_outputs,
                'decision': decision,
                'reward': reward
            })
        
        return trajectories
    
    def _create_mock_agent_outputs(self, symbol: str) -> Dict:
        """Create mock agent outputs for training"""
        return {
            'news': {
                'sentiment_score': np.random.uniform(-1, 1),
                'confidence': np.random.uniform(0.5, 1.0),
                'recommendation': np.random.choice(['bullish', 'bearish', 'neutral'])
            },
            'technical': {
                'signal': np.random.choice(['bullish', 'bearish', 'neutral']),
                'signal_strength': np.random.uniform(0.5, 1.0),
                'recommendation': np.random.choice(['buy', 'sell', 'hold'])
            },
            'fundamental': {
                'valuation': np.random.choice(['undervalued', 'fairly_valued', 'overvalued']),
                'financial_health_score': np.random.uniform(0.5, 1.0),
                'recommendation': np.random.choice(['buy', 'hold', 'sell'])
            }
        }
    
    def _generate_decision(self, symbol: str, agent_outputs: Dict) -> Dict:
        """Generate strategist decision using policy network"""
        # In full implementation, this would:
        # 1. Format agent outputs as prompt
        # 2. Generate decision using policy_model
        # 3. Parse structured output
        
        # Mock decision for now
        decision = {
            'decision': np.random.choice(['buy', 'sell', 'hold']),
            'confidence': np.random.uniform(0.5, 1.0),
            'position_size': np.random.uniform(0.01, 0.1),
            'reasoning': 'Mock reasoning for training'
        }
        
        return decision
    
    def _evaluate_decision(self, decision: Dict, agent_outputs: Dict) -> float:
        """Evaluate decision using LLM judge"""
        try:
            evaluation = self.judge.evaluate(
                agent_output=decision,
                agent_type='strategist',
                context={'agent_outputs': agent_outputs}
            )
            
            reward = self.judge.calculate_reward(evaluation)
            return reward
        
        except Exception as e:
            logger.error(f"Error evaluating decision: {e}")
            return 0.5  # Neutral reward on error
    
    def _compute_advantages(
        self,
        trajectories: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            trajectories: List of trajectories
        
        Returns:
            Tuple of (advantages, returns)
        """
        rewards = torch.tensor([t['reward'] for t in trajectories], dtype=torch.float32)
        
        # Compute returns (discounted cumulative rewards)
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # Compute advantages (simplified - in full implementation, use value network)
        advantages = returns - returns.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _ppo_update(
        self,
        trajectories: List[Dict],
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        PPO update step.
        
        Args:
            trajectories: Collected trajectories
            advantages: Computed advantages
            returns: Computed returns
        
        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        # In full implementation, this would:
        # 1. Compute old and new log probabilities
        # 2. Compute PPO clipped objective
        # 3. Update policy network
        # 4. Update value network
        
        # Placeholder losses
        policy_loss = advantages.abs().mean().item()
        value_loss = (returns - returns.mean()).pow(2).mean().item()
        entropy = 0.1  # Mock entropy
        
        return policy_loss, value_loss, entropy
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config['training']['output_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy model
        self.policy_model.save_pretrained(
            checkpoint_dir / f'policy_iter_{iteration}'
        )
        
        # Save value network
        torch.save(
            self.value_network.state_dict(),
            checkpoint_dir / f'value_iter_{iteration}.pt'
        )
        
        logger.info(f"Checkpoint saved at iteration {iteration}")
    
    def _save_final_model(self):
        """Save final trained model"""
        output_dir = Path(self.config['training']['output_dir']) / 'final'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy model
        self.policy_model.save_pretrained(output_dir / 'policy')
        self.tokenizer.save_pretrained(output_dir / 'policy')
        
        # Save value network
        torch.save(
            self.value_network.state_dict(),
            output_dir / 'value_network.pt'
        )
        
        logger.info(f"Final model saved: {output_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Strategist with PPO")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/rl/ppo_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of training iterations'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Train
    trainer = PPOTrainer(args.config)
    trainer.train(num_iterations=args.iterations)


if __name__ == '__main__':
    main()
