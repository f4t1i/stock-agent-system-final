#!/usr/bin/env python3
"""PPO Algorithm - Task 9.2"""
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class PPOConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
class PPOAlgorithm:
    def __init__(self, config: PPOConfig):
        self.config = config
        logger.info("PPO initialized")
    
    def compute_loss(self, old_log_probs, new_log_probs, advantages, values, returns):
        ratio = np.exp(new_log_probs - old_log_probs)
        clipped_ratio = np.clip(ratio, 1-self.config.clip_epsilon, 1+self.config.clip_epsilon)
        policy_loss = -np.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = ((returns - values) ** 2).mean()
        total_loss = policy_loss + self.config.value_coef * value_loss
        return total_loss, policy_loss, value_loss

if __name__ == "__main__":
    print("✓ PPO Algorithm ready")
