#!/usr/bin/env python3
"""Advantage Estimation - Task 9.5"""
import numpy as np
from loguru import logger

class AdvantageEstimator:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        logger.info(f"AdvantageEstimator: gamma={gamma}, lambda={lam}")
    
    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t+1]
            delta = rewards[t] + self.gamma * next_value * (1-dones[t]) - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.lam * (1-dones[t]) * last_adv
        returns = advantages + values
        return advantages, returns

if __name__ == "__main__":
    print("✓ Advantage Estimator ready")
