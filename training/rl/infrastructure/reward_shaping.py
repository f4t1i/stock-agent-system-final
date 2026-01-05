#!/usr/bin/env python3
"""Reward Shaping - Task 11.3"""
import numpy as np
from loguru import logger

class RewardShaper:
    def __init__(self, base_reward_weight=1.0, length_penalty=0.01):
        self.base_weight = base_reward_weight
        self.length_penalty = length_penalty
        logger.info("RewardShaper initialized")
    
    def shape(self, base_reward, response_length):
        shaped = self.base_weight * base_reward - self.length_penalty * response_length
        return shaped

if __name__ == "__main__":
    print("✓ Reward Shaper ready")
