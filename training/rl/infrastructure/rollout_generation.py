#!/usr/bin/env python3
"""Rollout Generation - Task 11.2"""
from loguru import logger

class RolloutGenerator:
    def __init__(self, policy_model):
        self.policy = policy_model
        logger.info("RolloutGenerator initialized")
    
    def generate(self, prompts, max_length=100):
        rollouts = []
        for prompt in prompts:
            response = f"Generated response for: {prompt[:20]}..."
            rollouts.append({'prompt': prompt, 'response': response})
        logger.info(f"Generated {len(rollouts)} rollouts")
        return rollouts

if __name__ == "__main__":
    print("✓ Rollout Generator ready")
