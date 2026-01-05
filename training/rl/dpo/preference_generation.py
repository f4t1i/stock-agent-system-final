#!/usr/bin/env python3
"""Preference Pair Generation - Task 10.5"""
from loguru import logger

class PreferenceGenerator:
    def __init__(self, reward_model):
        self.reward_model = reward_model
        logger.info("PreferenceGenerator initialized")
    
    def generate_pairs(self, prompts, num_responses=2):
        pairs = []
        for prompt in prompts:
            responses = [f"Response {i} to: {prompt}" for i in range(num_responses)]
            rewards = [self.reward_model.predict_reward(prompt, r) for r in responses]
            best_idx = rewards.index(max(rewards))
            worst_idx = rewards.index(min(rewards))
            pairs.append((prompt, responses[best_idx], responses[worst_idx]))
        logger.info(f"Generated {len(pairs)} preference pairs")
        return pairs

if __name__ == "__main__":
    print("✓ Preference Generator ready")
