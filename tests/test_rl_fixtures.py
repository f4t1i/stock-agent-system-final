#!/usr/bin/env python3
"""Minimal RL Fixtures for test collection"""
import tempfile
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class MockEnvironment:
    state_dim: int
    action_dim: int

class RLFixtures:
    """Minimal fixtures for RL tests"""
    
    @staticmethod
    def create_mock_experiences(batch_size=32):
        """Create mock experiences for RL training"""
        state_dim = 10
        action_dim = 2
        states = np.random.randn(batch_size, state_dim)
        actions = np.random.randint(0, action_dim, batch_size)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, state_dim)
        dones = np.random.choice([True, False], batch_size)
        old_log_probs = np.random.randn(batch_size)
        new_log_probs = np.random.randn(batch_size)
        advantages = np.random.randn(batch_size)
        values = np.random.randn(batch_size)
        returns = np.random.randn(batch_size)
        return {
            'states': states, 'actions': actions, 'rewards': rewards,
            'next_states': next_states, 'dones': dones,
            'old_log_probs': old_log_probs, 'new_log_probs': new_log_probs,
            'advantages': advantages, 'values': values, 'returns': returns
        }
    
    @staticmethod
    def create_truncated_experience(batch_size=16):
        return RLFixtures.create_mock_experiences(batch_size)
    
    @staticmethod
    def create_mock_policy_output(logits):
        return {
            'action': np.argmax(logits, axis=1),
            'log_prob': np.log(np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True) + 1e-8)
        }
    
    @staticmethod
    def create_reward_hints(episode_rewards):
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'episode_lengths': len(episode_rewards)
        }
    
    @staticmethod
    def create_dataloader(batch_size=16, num_batches=10):
        experiences = RLFixtures.create_mock_experiences(batch_size * num_batches)
        return {'states': experiences['states'], 'actions': experiences['actions'],
                'rewards': experiences['rewards'], 'next_states': experiences['next_states'],
                'dones': experiences['dones']}
    
    @staticmethod
    def create_policy_dataset(num_samples=100):
        states = np.random.randn(num_samples, 10)
        actions = np.random.randint(0, 2, num_samples)
        log_probs = np.random.randn(num_samples)
        actions_onehot = np.eye(2)[actions]
        return {'states': states, 'actions': actions, 'action_probs': log_probs, 'actions_onehot': actions_onehot}
