#!/usr/bin/env python3
"""Experience Replay Buffer - Task 11.1"""
from collections import deque
from loguru import logger

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        logger.info(f"ReplayBuffer: capacity={capacity}")
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

if __name__ == "__main__":
    print("✓ Replay Buffer ready")
