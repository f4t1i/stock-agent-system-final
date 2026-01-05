#!/usr/bin/env python3
"""Policy Optimization Loop - Task 9.3"""
import numpy as np
from loguru import logger

class PolicyOptimizer:
    def __init__(self, learning_rate=3e-4, num_epochs=4):
        self.lr = learning_rate
        self.num_epochs = num_epochs
        logger.info(f"PolicyOptimizer initialized: lr={learning_rate}")
    
    def optimize(self, experiences, ppo_algorithm):
        for epoch in range(self.num_epochs):
            loss, p_loss, v_loss = ppo_algorithm.compute_loss(
                experiences['old_log_probs'],
                experiences['new_log_probs'],
                experiences['advantages'],
                experiences['values'],
                experiences['returns']
            )
            logger.info(f"Epoch {epoch+1}: loss={loss:.4f}")
        return {'final_loss': loss}

if __name__ == "__main__":
    print("✓ Policy Optimization ready")
