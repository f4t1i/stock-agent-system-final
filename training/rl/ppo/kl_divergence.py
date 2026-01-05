#!/usr/bin/env python3
"""KL Divergence Monitoring - Task 9.4"""
import numpy as np
from loguru import logger

class KLMonitor:
    def __init__(self, target_kl=0.01):
        self.target_kl = target_kl
        self.kl_history = []
        logger.info(f"KLMonitor initialized: target={target_kl}")
    
    def compute_kl(self, old_log_probs, new_log_probs):
        kl = (np.exp(old_log_probs) * (old_log_probs - new_log_probs)).mean()
        self.kl_history.append(kl)
        return kl
    
    def should_stop(self, kl):
        return kl > 1.5 * self.target_kl

if __name__ == "__main__":
    print("✓ KL Divergence Monitor ready")
