#!/usr/bin/env python3
"""Test Fixtures for RL - Task 12.1"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class RLFixtures:
    @staticmethod
    def create_mock_experiences(n=10):
        import numpy as np
        return {
            'old_log_probs': np.random.normal(0, 1, n),
            'new_log_probs': np.random.normal(0, 1, n),
            'advantages': np.random.normal(0, 1, n),
            'values': np.random.normal(0, 1, n),
            'returns': np.random.normal(0, 1, n)
        }

if __name__ == "__main__":
    print("✓ RL Fixtures ready")
