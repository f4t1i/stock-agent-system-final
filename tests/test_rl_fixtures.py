#!/usr/bin/env python3
import numpy as np

class RLFixtures:
    @staticmethod
    def create_mock_experiences(batch_size=32):
        return {
            'states': np.random.randn(batch_size, 10),
            'actions': np.random.randint(0, 2, batch_size),
            'rewards': np.random.randn(batch_size),
            'next_states': np.random.randn(batch_size, 10),
            'dones': np.random.choice([True, False], batch_size)
        }
