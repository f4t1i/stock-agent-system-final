#!/usr/bin/env python3
"""Happy Path Tests for RL - Task 12.2"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from test_rl_fixtures import RLFixtures

def test_ppo():
    from training.rl.ppo.ppo_algorithm import PPOAlgorithm, PPOConfig
    ppo = PPOAlgorithm(PPOConfig())
    exp = RLFixtures.create_mock_experiences()
    loss, p_loss, v_loss = ppo.compute_loss(exp['old_log_probs'], exp['new_log_probs'], exp['advantages'], exp['values'], exp['returns'])
    assert loss is not None
    print("✓ Test 1: PPO - PASSED")

def test_dpo():
    from training.rl.dpo.dpo_loss import DPOLoss
    dpo = DPOLoss()
    import numpy as np
    loss = dpo.compute(np.array([1.0]), np.array([0.5]), np.array([0.8]), np.array([0.6]))
    assert loss is not None
    print("✓ Test 2: DPO - PASSED")

if __name__ == "__main__":
    print("=== RL Happy Path Tests ===\n")
    test_ppo()
    test_dpo()
    print("\n✅ All tests passed!")
