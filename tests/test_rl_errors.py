#!/usr/bin/env python3
"""Error Cases Tests for RL - Task 12.3"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_empty_buffer():
    from training.rl.infrastructure.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=100)
    samples = buffer.sample(10)
    assert len(samples) == 0
    print("✓ Test 1: Empty buffer - PASSED")

if __name__ == "__main__":
    print("=== RL Error Cases Tests ===\n")
    test_empty_buffer()
    print("\n✅ All tests passed!")
