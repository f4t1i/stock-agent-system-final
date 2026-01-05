#!/usr/bin/env python3
"""Performance Tests for RL - Task 12.4"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_buffer_performance():
    from training.rl.infrastructure.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=10000)
    start = time.time()
    for i in range(1000):
        buffer.add({'data': i})
    elapsed = time.time() - start
    assert elapsed < 1.0
    print(f"✓ Test 1: Buffer ops (1000 items) - {elapsed:.3f}s - PASSED")

if __name__ == "__main__":
    print("=== RL Performance Tests ===\n")
    test_buffer_performance()
    print("\n✅ All tests passed!")
    print("\n🎉🎉🎉 PHASE A1 WEEK 7-8 COMPLETE! 🎉🎉🎉")
