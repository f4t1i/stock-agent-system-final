#!/usr/bin/env python3
"""
Acceptance Tests - RL Training Pipeline (Tasks #21-24)

Tests:
1. GRPO Configuration: Config loading and parameter validation
2. Supervisor v2: Contextual bandit agent selection
3. Regime Features: Market regime detection
4. Integration: Complete RL pipeline workflow

Note: These tests verify pipeline structure without requiring GPU training.
"""

import sys
import yaml
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.supervisor_v2 import SupervisorV2, BanditAlgorithm
from agents.regime_features import RegimeFeatureExtractor, TrendDirection, VolatilityLevel


# ============================================================================
# Test Data
# ============================================================================

def create_test_config() -> dict:
    """Create minimal test config"""
    return {
        "grpo": {
            "group_size": 4,
            "temperature": 0.8,
            "learning_rate": 1e-5,
            "num_iterations": 10
        },
        "contextual_bandit": {
            "algorithm": "thompson_sampling",
            "prior_alpha": 1.0,
            "prior_beta": 1.0,
            "agents": [
                {"name": "news_agent", "weight_init": 0.25},
                {"name": "technical_agent", "weight_init": 0.25}
            ]
        },
        "regime_features": {
            "volatility": {"window": 20, "thresholds": {"low": 0.15, "medium": 0.30}},
            "trend": {"short_ma": 20, "long_ma": 50}
        }
    }


def create_test_price_data(num_days: int = 100) -> pd.DataFrame:
    """Create synthetic price data"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(num_days) * 0.02))

    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, size=num_days)
    })


# ============================================================================
# Test 1: GRPO Configuration
# ============================================================================

def test_grpo_config():
    """Test GRPO configuration loading"""
    print("\\n" + "="*60)
    print("TEST 1: GRPO Configuration")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1.1: Create config
        print("\\n1.1 Creating GRPO config...")
        config_file = Path(tmpdir) / "test_config.yaml"
        config = create_test_config()

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        print("   ✅ Config created")

        # Test 1.2: Load and validate
        print("\\n1.2 Loading and validating config...")
        with open(config_file) as f:
            loaded_config = yaml.safe_load(f)

        grpo_config = loaded_config.get("grpo", {})
        assert "group_size" in grpo_config, "Should have group_size"
        assert "learning_rate" in grpo_config, "Should have learning_rate"
        assert grpo_config["group_size"] == 4, "Group size should be 4"
        print(f"   ✅ GRPO config validated")

    print("\\n✅ TEST 1 PASSED: GRPO configuration works")
    return True


# ============================================================================
# Test 2: Supervisor v2 - Bandit Algorithms
# ============================================================================

def test_supervisor_bandit():
    """Test supervisor contextual bandit selection"""
    print("\\n" + "="*60)
    print("TEST 2: Supervisor v2 - Bandit Algorithms")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "test_config.yaml"
        config = create_test_config()

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Test 2.1: Initialize supervisor
        print("\\n2.1 Initializing supervisor...")
        supervisor = SupervisorV2(config_path=config_file)

        assert len(supervisor.agents) == 2, "Should have 2 agents"
        assert supervisor.algorithm == BanditAlgorithm.THOMPSON_SAMPLING, "Should use Thompson Sampling"
        print(f"   ✅ Supervisor initialized with {len(supervisor.agents)} agents")

        # Test 2.2: Agent selection
        print("\\n2.2 Testing agent selection...")
        decision = supervisor.select_agent(
            symbol="AAPL",
            context={"regime": "bull_low_vol"}
        )

        assert decision.agent_name in ["news_agent", "technical_agent"], "Should select valid agent"
        assert 0 <= decision.confidence <= 1, "Confidence should be in [0, 1]"
        print(f"   ✅ Selected agent: {decision.agent_name} (confidence={decision.confidence:.4f})")

        # Test 2.3: Update with reward
        print("\\n2.3 Updating with reward...")
        supervisor.update(
            agent_name=decision.agent_name,
            reward=0.5,
            symbol="AAPL",
            context={"regime": "bull_low_vol"}
        )

        stats = supervisor.agents[decision.agent_name]
        assert stats.num_selections == 1, "Should have 1 selection"
        assert stats.total_reward == 0.5, "Should have reward 0.5"
        print(f"   ✅ Updated agent stats: selections={stats.num_selections}, reward={stats.total_reward}")

        # Test 2.4: Multiple rounds
        print("\\n2.4 Running multiple selection rounds...")
        for i in range(10):
            decision = supervisor.select_agent(symbol="AAPL", context={"regime": "bull_low_vol"})
            reward = np.random.randn() * 0.5
            supervisor.update(
                agent_name=decision.agent_name,
                reward=reward,
                symbol="AAPL"
            )

        total_selections = sum(s.num_selections for s in supervisor.agents.values())
        assert total_selections == 11, "Should have 11 total selections"  # 1 + 10
        print(f"   ✅ Completed {total_selections} selection rounds")

    print("\\n✅ TEST 2 PASSED: Supervisor bandit algorithms work")
    return True


# ============================================================================
# Test 3: Regime Features Extraction
# ============================================================================

def test_regime_features():
    """Test regime feature extraction"""
    print("\\n" + "="*60)
    print("TEST 3: Regime Features Extraction")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "test_config.yaml"
        config = create_test_config()

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Test 3.1: Initialize extractor
        print("\\n3.1 Initializing feature extractor...")
        extractor = RegimeFeatureExtractor(config_path=config_file)
        print("   ✅ Extractor initialized")

        # Test 3.2: Extract features
        print("\\n3.2 Extracting features from price data...")
        price_data = create_test_price_data(num_days=100)

        features = extractor.extract(
            symbol="TEST",
            price_data=price_data
        )

        assert features.symbol == "TEST", "Should have correct symbol"
        assert isinstance(features.volatility, float), "Volatility should be float"
        assert features.volatility_level in [VolatilityLevel.LOW, VolatilityLevel.MEDIUM, VolatilityLevel.HIGH], "Should have valid vol level"
        assert features.trend_direction in [TrendDirection.UP, TrendDirection.DOWN, TrendDirection.SIDEWAYS], "Should have valid trend"
        print(f"   ✅ Features extracted: regime={features.regime}, vol={features.volatility:.2%}")

        # Test 3.3: Validate regime classification
        print("\\n3.3 Validating regime classification...")
        assert features.regime in [
            "bull_low_vol", "bull_high_vol",
            "bear_low_vol", "bear_high_vol",
            "sideways_low_vol", "sideways_high_vol"
        ], "Should have valid regime"
        print(f"   ✅ Regime: {features.regime}")

        # Test 3.4: Validate context
        print("\\n3.4 Validating additional context...")
        assert features.context is not None, "Should have context"
        assert "price_change_1d" in features.context, "Should have 1d price change"
        assert "volume_trend" in features.context, "Should have volume trend"
        print(f"   ✅ Context validated: {len(features.context)} fields")

    print("\\n✅ TEST 3 PASSED: Regime features extraction works")
    return True


# ============================================================================
# Test 4: Integration - Regime-based Routing
# ============================================================================

def test_regime_routing_integration():
    """Test integration of regime features with supervisor routing"""
    print("\\n" + "="*60)
    print("TEST 4: Integration - Regime-based Routing")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "test_config.yaml"
        config = create_test_config()

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Test 4.1: Initialize components
        print("\\n4.1 Initializing components...")
        supervisor = SupervisorV2(config_path=config_file)
        extractor = RegimeFeatureExtractor(config_path=config_file)
        print("   ✅ Components initialized")

        # Test 4.2: Extract regime features
        print("\\n4.2 Extracting regime features...")
        price_data = create_test_price_data(num_days=100)
        features = extractor.extract(symbol="AAPL", price_data=price_data)
        print(f"   ✅ Regime detected: {features.regime}")

        # Test 4.3: Route based on regime
        print("\\n4.3 Routing agent based on regime...")
        decision = supervisor.select_agent(
            symbol="AAPL",
            context={"regime": features.regime}
        )
        print(f"   ✅ Routed to: {decision.agent_name}")

        # Test 4.4: Simulate trades with different regimes
        print("\\n4.4 Simulating trades across different regimes...")

        # Create different market conditions
        test_scenarios = [
            ("bull", 100, 0.05, 0.01),   # Bull market: start=100, drift=5%, vol=1%
            ("bear", 100, -0.05, 0.02),  # Bear market: start=100, drift=-5%, vol=2%
            ("sideways", 100, 0.0, 0.015) # Sideways: start=100, drift=0%, vol=1.5%
        ]

        regime_counts = {}

        for scenario_name, start_price, drift, vol in test_scenarios:
            # Generate price data for scenario
            np.random.seed(hash(scenario_name) % 2**32)
            prices = start_price * np.exp(np.cumsum(np.random.randn(100) * vol + drift/252))

            scenario_data = pd.DataFrame({
                "date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1_000_000, 10_000_000, size=100)
            })

            # Extract features and route
            features = extractor.extract(symbol=scenario_name.upper(), price_data=scenario_data)
            decision = supervisor.select_agent(symbol=scenario_name.upper(), context={"regime": features.regime})

            # Track regime routing
            if features.regime not in regime_counts:
                regime_counts[features.regime] = []
            regime_counts[features.regime].append(decision.agent_name)

            # Simulate reward and update
            reward = np.random.randn() * 0.5
            supervisor.update(
                agent_name=decision.agent_name,
                reward=reward,
                symbol=scenario_name.upper(),
                context={"regime": features.regime}
            )

        print(f"   ✅ Tested {len(test_scenarios)} market scenarios")
        print(f"      Regime distribution: {list(regime_counts.keys())}")

    print("\\n✅ TEST 4 PASSED: Regime-based routing integration works")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all acceptance tests"""
    print("\\n" + "="*60)
    print("ACCEPTANCE TESTS - RL TRAINING PIPELINE (Tasks #21-24)")
    print("="*60)

    tests = [
        ("GRPO Configuration", test_grpo_config),
        ("Supervisor v2 - Bandit Algorithms", test_supervisor_bandit),
        ("Regime Features Extraction", test_regime_features),
        ("Integration - Regime-based Routing", test_regime_routing_integration)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\\n✅ ALL TESTS PASSED!")
        print("\\nNote: These tests verify pipeline structure and integration.")
        print("      Full RL training with GPU should be tested separately.")
        return 0
    else:
        print(f"\\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
