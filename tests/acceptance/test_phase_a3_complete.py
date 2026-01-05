#!/usr/bin/env python3
"""
Phase A3 Complete Acceptance Tests

End-to-end tests for all Phase A3 features:
- Explainability
- Alerts & Watchlists
- Risk Management
- Confidence Calibration

Tests verify full-stack integration and production readiness.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.reasoning_extractor import ReasoningExtractor
from agents.decision_logger import DecisionLogger
from monitoring.alert_evaluator import AlertEvaluator
from monitoring.notification_dispatcher import NotificationDispatcher
from risk_management.risk_engine import RiskEngine
from risk_management.policy_evaluator import PolicyEvaluator, Policy, PolicyRule, PolicyRuleType, ComparisonOperator
from calibration.confidence_calibrator import ConfidenceCalibrator


def test_explainability_system():
    """Test #1: Explainability System E2E"""
    print("\n" + "="*60)
    print("TEST #1: Explainability System")
    print("="*60)
    
    # Extract reasoning with proper fields
    extractor = ReasoningExtractor()
    decision = {
        "recommendation": "BUY",
        "symbol": "AAPL",
        "confidence": 0.85,
        "reasoning": "Strong technical indicators with positive news sentiment",
        "position_size": 0.05,
        "risk_assessment": "Moderate risk with good upside potential"
    }
    
    # Extract factors
    factors = extractor.extract_factors(decision, "strategist")
    print(f"✓ Reasoning extracted: {len(factors)} factors")
    
    # Extract reasoning text
    reasoning_text = extractor.extract_reasoning(decision, "strategist")
    print(f"✓ Reasoning text: {reasoning_text[:50]}...")
    
    # Generate alternatives
    alternatives = extractor.generate_alternatives(decision, "strategist")
    print(f"✓ Alternatives generated: {len(alternatives)}")
    
    assert len(factors) > 0 or len(alternatives) > 0, "Should extract reasoning or alternatives"
    
    print("✅ Explainability System: PASS\n")
    return True


def test_alerts_watchlist_system():
    """Test #2: Alerts & Watchlist System E2E"""
    print("="*60)
    print("TEST #2: Alerts & Watchlist System")
    print("="*60)
    
    # Create alert
    evaluator = AlertEvaluator()
    # Test alert evaluation
    evaluator = AlertEvaluator()
    current_price = 155.0
    threshold = 150.0
    
    triggered = evaluator.evaluate_condition(current_price, "above", threshold, "test_001")
    print(f"✓ Alert created: AAPL @ ${threshold}")
    print(f"✓ Alert evaluated: {'Triggered' if triggered else 'Not triggered'}")
    assert triggered, "Should trigger alert when price above threshold"
    
    # Dispatch notification
    dispatcher = NotificationDispatcher()
    success = dispatcher.send(
        alert_id="test_001",
        symbol="AAPL",
        alert_type="price_above",
        current_value=155.0,
        threshold=150.0,
        channels=["email"]
    )
    print(f"✓ Notification dispatched: email")
    assert success, "Should dispatch notification"
    
    print("✅ Alerts & Watchlist System: PASS\n")
    return True


def test_risk_management_system():
    """Test #3: Risk Management System E2E"""
    print("="*60)
    print("TEST #3: Risk Management System")
    print("="*60)
    
    # Evaluate trade risk
    engine = RiskEngine()
    trade = {
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 100,
        "price": 150.0,
        "confidence": 0.85,
        "volatility": 0.25,
    }
    
    portfolio = {
        "total_value": 100000.0,
        "peak_value": 120000.0,
    }
    
    result = engine.evaluate_trade(trade, portfolio)
    print(f"✓ Risk evaluated: {'Approved' if result.approved else 'Rejected'}")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"  Checks: {len(result.checks)} performed")
    
    for check in result.checks:
        print(f"    - {check.name}: {check.status.value}")
    
    assert len(result.checks) > 0, "Should perform risk checks"
    
    # Evaluate custom policy
    evaluator = PolicyEvaluator()
    policy = PolicyEvaluator.create_template_policy("conservative")
    
    policy_result = evaluator.evaluate(trade, policy)
    print(f"✓ Policy evaluated: {policy_result.policy_name}")
    print(f"  Result: {'Passed' if policy_result.passed else 'Failed'}")
    
    print("✅ Risk Management System: PASS\n")
    return True


def test_calibration_system():
    """Test #4: Confidence Calibration System E2E"""
    print("="*60)
    print("TEST #4: Confidence Calibration System")
    print("="*60)
    
    # Generate synthetic data
    import numpy as np
    np.random.seed(42)
    n_samples = 1000
    
    # Overconfident predictions
    confidences = np.random.beta(5, 2, n_samples).tolist()
    outcomes = (np.random.rand(n_samples) < (np.array(confidences) * 0.8)).tolist()
    
    # Train calibrator
    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(confidences[:800], outcomes[:800])
    print("✓ Calibrator trained on 800 samples")
    
    # Evaluate before calibration
    metrics_before = calibrator.evaluate(confidences[800:], outcomes[800:])
    print(f"✓ Before calibration:")
    print(f"    ECE: {metrics_before.ece:.4f}")
    print(f"    Accuracy: {metrics_before.accuracy:.4f}")
    
    # Calibrate
    calibrated = calibrator.transform(confidences[800:])
    print(f"✓ Calibrated {len(calibrated)} predictions")
    
    # Evaluate after calibration
    metrics_after = calibrator.evaluate(calibrated, outcomes[800:])
    print(f"✓ After calibration:")
    print(f"    ECE: {metrics_after.ece:.4f}")
    print(f"    Accuracy: {metrics_after.accuracy:.4f}")
    
    # Get reliability diagram
    centers, accuracies, counts = calibrator.get_reliability_diagram(
        confidences[800:], outcomes[800:]
    )
    print(f"✓ Reliability diagram: {len(centers)} bins")
    
    assert metrics_after.ece <= metrics_before.ece, "Calibration should improve ECE"
    
    print("✅ Calibration System: PASS\n")
    return True


def test_full_stack_integration():
    """Test #5: Full Stack Integration"""
    print("="*60)
    print("TEST #5: Full Stack Integration")
    print("="*60)
    
    # Simulate full trading workflow
    print("Simulating complete trading workflow...")
    
    # 1. Agent makes decision
    decision = {
        "action": "BUY",
        "symbol": "AAPL",
        "quantity": 100,
        "price": 150.0,
        "confidence": 0.85,
        "volatility": 0.25,
        "reasoning": "Strong fundamentals and positive technical indicators"
    }
    print(f"✓ Step 1: Agent decision - {decision['action']} {decision['symbol']}")
    
    # 2. Extract reasoning
    extractor = ReasoningExtractor()
    decision_with_fields = {
        **decision,
        "recommendation": decision["action"],
        "position_size": 0.05,
        "risk_assessment": "Moderate"
    }
    factors = extractor.extract_factors(decision_with_fields, "strategist")
    reasoning_text = extractor.extract_reasoning(decision_with_fields, "strategist")
    print(f"✓ Step 2: Reasoning extracted - {len(factors)} factors")
    
    # 3. Evaluate risk
    engine = RiskEngine()
    portfolio = {"total_value": 100000.0, "peak_value": 120000.0}
    risk_result = engine.evaluate_trade(decision, portfolio)
    print(f"✓ Step 3: Risk evaluated - {'Approved' if risk_result.approved else 'Rejected'}")
    
    if not risk_result.approved:
        print("  Trade rejected by risk engine")
        print("✅ Full Stack Integration: PASS (Risk rejection)\n")
        return True
    
    # 4. Check alerts
    evaluator = AlertEvaluator()
    market_data = {decision["symbol"]: decision["price"]}
    triggered_alerts = evaluator.evaluate(market_data)
    print(f"✓ Step 4: Alerts checked - {len(triggered_alerts)} triggered")
    
    # 5. Calibrate confidence
    calibrator = ConfidenceCalibrator()
    # Mock training data
    import numpy as np
    np.random.seed(42)
    mock_conf = np.random.beta(5, 2, 100).tolist()
    mock_out = (np.random.rand(100) < (np.array(mock_conf) * 0.8)).tolist()
    calibrator.fit(mock_conf, mock_out)
    
    calibrated_conf = calibrator.transform([decision["confidence"]])[0]
    print(f"✓ Step 5: Confidence calibrated - {decision['confidence']:.2f} → {calibrated_conf:.2f}")
    
    print("✅ Full Stack Integration: PASS\n")
    return True


def test_performance_benchmarks():
    """Test #6: Performance Benchmarks"""
    print("="*60)
    print("TEST #6: Performance Benchmarks")
    print("="*60)
    
    import time
    
    # Benchmark risk evaluation
    engine = RiskEngine()
    trade = {
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 100,
        "price": 150.0,
        "confidence": 0.85,
        "volatility": 0.25,
    }
    portfolio = {"total_value": 100000.0, "peak_value": 120000.0}
    
    start = time.time()
    for _ in range(100):
        engine.evaluate_trade(trade, portfolio)
    duration = time.time() - start
    avg_time = duration / 100 * 1000  # ms
    
    print(f"✓ Risk evaluation: {avg_time:.2f}ms avg (100 iterations)")
    assert avg_time < 10, "Risk evaluation should be < 10ms"
    
    # Benchmark alert evaluation
    evaluator = AlertEvaluator()
    
    start = time.time()
    for _ in range(100):
        evaluator.evaluate_condition(155.0, "above", 150.0, "bench_001")
    duration = time.time() - start
    avg_time = duration / 100 * 1000  # ms
    
    print(f"✓ Alert evaluation: {avg_time:.2f}ms avg (100 iterations)")
    assert avg_time < 5, "Alert evaluation should be < 5ms"
    
    print("✅ Performance Benchmarks: PASS\n")
    return True


def main():
    """Run all acceptance tests"""
    print("\n" + "="*60)
    print("PHASE A3 COMPLETE - ACCEPTANCE TESTS")
    print("="*60)
    print("Testing production readiness of v1.0.0\n")
    
    tests = [
        ("Explainability System", test_explainability_system),
        ("Alerts & Watchlist System", test_alerts_watchlist_system),
        ("Risk Management System", test_risk_management_system),
        ("Calibration System", test_calibration_system),
        ("Full Stack Integration", test_full_stack_integration),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}: FAILED")
            print(f"   Error: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - PRODUCTION READY! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
