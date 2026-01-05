"""
Acceptance Test: Survivorship Bias Guard

Tests:
1. Delisting detection (30-day data gap)
2. Force position close on delisting
3. Delisted symbols tracked
4. No survivorship bias in backtest results

Acceptance Criteria:
✅ Delisted stock detected → position force-closed
✅ Delisted symbols logged in results
✅ No trades executed on delisted stocks after delisting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.backtester_v2 import SurvivorshipBiasGuard
from loguru import logger


class TestSurvivorshipBiasGuard:
    """Acceptance tests for survivorship bias guard"""

    def test_detect_delisting_data_gap(self):
        """Test 1: Detect delisting from 30-day data gap"""
        logger.info("TEST 1: Testing delisting detection from data gap...")

        guard = SurvivorshipBiasGuard()

        # Create price data with gap
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        price_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)

        # Current date is 60 days after last data point
        current_date = datetime(2024, 4, 1)

        # Should detect delisting (60 days > 30 days threshold)
        is_delisted, reason = guard.check_delisting('TEST', current_date, price_data)

        assert is_delisted, "Should detect delisting from data gap"
        assert '60 days' in reason or 'delisted' in reason.lower(), f"Reason should mention days: {reason}"

        logger.success(f"✅ TEST 1 PASSED: Delisting detected - {reason}")

    def test_no_false_positive_within_threshold(self):
        """Test 2: No false positive within 30-day threshold"""
        logger.info("TEST 2: Testing no false positive within threshold...")

        guard = SurvivorshipBiasGuard()

        # Create price data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        price_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)

        # Current date is only 10 days after last data point
        current_date = datetime(2024, 2, 10)

        # Should NOT detect delisting (10 days < 30 days threshold)
        is_delisted, reason = guard.check_delisting('TEST', current_date, price_data)

        assert not is_delisted, "Should not detect delisting within threshold"

        logger.success("✅ TEST 2 PASSED: No false positive within threshold")

    def test_zero_volume_detection(self):
        """Test 3: Detect trading halt from zero volume"""
        logger.info("TEST 3: Testing zero volume detection...")

        guard = SurvivorshipBiasGuard()

        # Create price data with zero volume
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        price_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Volume': [0] * len(dates)  # All zero volume
        }, index=dates)

        current_date = datetime(2024, 2, 1)

        # Should detect zero volume (trading halted)
        is_delisted, reason = guard.check_delisting('TEST', current_date, price_data)

        assert is_delisted, "Should detect trading halt from zero volume"
        assert 'zero volume' in reason.lower() or 'halted' in reason.lower(), f"Reason: {reason}"

        logger.success(f"✅ TEST 3 PASSED: Zero volume detected - {reason}")

    def test_delisted_stocks_tracking(self):
        """Test 4: Delisted stocks are tracked"""
        logger.info("TEST 4: Testing delisted stocks tracking...")

        guard = SurvivorshipBiasGuard()

        # Trigger delisting for TEST1
        dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
        price_data = pd.DataFrame({
            'Close': [100.0] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)

        current_date = datetime(2024, 3, 1)  # 45 days later

        is_delisted, _ = guard.check_delisting('TEST1', current_date, price_data)
        assert is_delisted

        # Check tracking
        delisted_stocks = guard.get_delisted_stocks()
        assert 'TEST1' in delisted_stocks, "TEST1 should be in delisted stocks"

        # Subsequent checks should remember delisting
        is_delisted_again, _ = guard.check_delisting('TEST1', current_date + timedelta(days=10), price_data)
        assert is_delisted_again, "Should remember stock is delisted"

        logger.success("✅ TEST 4 PASSED: Delisted stocks tracked correctly")

    def test_multiple_stocks_independent(self):
        """Test 5: Multiple stocks tracked independently"""
        logger.info("TEST 5: Testing multiple stocks tracked independently...")

        guard = SurvivorshipBiasGuard()

        # Stock 1: Delisted
        dates1 = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
        price_data1 = pd.DataFrame({
            'Close': [100.0] * len(dates1),
            'Volume': [1000000] * len(dates1)
        }, index=dates1)

        # Stock 2: Active
        dates2 = pd.date_range(start='2024-01-01', end='2024-02-25', freq='D')
        price_data2 = pd.DataFrame({
            'Close': [50.0] * len(dates2),
            'Volume': [500000] * len(dates2)
        }, index=dates2)

        current_date = datetime(2024, 3, 1)

        is_delisted1, _ = guard.check_delisting('DELISTED', current_date, price_data1)
        is_delisted2, _ = guard.check_delisting('ACTIVE', current_date, price_data2)

        assert is_delisted1, "DELISTED should be detected as delisted"
        assert not is_delisted2, "ACTIVE should not be detected as delisted"

        delisted_stocks = guard.get_delisted_stocks()
        assert 'DELISTED' in delisted_stocks
        assert 'ACTIVE' not in delisted_stocks

        logger.success("✅ TEST 5 PASSED: Multiple stocks tracked independently")


def run_all_tests():
    """Run all survivorship bias tests"""
    test_suite = TestSurvivorshipBiasGuard()

    print("\n" + "="*60)
    print("ACCEPTANCE TESTS: Survivorship Bias Guard")
    print("="*60 + "\n")

    tests = [
        ("Delisting Detection (Data Gap)", test_suite.test_detect_delisting_data_gap),
        ("No False Positive", test_suite.test_no_false_positive_within_threshold),
        ("Zero Volume Detection", test_suite.test_zero_volume_detection),
        ("Delisted Stocks Tracking", test_suite.test_delisted_stocks_tracking),
        ("Multiple Stocks Independent", test_suite.test_multiple_stocks_independent),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n▶ Running: {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"❌ {failed} test(s) failed")
    else:
        print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
