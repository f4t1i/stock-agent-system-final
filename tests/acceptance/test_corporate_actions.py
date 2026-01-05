"""
Acceptance Test: Corporate Actions Handler

Tests:
1. Stock split detection from price data
2. Position adjustment for splits
3. Dividend payment handling
4. AAPL 2020 split test (real-world example)

Acceptance Criteria:
✅ Auto-detect splits from price data
✅ AAPL 4:1 split → position adjusted correctly
✅ Dividend payment → cash adjusted correctly
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.backtester_v2 import CorporateActionsHandler, CorporateAction
from loguru import logger


class TestCorporateActionsHandler:
    """Acceptance tests for corporate actions handler"""

    def test_stock_split_detection(self):
        """Test 1: Auto-detect stock split from price data"""
        logger.info("TEST 1: Testing stock split auto-detection...")

        handler = CorporateActionsHandler()

        # Create price data with 2:1 split (50% price drop)
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        prices = [200.0] * 15 + [100.0] * (len(dates) - 15)  # Split on day 15

        price_data = pd.DataFrame({
            'Close': prices,
            'Volume': [1000000] * len(dates)
        }, index=dates)

        # Detect splits
        handler.detect_splits_from_data('TEST', price_data)

        # Should have detected one split
        actions = handler.actions
        assert len(actions) > 0, "Should have detected at least one split"

        split_actions = [a for a in actions if a.action_type == 'split']
        assert len(split_actions) > 0, "Should have detected split action"

        split = split_actions[0]
        assert split.ratio == 2.0, f"Should detect 2:1 split, got {split.ratio}"

        logger.success(f"✅ TEST 1 PASSED: Detected 2:1 split on {split.date}")

    def test_position_adjustment_for_split(self):
        """Test 2: Position quantity adjusted correctly for split"""
        logger.info("TEST 2: Testing position adjustment for split...")

        handler = CorporateActionsHandler()

        # Initial position: 100 shares
        initial_quantity = 100.0

        # 4:1 split (like AAPL 2020)
        split_ratio = 4.0

        adjusted_quantity = handler.adjust_position_for_split('AAPL', initial_quantity, split_ratio)

        expected_quantity = 100.0 * 4.0  # 400 shares
        assert adjusted_quantity == expected_quantity, \
            f"Expected {expected_quantity}, got {adjusted_quantity}"

        logger.success(f"✅ TEST 2 PASSED: Position adjusted {initial_quantity} → {adjusted_quantity} shares")

    def test_dividend_payment(self):
        """Test 3: Dividend payment added to cash"""
        logger.info("TEST 3: Testing dividend payment...")

        handler = CorporateActionsHandler()

        # Hold 1000 shares, $2.50 dividend per share
        quantity = 1000.0
        dividend_per_share = 2.50
        initial_cash = 10000.0

        adjusted_cash = handler.adjust_cash_for_dividend(
            'AAPL',
            quantity,
            dividend_per_share,
            initial_cash
        )

        expected_payment = 1000.0 * 2.50  # $2,500
        expected_cash = initial_cash + expected_payment

        assert adjusted_cash == expected_cash, \
            f"Expected ${expected_cash}, got ${adjusted_cash}"

        logger.success(f"✅ TEST 3 PASSED: Dividend payment ${expected_payment} added to cash")

    def test_aapl_2020_split(self):
        """Test 4: AAPL 4:1 split (Aug 31, 2020)"""
        logger.info("TEST 4: Testing AAPL 2020 4:1 split...")

        handler = CorporateActionsHandler()

        # Simulate AAPL price before/after 4:1 split
        # Before split: ~$400-500, After split: ~$100-125
        dates = pd.date_range(start='2020-08-01', end='2020-09-30', freq='D')

        # Price drops ~75% on split date (Aug 31, 2020)
        prices = []
        for date in dates:
            if date < pd.Timestamp('2020-08-31'):
                prices.append(450.0 + np.random.randn() * 10)  # Pre-split
            else:
                prices.append(112.5 + np.random.randn() * 2.5)  # Post-split (450/4)

        price_data = pd.DataFrame({
            'Close': prices,
            'Volume': [1000000] * len(dates)
        }, index=dates)

        # Detect split
        handler.detect_splits_from_data('AAPL', price_data)

        # Should detect 4:1 split
        split_actions = [a for a in handler.actions if a.action_type == 'split' and a.symbol == 'AAPL']

        assert len(split_actions) > 0, "Should detect AAPL split"

        split = split_actions[0]
        assert split.ratio == 4.0, f"Should detect 4:1 split, got {split.ratio}:1"
        assert '2020-08' in split.date, f"Split should be in August 2020, got {split.date}"

        # Test position adjustment
        pre_split_shares = 25  # 25 shares @ $450 = $11,250
        post_split_shares = handler.adjust_position_for_split('AAPL', pre_split_shares, split.ratio)

        assert post_split_shares == 100.0, \
            f"25 shares should become 100 shares after 4:1 split, got {post_split_shares}"

        logger.success(f"✅ TEST 4 PASSED: AAPL 2020 4:1 split detected and adjusted correctly")

    def test_get_actions_for_date(self):
        """Test 5: Retrieve actions for specific date"""
        logger.info("TEST 5: Testing get actions for specific date...")

        handler = CorporateActionsHandler()

        # Add multiple actions
        split_action = CorporateAction(
            symbol='TEST',
            date='2024-01-15',
            action_type='split',
            ratio=2.0
        )

        dividend_action = CorporateAction(
            symbol='TEST',
            date='2024-01-15',
            action_type='dividend',
            amount=1.50
        )

        handler.add_action(split_action)
        handler.add_action(dividend_action)

        # Get actions for that date
        actions = handler.get_actions_for_date('TEST', datetime(2024, 1, 15))

        assert len(actions) == 2, f"Should have 2 actions, got {len(actions)}"

        action_types = [a.action_type for a in actions]
        assert 'split' in action_types and 'dividend' in action_types, \
            "Should have both split and dividend actions"

        # Get actions for different date (no actions)
        no_actions = handler.get_actions_for_date('TEST', datetime(2024, 2, 1))
        assert len(no_actions) == 0, "Should have no actions for different date"

        logger.success("✅ TEST 5 PASSED: Actions retrieved correctly for specific date")


def run_all_tests():
    """Run all corporate actions tests"""
    test_suite = TestCorporateActionsHandler()

    print("\n" + "="*60)
    print("ACCEPTANCE TESTS: Corporate Actions Handler")
    print("="*60 + "\n")

    tests = [
        ("Stock Split Detection", test_suite.test_stock_split_detection),
        ("Position Adjustment", test_suite.test_position_adjustment_for_split),
        ("Dividend Payment", test_suite.test_dividend_payment),
        ("AAPL 2020 4:1 Split", test_suite.test_aapl_2020_split),
        ("Get Actions for Date", test_suite.test_get_actions_for_date),
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
