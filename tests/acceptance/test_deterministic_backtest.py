"""
Acceptance Test: Deterministic Backtest

Tests:
1. Make backtest works (one-click execution)
2. Deterministic results (same seed → same results)
3. Signal validation integration
4. Reproducible backtest report

Acceptance Criteria:
✅ make backtest executes without errors
✅ Same config + same seed → identical results
✅ Backtest report generated with expected format
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.backtester_v2 import BacktesterV2, BacktestConfig
from loguru import logger


class TestDeterministicBacktest:
    """Acceptance tests for deterministic backtest"""

    def test_make_backtest_command(self):
        """Test 1: make backtest works"""
        logger.info("TEST 1: Testing 'make backtest' command...")

        # Note: This would require actual make execution in CI
        # For now, we test the Python script directly
        result = subprocess.run(
            ['python', 'scripts/run_backtest.py', '--symbols', 'AAPL', '--start', '2024-01-01', '--end', '2024-01-31'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        assert result.returncode == 0, f"Backtest failed with error: {result.stderr}"
        logger.success("✅ TEST 1 PASSED: make backtest executes successfully")

    def test_deterministic_results(self):
        """Test 2: Same seed produces identical results"""
        logger.info("TEST 2: Testing deterministic results...")

        # Create identical configs
        config1 = BacktestConfig(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=100000,
            random_seed=42,
            enable_survivorship_bias_guard=False,
            enable_corporate_actions=False,
            fail_fast_on_missing_data=False,
            validate_signals=False
        )

        config2 = BacktestConfig(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=100000,
            random_seed=42,  # Same seed
            enable_survivorship_bias_guard=False,
            enable_corporate_actions=False,
            fail_fast_on_missing_data=False,
            validate_signals=False
        )

        # Run backtest 1
        backtester1 = BacktesterV2(config1)
        results1 = backtester1.run()

        # Run backtest 2
        backtester2 = BacktesterV2(config2)
        results2 = backtester2.run()

        # Compare results
        assert results1['final_value'] == results2['final_value'], \
            f"Final values differ: {results1['final_value']} vs {results2['final_value']}"

        assert results1['num_trades'] == results2['num_trades'], \
            f"Trade counts differ: {results1['num_trades']} vs {results2['num_trades']}"

        assert len(backtester1.equity_curve) == len(backtester2.equity_curve), \
            "Equity curves have different lengths"

        logger.success("✅ TEST 2 PASSED: Deterministic results verified (same seed → same output)")

    def test_different_seeds_produce_different_results(self):
        """Test 3: Different seeds produce different results"""
        logger.info("TEST 3: Testing different seeds produce different results...")

        config1 = BacktestConfig(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=100000,
            random_seed=42
        )

        config2 = BacktestConfig(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=100000,
            random_seed=123  # Different seed
        )

        backtester1 = BacktesterV2(config1)
        results1 = backtester1.run()

        backtester2 = BacktesterV2(config2)
        results2 = backtester2.run()

        # Different seeds should produce different results (with high probability)
        # Note: There's a small chance they could be the same, but extremely unlikely
        assert results1['final_value'] != results2['final_value'] or \
               results1['num_trades'] != results2['num_trades'], \
            "Different seeds should produce different results"

        logger.success("✅ TEST 3 PASSED: Different seeds produce different results")

    def test_backtest_config_validation(self):
        """Test 4: Config validation catches invalid configs"""
        logger.info("TEST 4: Testing config validation...")

        # Test invalid date range
        config_bad_dates = BacktestConfig(
            symbols=['AAPL'],
            start_date='2024-12-31',
            end_date='2024-01-01',  # End before start
            initial_capital=100000
        )

        is_valid, errors = config_bad_dates.validate()
        assert not is_valid, "Invalid date range should fail validation"
        assert len(errors) > 0, "Should have validation errors"

        logger.info(f"  Caught expected error: {errors[0]}")

        # Test invalid symbols
        config_bad_symbols = BacktestConfig(
            symbols=['aapl'],  # Lowercase - invalid
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=100000
        )

        is_valid, errors = config_bad_symbols.validate()
        assert not is_valid, "Invalid symbol format should fail validation"

        logger.info(f"  Caught expected error: {errors[0]}")

        logger.success("✅ TEST 4 PASSED: Config validation works correctly")

    def test_output_files_generated(self):
        """Test 5: Backtest generates expected output files"""
        logger.info("TEST 5: Testing output file generation...")

        config = BacktestConfig(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_capital=100000,
            random_seed=42,
            output_dir='test_backtest_results',
            save_trades=True,
            save_signals=False
        )

        backtester = BacktesterV2(config)
        results = backtester.run()

        # Check results directory exists
        output_dir = Path('test_backtest_results')
        assert output_dir.exists(), "Output directory should exist"

        # Check metrics file exists
        metrics_files = list(output_dir.glob('backtest_metrics_*.json'))
        assert len(metrics_files) > 0, "Should have generated metrics file"

        # Check trades file exists
        trades_files = list(output_dir.glob('backtest_trades_*.json'))
        assert len(trades_files) > 0, "Should have generated trades file"

        # Validate metrics file content
        with open(metrics_files[0], 'r') as f:
            metrics = json.load(f)

        assert 'final_value' in metrics, "Metrics should contain final_value"
        assert 'total_return' in metrics, "Metrics should contain total_return"
        assert 'num_trades' in metrics, "Metrics should contain num_trades"

        logger.success("✅ TEST 5 PASSED: Output files generated correctly")

        # Cleanup
        import shutil
        shutil.rmtree(output_dir)


def run_all_tests():
    """Run all acceptance tests"""
    test_suite = TestDeterministicBacktest()

    print("\n" + "="*60)
    print("ACCEPTANCE TESTS: Deterministic Backtest")
    print("="*60 + "\n")

    tests = [
        ("Make Backtest Command", test_suite.test_make_backtest_command),
        ("Deterministic Results", test_suite.test_deterministic_results),
        ("Different Seeds", test_suite.test_different_seeds_produce_different_results),
        ("Config Validation", test_suite.test_backtest_config_validation),
        ("Output Files", test_suite.test_output_files_generated),
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
