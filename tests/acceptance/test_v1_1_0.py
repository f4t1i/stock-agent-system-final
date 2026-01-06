#!/usr/bin/env python3
"""
Acceptance Tests - v1.1.0 Features

Tests new features in v1.1.0:
1. OpenBB market data integration
2. WebSocket real-time updates
3. PostgreSQL database persistence

Note: Some tests require running services (API server, PostgreSQL)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Test 1: Database Models
# ============================================================================

def test_database_models():
    """Test SQLAlchemy models can be imported and tables created"""
    print("\n" + "=" * 80)
    print("TEST 1: Database Models")
    print("=" * 80)

    try:
        from database.models import (
            Analysis,
            TrainingRun,
            ModelVersion,
            ExperienceRecord,
            Alert,
            Watchlist,
            Decision,
            RiskViolation,
            Base
        )

        print(f"\n✓ Imported 8 database models")

        # Check tables
        tables = [table.name for table in Base.metadata.sorted_tables]
        print(f"\n✓ Defined {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")

        expected_tables = [
            'analyses', 'training_runs', 'model_versions', 'experiences',
            'alerts', 'watchlists', 'decisions', 'risk_violations'
        ]

        assert len(tables) == len(expected_tables)

        print(f"\n✅ Database models test passed")
        return True

    except Exception as e:
        print(f"\n❌ Database models test failed: {e}")
        return False


# ============================================================================
# Test 2: Database Configuration
# ============================================================================

def test_database_config():
    """Test database configuration and connection"""
    print("\n" + "=" * 80)
    print("TEST 2: Database Configuration")
    print("=" * 80)

    try:
        from database.config import engine, SessionLocal, DATABASE_URL

        print(f"\n✓ Database URL: {DATABASE_URL}")
        print(f"✓ Engine: {engine.name}")

        # Test connection
        with engine.connect() as conn:
            print(f"✓ Database connection successful")

        # Test session
        db = SessionLocal()
        db.close()
        print(f"✓ Session creation successful")

        print(f"\n✅ Database configuration test passed")
        return True

    except Exception as e:
        print(f"\n❌ Database configuration test failed: {e}")
        return False


# ============================================================================
# Test 3: OpenBB API Integration
# ============================================================================

def test_openbb_api():
    """Test OpenBB API endpoints (mock data)"""
    print("\n" + "=" * 80)
    print("TEST 3: OpenBB API Integration")
    print("=" * 80)

    try:
        from api.openbb import (
            generate_price_data,
            generate_fundamentals,
            generate_news,
            generate_technical,
            router
        )

        # Test price data generation
        price = generate_price_data("AAPL")
        print(f"\n✓ Price data generated:")
        print(f"  Symbol: {price.symbol}")
        print(f"  Price: ${price.price:.2f}")
        print(f"  Change: {price.change:+.2f} ({price.change_percent:+.2f}%)")

        # Test fundamentals
        fundamentals = generate_fundamentals("AAPL")
        print(f"\n✓ Fundamental data generated:")
        print(f"  Market Cap: ${fundamentals.market_cap / 1e9:.1f}B")
        print(f"  P/E Ratio: {fundamentals.pe_ratio:.2f}")

        # Test news
        news = generate_news("AAPL", limit=3)
        print(f"\n✓ News generated: {len(news)} items")

        # Test technical
        technical = generate_technical("AAPL")
        print(f"\n✓ Technical indicators generated:")
        print(f"  RSI: {technical.rsi:.2f}")
        print(f"  MACD: {technical.macd:.2f}")

        # Test router
        print(f"\n✓ OpenBB router has {len(router.routes)} routes")

        print(f"\n✅ OpenBB API integration test passed")
        return True

    except Exception as e:
        print(f"\n❌ OpenBB API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 4: WebSocket Integration
# ============================================================================

def test_websocket_integration():
    """Test WebSocket module can be imported"""
    print("\n" + "=" * 80)
    print("TEST 4: WebSocket Integration")
    print("=" * 80)

    try:
        from api.websocket import (
            ConnectionManager,
            manager,
            MessageType,
            Channel
        )

        print(f"\n✓ WebSocket module imported")

        # Check message types
        print(f"\n✓ Message types: {len(MessageType)} types")
        print(f"  Client → Server: {MessageType.SUBSCRIBE}, {MessageType.PING}")
        print(f"  Server → Client: {MessageType.PRICE_UPDATE}, {MessageType.ALERT_TRIGGERED}")

        # Check channels
        print(f"\n✓ Channels: {len(Channel)} channels")
        for channel in Channel:
            print(f"  - {channel.value}")

        # Check manager
        stats = manager.get_stats()
        print(f"\n✓ Connection manager initialized")
        print(f"  Active connections: {stats['total_connections']}")

        print(f"\n✅ WebSocket integration test passed")
        return True

    except Exception as e:
        print(f"\n❌ WebSocket integration test failed: {e}")
        return False


# ============================================================================
# Test 5: Docker Configuration
# ============================================================================

def test_docker_config():
    """Test Docker Compose configuration exists"""
    print("\n" + "=" * 80)
    print("TEST 5: Docker Configuration")
    print("=" * 80)

    try:
        docker_file = project_root / "docker-compose.postgres.yml"

        if docker_file.exists():
            print(f"\n✓ Docker Compose file exists: {docker_file}")

            # Read and validate
            import yaml
            with open(docker_file) as f:
                config = yaml.safe_load(f)

            services = config.get('services', {})
            print(f"\n✓ Services defined: {list(services.keys())}")

            assert 'postgres' in services
            print(f"  - PostgreSQL configured")

            if 'pgadmin' in services:
                print(f"  - PgAdmin configured (optional)")

            print(f"\n✅ Docker configuration test passed")
            return True
        else:
            print(f"\n⚠  Docker Compose file not found (optional)")
            return True

    except Exception as e:
        print(f"\n❌ Docker configuration test failed: {e}")
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all v1.1.0 acceptance tests"""
    print("\n" + "=" * 80)
    print("V1.1.0 ACCEPTANCE TESTS")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Database Models", test_database_models()))
    results.append(("Database Config", test_database_config()))
    results.append(("OpenBB API", test_openbb_api()))
    results.append(("WebSocket", test_websocket_integration()))
    results.append(("Docker Config", test_docker_config()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
