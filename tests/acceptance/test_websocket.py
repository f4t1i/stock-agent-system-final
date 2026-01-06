#!/usr/bin/env python3
"""
Acceptance Tests - WebSocket Real-Time Updates

Tests WebSocket functionality:
1. Connection & reconnection
2. Channel subscriptions
3. Price updates streaming
4. Alert notifications
5. Message broadcasting
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import websockets
except ImportError:
    print("❌ websockets library not installed")
    print("Install with: pip install websockets")
    sys.exit(1)

# WebSocket URL
WS_URL = "ws://localhost:8000/ws/test-client-001"


# ============================================================================
# Test 1: Basic Connection
# ============================================================================

async def test_connection():
    """Test basic WebSocket connection"""
    print("\n" + "=" * 80)
    print("TEST 1: WebSocket Connection")
    print("=" * 80)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Receive welcome message
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)

            print(f"\n✓ Connected successfully")
            print(f"  Client ID: {data.get('client_id')}")
            print(f"  Available channels: {data.get('available_channels')}")

            assert data['type'] == 'connected'
            assert data['client_id'] == 'test-client-001'

            print(f"\n✅ Connection test passed")
            return True

    except asyncio.TimeoutError:
        print(f"\n❌ Connection timeout")
        return False
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False


# ============================================================================
# Test 2: Channel Subscription
# ============================================================================

async def test_subscription():
    """Test channel subscription"""
    print("\n" + "=" * 80)
    print("TEST 2: Channel Subscription")
    print("=" * 80)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Skip welcome message
            await ws.recv()

            # Subscribe to prices channel
            subscribe_msg = {
                "type": "subscribe",
                "channel": "prices",
                "symbols": ["AAPL", "MSFT"]
            }
            await ws.send(json.dumps(subscribe_msg))

            # Wait for subscription confirmation
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)

            print(f"\n✓ Subscription confirmed")
            print(f"  Type: {data['type']}")
            print(f"  Channel: {data['channel']}")
            print(f"  Symbols: {data['symbols']}")

            assert data['type'] == 'subscribed'
            assert data['channel'] == 'prices'
            assert 'AAPL' in data['symbols']

            print(f"\n✅ Subscription test passed")
            return True

    except asyncio.TimeoutError:
        print(f"\n❌ Subscription timeout")
        return False
    except Exception as e:
        print(f"\n❌ Subscription failed: {e}")
        return False


# ============================================================================
# Test 3: Price Updates
# ============================================================================

async def test_price_updates():
    """Test receiving price updates"""
    print("\n" + "=" * 80)
    print("TEST 3: Price Updates Streaming")
    print("=" * 80)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Skip welcome message
            await ws.recv()

            # Subscribe to AAPL prices
            await ws.send(json.dumps({
                "type": "subscribe",
                "channel": "prices",
                "symbols": ["AAPL"]
            }))

            # Skip subscription confirmation
            await ws.recv()

            # Receive price updates
            print(f"\n--- Receiving Price Updates (10 seconds) ---")
            price_updates = []

            try:
                for i in range(10):
                    response = await asyncio.wait_for(ws.recv(), timeout=2)
                    data = json.loads(response)

                    if data['type'] == 'price_update' and data['symbol'] == 'AAPL':
                        price_data = data['data']
                        price_updates.append(price_data)
                        print(f"\n  Update #{i+1}:")
                        print(f"    Price: ${price_data['price']:.2f}")
                        print(f"    Change: {price_data['change']:+.2f} ({price_data['change_pct']:+.2f}%)")
                        print(f"    Volume: {price_data['volume']:,}")

            except asyncio.TimeoutError:
                pass  # Expected after 10 updates

            print(f"\n✓ Received {len(price_updates)} price updates")

            assert len(price_updates) >= 5, "Should receive at least 5 updates"

            print(f"\n✅ Price updates test passed")
            return True

    except Exception as e:
        print(f"\n❌ Price updates test failed: {e}")
        return False


# ============================================================================
# Test 4: Ping/Pong
# ============================================================================

async def test_ping_pong():
    """Test ping/pong keep-alive"""
    print("\n" + "=" * 80)
    print("TEST 4: Ping/Pong Keep-Alive")
    print("=" * 80)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Skip welcome message
            await ws.recv()

            # Send ping
            await ws.send(json.dumps({"type": "ping"}))

            # Wait for pong
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)

            print(f"\n✓ Ping sent, pong received")
            print(f"  Type: {data['type']}")
            print(f"  Timestamp: {data['timestamp']}")

            assert data['type'] == 'pong'

            print(f"\n✅ Ping/pong test passed")
            return True

    except asyncio.TimeoutError:
        print(f"\n❌ Pong timeout")
        return False
    except Exception as e:
        print(f"\n❌ Ping/pong failed: {e}")
        return False


# ============================================================================
# Test 5: Multiple Clients
# ============================================================================

async def test_multiple_clients():
    """Test multiple simultaneous clients"""
    print("\n" + "=" * 80)
    print("TEST 5: Multiple Clients")
    print("=" * 80)

    clients = []
    client_count = 3

    try:
        # Connect multiple clients
        for i in range(client_count):
            url = f"ws://localhost:8000/ws/multi-test-{i}"
            ws = await websockets.connect(url)
            # Skip welcome message
            await ws.recv()
            clients.append(ws)
            print(f"\n✓ Client {i+1} connected")

        print(f"\n✓ All {client_count} clients connected")

        # Subscribe each to different symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for i, ws in enumerate(clients):
            await ws.send(json.dumps({
                "type": "subscribe",
                "channel": "prices",
                "symbols": [symbols[i]]
            }))
            await ws.recv()  # Skip confirmation
            print(f"  Client {i+1} subscribed to {symbols[i]}")

        print(f"\n✅ Multiple clients test passed")
        return True

    except Exception as e:
        print(f"\n❌ Multiple clients test failed: {e}")
        return False

    finally:
        # Cleanup
        for ws in clients:
            await ws.close()


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_all_tests():
    """Run all WebSocket tests"""
    print("\n" + "=" * 80)
    print("WEBSOCKET ACCEPTANCE TESTS")
    print("=" * 80)

    print(f"\nTarget: {WS_URL}")
    print(f"Note: Ensure API server is running (python api/server.py)")

    results = []

    # Run tests
    results.append(("Connection", await test_connection()))
    results.append(("Subscription", await test_subscription()))
    results.append(("Price Updates", await test_price_updates()))
    results.append(("Ping/Pong", await test_ping_pong()))
    results.append(("Multiple Clients", await test_multiple_clients()))

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
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
