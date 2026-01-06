# WebSocket Real-Time Updates

**Version**: v1.1.0-dev
**Status**: ✅ Implemented
**Last Updated**: 2026-01-06

## Overview

The Stock Agent System now supports WebSocket connections for real-time updates, eliminating the need for polling and providing instant notifications for price changes, alerts, trading signals, and more.

## Features

✅ **Real-Time Price Streaming** - Live market data updates (simulated in dev, ready for real feeds)
✅ **Alert Notifications** - Instant push notifications when alerts are triggered
✅ **Trading Signals** - Real-time trading recommendations from agents
✅ **Portfolio Updates** - Live portfolio value and position changes
✅ **System Notifications** - Status updates and system events
✅ **Automatic Reconnection** - Exponential backoff with configurable retry limits
✅ **Channel-Based Subscriptions** - Subscribe only to the data you need
✅ **Multiple Clients** - Support for concurrent connections
✅ **Type-Safe** - Full TypeScript support on frontend

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  React Client (Browser)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  useWebSocket Hook                                │  │
│  │  - Connection management                          │  │
│  │  - Automatic reconnection                         │  │
│  │  - Event subscriptions                            │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ WebSocket (ws://localhost:8000)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend + WebSocket Server          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ConnectionManager                                │  │
│  │  - Client connections                             │  │
│  │  - Channel subscriptions                          │  │
│  │  - Message broadcasting                           │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Background Tasks                                 │  │
│  │  - Price streaming (1s interval)                  │  │
│  │  - Alert evaluation                               │  │
│  │  - Portfolio monitoring                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Backend API

### WebSocket Endpoint

```
ws://localhost:8000/ws/{client_id}
```

**Parameters**:
- `client_id` (string): Unique identifier for the client connection

### Message Types

#### Client → Server

**Subscribe to Channel**
```json
{
  "type": "subscribe",
  "channel": "prices",
  "symbols": ["AAPL", "MSFT"]  // Optional, for prices channel
}
```

**Unsubscribe from Channel**
```json
{
  "type": "unsubscribe",
  "channel": "prices"
}
```

**Ping (Keep-Alive)**
```json
{
  "type": "ping"
}
```

#### Server → Client

**Price Update**
```json
{
  "type": "price_update",
  "symbol": "AAPL",
  "data": {
    "price": 185.50,
    "change": 2.30,
    "change_pct": 1.26,
    "volume": 45000000,
    "bid": 185.45,
    "ask": 185.55
  },
  "timestamp": "2026-01-06T12:00:00Z"
}
```

**Alert Triggered**
```json
{
  "type": "alert_triggered",
  "data": {
    "id": "alert-123",
    "symbol": "AAPL",
    "type": "price_threshold",
    "message": "AAPL crossed $185",
    "severity": "info"
  },
  "timestamp": "2026-01-06T12:00:00Z"
}
```

**Trading Signal**
```json
{
  "type": "trading_signal",
  "data": {
    "symbol": "AAPL",
    "action": "buy",
    "confidence": 0.85,
    "reasoning": "Strong bullish momentum..."
  },
  "timestamp": "2026-01-06T12:00:00Z"
}
```

**Notification**
```json
{
  "type": "notification",
  "data": {
    "id": "notif-456",
    "message": "Model training completed",
    "type": "success"
  },
  "timestamp": "2026-01-06T12:00:00Z"
}
```

**Pong (Keep-Alive Response)**
```json
{
  "type": "pong",
  "timestamp": "2026-01-06T12:00:00Z"
}
```

**Error**
```json
{
  "type": "error",
  "error": "Unknown channel: invalid_channel",
  "timestamp": "2026-01-06T12:00:00Z"
}
```

### Available Channels

| Channel | Description | Subscribe Options |
|---------|-------------|-------------------|
| `prices` | Real-time price updates | `symbols`: List of symbols |
| `alerts` | Alert notifications | None |
| `signals` | Trading signals | None |
| `portfolio` | Portfolio updates | None |
| `notifications` | System notifications | None |

### REST API

**Get WebSocket Stats**
```bash
GET /ws/stats
```

Response:
```json
{
  "total_connections": 3,
  "clients": ["client-1", "client-2", "client-3"],
  "channel_subscriptions": {
    "client-1": ["prices", "alerts"],
    "client-2": ["signals"]
  },
  "symbol_subscriptions": {
    "client-1": ["AAPL", "MSFT"]
  }
}
```

## Frontend Usage

### React Hooks

#### useWebSocket (Main Hook)

```typescript
import { useWebSocket } from '@/hooks/useWebSocket';

function MyComponent() {
  const ws = useWebSocket();

  useEffect(() => {
    if (ws.isConnected) {
      // Subscribe to prices
      ws.subscribe(Channel.PRICES, ['AAPL', 'MSFT']);

      // Listen for price updates
      const cleanup = ws.onMessage(MessageType.PRICE_UPDATE, (msg) => {
        console.log('Price update:', msg);
      });

      return cleanup;
    }
  }, [ws.isConnected]);

  return (
    <div>
      Status: {ws.connectionState}
      {ws.canReconnect && (
        <button onClick={ws.reconnect}>Reconnect</button>
      )}
    </div>
  );
}
```

#### Specialized Hooks

**usePriceUpdates** - Real-time price streaming
```typescript
import { usePriceUpdates } from '@/hooks/useWebSocket';

function PriceDisplay() {
  const { prices, isConnected } = usePriceUpdates(['AAPL', 'MSFT']);

  return (
    <div>
      {Array.from(prices.values()).map((price) => (
        <div key={price.symbol}>
          {price.symbol}: ${price.price.toFixed(2)}
          <span className={price.change >= 0 ? 'text-green-500' : 'text-red-500'}>
            {price.change_pct >= 0 ? '+' : ''}{price.change_pct.toFixed(2)}%
          </span>
        </div>
      ))}
    </div>
  );
}
```

**useAlertNotifications** - Alert notifications
```typescript
import { useAlertNotifications } from '@/hooks/useWebSocket';

function AlertsPanel() {
  const { alerts, clearAlerts } = useAlertNotifications();

  return (
    <div>
      <h3>Alerts ({alerts.length})</h3>
      {alerts.map((alert) => (
        <div key={alert.id} className={`alert-${alert.severity}`}>
          {alert.message}
        </div>
      ))}
      <button onClick={clearAlerts}>Clear All</button>
    </div>
  );
}
```

**useTradingSignals** - Trading signals
```typescript
import { useTradingSignals } from '@/hooks/useWebSocket';

function SignalsPanel() {
  const { signals } = useTradingSignals();

  return (
    <div>
      {signals.map((signal, i) => (
        <div key={i}>
          {signal.symbol}: {signal.action.toUpperCase()}
          (Confidence: {(signal.confidence * 100).toFixed(0)}%)
        </div>
      ))}
    </div>
  );
}
```

**useSystemNotifications** - System notifications
```typescript
import { useSystemNotifications } from '@/hooks/useWebSocket';

function NotificationCenter() {
  const { notifications, dismissNotification, clearAll } = useSystemNotifications();

  return (
    <div>
      {notifications.map((notif) => (
        <div key={notif.id}>
          {notif.message}
          <button onClick={() => dismissNotification(notif.id)}>×</button>
        </div>
      ))}
    </div>
  );
}
```

### WebSocket Client (Low-Level)

```typescript
import { wsClient } from '@/lib/websocket';

// Connect
wsClient.connect();

// Subscribe to events
const unsubscribe = wsClient.subscribe((event) => {
  console.log('Event:', event.type, event.data);
});

// Send message
wsClient.send({ type: 'subscribe', channel: 'prices' });

// Disconnect
wsClient.disconnect();

// Cleanup
unsubscribe();
```

## Configuration

### Backend

**Connection Settings** (in `api/websocket.py`):
```python
# Reconnection settings
reconnectInterval = 2000      # Start with 2s
maxReconnectAttempts = 10     # Max 10 attempts

# Price streaming
update_interval = 1           # 1 second updates
```

### Frontend

**Environment Variables** (`.env`):
```bash
# WebSocket URL (defaults to same host on port 8000)
VITE_WS_URL=localhost:8000

# Production
VITE_WS_URL=api.yourdomain.com
```

## Testing

### Backend Tests

```bash
# Run WebSocket acceptance tests
python tests/acceptance/test_websocket.py

# Expected output:
# ✅ PASS - Connection
# ✅ PASS - Subscription
# ✅ PASS - Price Updates
# ✅ PASS - Ping/Pong
# ✅ PASS - Multiple Clients
```

### Manual Testing

```bash
# Install websockets client
pip install websockets

# Test with Python script
python -c "
import asyncio
import websockets
import json

async def test():
    async with websockets.connect('ws://localhost:8000/ws/test') as ws:
        # Receive welcome
        msg = await ws.recv()
        print('Welcome:', msg)

        # Subscribe to prices
        await ws.send(json.dumps({
            'type': 'subscribe',
            'channel': 'prices',
            'symbols': ['AAPL']
        }))

        # Receive updates
        for i in range(5):
            msg = await ws.recv()
            print('Update:', msg)

asyncio.run(test())
"
```

### Browser Testing

Open Developer Console and run:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/browser-test');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'prices',
    symbols: ['AAPL', 'MSFT']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Message:', data);
};

ws.onerror = (error) => {
  console.error('Error:', error);
};
```

## Production Deployment

### HTTPS/WSS Support

For production with HTTPS, WebSocket will automatically use WSS:

```typescript
// Frontend automatically detects protocol
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const url = `${protocol}//${host}/ws/${clientId}`;
```

### Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    # WebSocket proxy
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
```

### Load Balancing

For multiple backend instances, use sticky sessions:

```nginx
upstream websocket_backend {
    ip_hash;  # Sticky sessions
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

location /ws/ {
    proxy_pass http://websocket_backend;
    # ... other settings
}
```

## Monitoring

**Connection Stats**:
```bash
curl http://localhost:8000/ws/stats
```

**Logs**:
```bash
# Backend logs
tail -f logs/websocket.log

# Look for:
[WebSocket] Connected: client-123
[WebSocket] Client client-123 subscribed to prices
[WebSocket] Broadcasting price update: AAPL
```

## Troubleshooting

**Connection Fails**:
- Check if API server is running: `curl http://localhost:8000/health`
- Verify WebSocket URL in browser console
- Check CORS settings in `api/server.py`

**No Price Updates**:
- Ensure subscribed to `prices` channel with symbols
- Check `price_streaming_task` is running (backend logs)
- Verify connection is established (check `isConnected`)

**Frequent Reconnections**:
- Check network stability
- Increase `pingInterval` if on slow connection
- Review `maxReconnectAttempts` setting

**High Memory Usage**:
- Limit message history in hooks (default: 50 alerts, 20 signals, 30 notifications)
- Implement message cleanup after processing
- Consider pagination for large datasets

## Future Enhancements (v1.2.0)

- [ ] Message compression (gzip)
- [ ] Binary protocol (Protocol Buffers / MessagePack)
- [ ] Authentication & authorization per channel
- [ ] Rate limiting per client
- [ ] Horizontal scaling with Redis Pub/Sub
- [ ] Message persistence and replay
- [ ] WebRTC for audio/video alerts
- [ ] Mobile push notification integration

## References

- [WebSocket API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [React Hooks Best Practices](https://react.dev/reference/react/hooks)

---

**Last Updated**: 2026-01-06
**Version**: v1.1.0-dev
