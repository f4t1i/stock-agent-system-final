#!/usr/bin/env python3
"""
WebSocket Server - Real-time Updates

Provides WebSocket endpoints for:
- Real-time price streaming
- Alert notifications
- Trading signals
- Portfolio updates
- System events

Usage:
    # Client connects to ws://localhost:8000/ws/{client_id}
    # Subscribes to channels: prices, alerts, signals, portfolio
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from enum import Enum


class MessageType(str, Enum):
    """WebSocket message types"""
    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"

    # Server -> Client
    PRICE_UPDATE = "price_update"
    ALERT_TRIGGERED = "alert_triggered"
    TRADING_SIGNAL = "trading_signal"
    PORTFOLIO_UPDATE = "portfolio_update"
    NOTIFICATION = "notification"
    ERROR = "error"
    PONG = "pong"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class Channel(str, Enum):
    """Available subscription channels"""
    PRICES = "prices"              # Real-time price updates
    ALERTS = "alerts"              # Alert notifications
    SIGNALS = "signals"            # Trading signals
    PORTFOLIO = "portfolio"        # Portfolio updates
    NOTIFICATIONS = "notifications" # System notifications


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting

    Features:
    - Multiple clients with unique IDs
    - Channel-based subscriptions
    - Broadcast to all or specific channels
    - Automatic cleanup on disconnect
    """

    def __init__(self):
        # Active connections: {client_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}

        # Channel subscriptions: {client_id: Set[channel]}
        self.subscriptions: Dict[str, Set[str]] = {}

        # Symbol subscriptions: {client_id: Set[symbol]}
        self.symbol_subscriptions: Dict[str, Set[str]] = {}

        logger.info("WebSocket ConnectionManager initialized")

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        self.symbol_subscriptions[client_id] = set()

        logger.info(f"Client connected: {client_id} | Total: {len(self.active_connections)}")

        # Send welcome message
        await self.send_personal_message(client_id, {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_channels": [c.value for c in Channel]
        })

    def disconnect(self, client_id: str) -> None:
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.symbol_subscriptions:
            del self.symbol_subscriptions[client_id]

        logger.info(f"Client disconnected: {client_id} | Remaining: {len(self.active_connections)}")

    async def subscribe(self, client_id: str, channel: str, symbols: Optional[List[str]] = None) -> None:
        """Subscribe client to a channel"""
        if client_id not in self.subscriptions:
            return

        self.subscriptions[client_id].add(channel)

        # Subscribe to specific symbols for price channel
        if channel == Channel.PRICES and symbols:
            if client_id not in self.symbol_subscriptions:
                self.symbol_subscriptions[client_id] = set()
            self.symbol_subscriptions[client_id].update(symbols)

        logger.info(f"Client {client_id} subscribed to {channel}" +
                   (f" (symbols: {symbols})" if symbols else ""))

        await self.send_personal_message(client_id, {
            "type": MessageType.SUBSCRIBED,
            "channel": channel,
            "symbols": list(self.symbol_subscriptions.get(client_id, [])) if channel == Channel.PRICES else None,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def unsubscribe(self, client_id: str, channel: str) -> None:
        """Unsubscribe client from a channel"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(channel)

            # Clear symbol subscriptions if unsubscribing from prices
            if channel == Channel.PRICES and client_id in self.symbol_subscriptions:
                self.symbol_subscriptions[client_id].clear()

        logger.info(f"Client {client_id} unsubscribed from {channel}")

        await self.send_personal_message(client_id, {
            "type": MessageType.UNSUBSCRIBED,
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def send_personal_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict[str, Any], channel: Optional[str] = None) -> None:
        """Broadcast message to all subscribed clients"""
        disconnected = []

        for client_id, websocket in self.active_connections.items():
            # Filter by channel subscription
            if channel and channel not in self.subscriptions.get(client_id, set()):
                continue

            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

    async def broadcast_price_update(self, symbol: str, price_data: Dict[str, Any]) -> None:
        """Broadcast price update to subscribed clients"""
        message = {
            "type": MessageType.PRICE_UPDATE,
            "symbol": symbol,
            "data": price_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Send to clients subscribed to this symbol
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            # Check if subscribed to prices channel
            if Channel.PRICES not in self.subscriptions.get(client_id, set()):
                continue

            # Check if subscribed to this symbol
            if symbol not in self.symbol_subscriptions.get(client_id, set()):
                continue

            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending price update to {client_id}: {e}")
                disconnected.append(client_id)

        for client_id in disconnected:
            self.disconnect(client_id)

    async def broadcast_alert(self, alert_data: Dict[str, Any]) -> None:
        """Broadcast alert notification"""
        message = {
            "type": MessageType.ALERT_TRIGGERED,
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message, channel=Channel.ALERTS)

    async def broadcast_signal(self, signal_data: Dict[str, Any]) -> None:
        """Broadcast trading signal"""
        message = {
            "type": MessageType.TRADING_SIGNAL,
            "data": signal_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message, channel=Channel.SIGNALS)

    async def broadcast_notification(self, notification: Dict[str, Any]) -> None:
        """Broadcast system notification"""
        message = {
            "type": MessageType.NOTIFICATION,
            "data": notification,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message, channel=Channel.NOTIFICATIONS)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "clients": list(self.active_connections.keys()),
            "channel_subscriptions": {
                client_id: list(channels)
                for client_id, channels in self.subscriptions.items()
            },
            "symbol_subscriptions": {
                client_id: list(symbols)
                for client_id, symbols in self.symbol_subscriptions.items()
            }
        }


# Global connection manager instance
manager = ConnectionManager()


async def handle_client_message(client_id: str, message: Dict[str, Any]) -> None:
    """
    Handle incoming client messages

    Supported message types:
    - subscribe: {"type": "subscribe", "channel": "prices", "symbols": ["AAPL", "MSFT"]}
    - unsubscribe: {"type": "unsubscribe", "channel": "prices"}
    - ping: {"type": "ping"}
    """
    msg_type = message.get("type")

    if msg_type == MessageType.SUBSCRIBE:
        channel = message.get("channel")
        symbols = message.get("symbols")
        if channel:
            await manager.subscribe(client_id, channel, symbols)

    elif msg_type == MessageType.UNSUBSCRIBE:
        channel = message.get("channel")
        if channel:
            await manager.unsubscribe(client_id, channel)

    elif msg_type == MessageType.PING:
        await manager.send_personal_message(client_id, {
            "type": MessageType.PONG,
            "timestamp": datetime.utcnow().isoformat()
        })

    else:
        await manager.send_personal_message(client_id, {
            "type": MessageType.ERROR,
            "error": f"Unknown message type: {msg_type}",
            "timestamp": datetime.utcnow().isoformat()
        })


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Main WebSocket endpoint handler

    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await manager.connect(client_id, websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await handle_client_message(client_id, data)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


# Background task for price streaming simulation
async def price_streaming_task():
    """
    Background task that simulates price updates

    In production, this would connect to real market data feeds
    (Yahoo Finance, Alpha Vantage, etc.)
    """
    import random

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    base_prices = {
        "AAPL": 185.0,
        "MSFT": 375.0,
        "GOOGL": 140.0,
        "AMZN": 155.0,
        "TSLA": 245.0
    }

    while True:
        for symbol in symbols:
            # Simulate price change
            base_price = base_prices[symbol]
            change_pct = random.uniform(-0.02, 0.02)  # ±2%
            new_price = base_price * (1 + change_pct)

            price_data = {
                "price": round(new_price, 2),
                "change": round(new_price - base_price, 2),
                "change_pct": round(change_pct * 100, 2),
                "volume": random.randint(1000000, 5000000),
                "bid": round(new_price - 0.05, 2),
                "ask": round(new_price + 0.05, 2)
            }

            await manager.broadcast_price_update(symbol, price_data)

        # Update every 1 second
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Test the connection manager
    print("WebSocket module loaded successfully")
    print(f"Available channels: {[c.value for c in Channel]}")
    print(f"Message types: {[t.value for t in MessageType]}")
