"""
Real-time Market Data Streaming

Integration with Finnhub and Polygon.io for real-time market data.

Key Features:
1. Real-time price updates
2. News streaming
3. Trade and quote data
4. WebSocket connections
5. Data buffering and batching

APIs:
- Finnhub: News, sentiment, earnings
- Polygon.io: Real-time trades, quotes, aggregates
"""

import asyncio
import websockets
import json
import requests
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from loguru import logger
import time
from collections import deque
import threading


@dataclass
class MarketUpdate:
    """Market data update"""
    symbol: str
    timestamp: float
    update_type: str  # price, news, trade, quote
    data: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FinnhubStreamer:
    """
    Finnhub WebSocket streamer
    
    Streams real-time trades and news from Finnhub.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Finnhub streamer
        
        Args:
            api_key: Finnhub API key
        """
        self.api_key = api_key
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        self.ws = None
        self.subscribed_symbols = set()
        self.callbacks = []
        self.running = False
    
    async def connect(self):
        """Connect to Finnhub WebSocket"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            logger.info("Connected to Finnhub WebSocket")
            self.running = True
        except Exception as e:
            logger.error(f"Failed to connect to Finnhub: {e}")
            raise
    
    async def subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols
        
        Args:
            symbols: List of symbols to subscribe
        """
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                await self.ws.send(json.dumps({
                    'type': 'subscribe',
                    'symbol': symbol
                }))
                self.subscribed_symbols.add(symbol)
                logger.info(f"Subscribed to {symbol} on Finnhub")
    
    async def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols
        
        Args:
            symbols: List of symbols to unsubscribe
        """
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                await self.ws.send(json.dumps({
                    'type': 'unsubscribe',
                    'symbol': symbol
                }))
                self.subscribed_symbols.remove(symbol)
                logger.info(f"Unsubscribed from {symbol} on Finnhub")
    
    def add_callback(self, callback: Callable[[MarketUpdate], None]):
        """
        Add callback for market updates
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    async def stream(self):
        """Stream market data"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                # Parse Finnhub message
                if data.get('type') == 'trade':
                    for trade in data.get('data', []):
                        update = MarketUpdate(
                            symbol=trade['s'],
                            timestamp=trade['t'] / 1000.0,  # Convert to seconds
                            update_type='trade',
                            data={
                                'price': trade['p'],
                                'volume': trade['v'],
                                'conditions': trade.get('c', [])
                            }
                        )
                        
                        # Call callbacks
                        for callback in self.callbacks:
                            try:
                                callback(update)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Finnhub WebSocket connection closed")
            self.running = False
        except Exception as e:
            logger.error(f"Finnhub streaming error: {e}")
            self.running = False
    
    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Finnhub WebSocket closed")
            self.running = False


class PolygonStreamer:
    """
    Polygon.io WebSocket streamer
    
    Streams real-time trades, quotes, and aggregates from Polygon.io.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Polygon streamer
        
        Args:
            api_key: Polygon.io API key
        """
        self.api_key = api_key
        self.ws_url = "wss://socket.polygon.io/stocks"
        self.ws = None
        self.subscribed_symbols = set()
        self.callbacks = []
        self.running = False
    
    async def connect(self):
        """Connect to Polygon WebSocket"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            
            # Authenticate
            await self.ws.send(json.dumps({
                'action': 'auth',
                'params': self.api_key
            }))
            
            # Wait for auth response
            response = await self.ws.recv()
            auth_data = json.loads(response)
            
            if auth_data[0].get('status') == 'auth_success':
                logger.info("Connected and authenticated to Polygon WebSocket")
                self.running = True
            else:
                raise Exception(f"Polygon auth failed: {auth_data}")
        
        except Exception as e:
            logger.error(f"Failed to connect to Polygon: {e}")
            raise
    
    async def subscribe(self, symbols: List[str], channels: List[str] = ['T', 'Q', 'A']):
        """
        Subscribe to symbols and channels
        
        Args:
            symbols: List of symbols to subscribe
            channels: List of channels (T=trades, Q=quotes, A=aggregates)
        """
        for symbol in symbols:
            for channel in channels:
                subscription = f"{channel}.{symbol}"
                if subscription not in self.subscribed_symbols:
                    await self.ws.send(json.dumps({
                        'action': 'subscribe',
                        'params': subscription
                    }))
                    self.subscribed_symbols.add(subscription)
                    logger.info(f"Subscribed to {subscription} on Polygon")
    
    async def unsubscribe(self, symbols: List[str], channels: List[str] = ['T', 'Q', 'A']):
        """
        Unsubscribe from symbols and channels
        
        Args:
            symbols: List of symbols to unsubscribe
            channels: List of channels
        """
        for symbol in symbols:
            for channel in channels:
                subscription = f"{channel}.{symbol}"
                if subscription in self.subscribed_symbols:
                    await self.ws.send(json.dumps({
                        'action': 'unsubscribe',
                        'params': subscription
                    }))
                    self.subscribed_symbols.remove(subscription)
                    logger.info(f"Unsubscribed from {subscription} on Polygon")
    
    def add_callback(self, callback: Callable[[MarketUpdate], None]):
        """
        Add callback for market updates
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    async def stream(self):
        """Stream market data"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                # Parse Polygon messages
                for item in data:
                    ev = item.get('ev')  # Event type
                    
                    if ev == 'T':  # Trade
                        update = MarketUpdate(
                            symbol=item['sym'],
                            timestamp=item['t'] / 1000.0,  # Convert to seconds
                            update_type='trade',
                            data={
                                'price': item['p'],
                                'size': item['s'],
                                'exchange': item.get('x', 0)
                            }
                        )
                        
                        # Call callbacks
                        for callback in self.callbacks:
                            try:
                                callback(update)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                    
                    elif ev == 'Q':  # Quote
                        update = MarketUpdate(
                            symbol=item['sym'],
                            timestamp=item['t'] / 1000.0,
                            update_type='quote',
                            data={
                                'bid_price': item['bp'],
                                'bid_size': item['bs'],
                                'ask_price': item['ap'],
                                'ask_size': item['as']
                            }
                        )
                        
                        for callback in self.callbacks:
                            try:
                                callback(update)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                    
                    elif ev == 'A':  # Aggregate (minute bar)
                        update = MarketUpdate(
                            symbol=item['sym'],
                            timestamp=item['s'] / 1000.0,  # Start time
                            update_type='aggregate',
                            data={
                                'open': item['o'],
                                'high': item['h'],
                                'low': item['l'],
                                'close': item['c'],
                                'volume': item['v']
                            }
                        )
                        
                        for callback in self.callbacks:
                            try:
                                callback(update)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Polygon WebSocket connection closed")
            self.running = False
        except Exception as e:
            logger.error(f"Polygon streaming error: {e}")
            self.running = False
    
    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Polygon WebSocket closed")
            self.running = False


class MarketDataBuffer:
    """
    Buffer for market data updates
    
    Batches updates for efficient processing.
    """
    
    def __init__(self, max_size: int = 1000, flush_interval: float = 1.0):
        """
        Initialize buffer
        
        Args:
            max_size: Maximum buffer size before auto-flush
            flush_interval: Auto-flush interval in seconds
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.flush_callbacks = []
    
    def add(self, update: MarketUpdate):
        """Add update to buffer"""
        with self.lock:
            self.buffer.append(update)
            
            # Auto-flush if buffer is full
            if len(self.buffer) >= self.max_size:
                self._flush()
            
            # Auto-flush if interval elapsed
            elif time.time() - self.last_flush >= self.flush_interval:
                self._flush()
    
    def _flush(self):
        """Flush buffer (internal, assumes lock held)"""
        if not self.buffer:
            return
        
        # Get all updates
        updates = list(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()
        
        # Call flush callbacks
        for callback in self.flush_callbacks:
            try:
                callback(updates)
            except Exception as e:
                logger.error(f"Flush callback error: {e}")
    
    def flush(self):
        """Manually flush buffer"""
        with self.lock:
            self._flush()
    
    def add_flush_callback(self, callback: Callable[[List[MarketUpdate]], None]):
        """
        Add callback for buffer flush
        
        Args:
            callback: Callback function receiving list of updates
        """
        self.flush_callbacks.append(callback)


class RealtimeDataPipeline:
    """
    Complete real-time data pipeline
    
    Combines Finnhub and Polygon streamers with buffering.
    """
    
    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        polygon_api_key: Optional[str] = None,
        buffer_size: int = 1000,
        flush_interval: float = 1.0
    ):
        """
        Initialize real-time data pipeline
        
        Args:
            finnhub_api_key: Finnhub API key
            polygon_api_key: Polygon.io API key
            buffer_size: Buffer size
            flush_interval: Flush interval in seconds
        """
        self.finnhub = FinnhubStreamer(finnhub_api_key) if finnhub_api_key else None
        self.polygon = PolygonStreamer(polygon_api_key) if polygon_api_key else None
        self.buffer = MarketDataBuffer(max_size=buffer_size, flush_interval=flush_interval)
        
        # Add buffer as callback
        if self.finnhub:
            self.finnhub.add_callback(self.buffer.add)
        if self.polygon:
            self.polygon.add_callback(self.buffer.add)
    
    async def start(self, symbols: List[str]):
        """
        Start streaming for symbols
        
        Args:
            symbols: List of symbols to stream
        """
        tasks = []
        
        if self.finnhub:
            await self.finnhub.connect()
            await self.finnhub.subscribe(symbols)
            tasks.append(asyncio.create_task(self.finnhub.stream()))
        
        if self.polygon:
            await self.polygon.connect()
            await self.polygon.subscribe(symbols)
            tasks.append(asyncio.create_task(self.polygon.stream()))
        
        logger.info(f"Started streaming for {len(symbols)} symbols")
        
        # Wait for all streams
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop streaming"""
        if self.finnhub:
            await self.finnhub.close()
        if self.polygon:
            await self.polygon.close()
        
        # Final flush
        self.buffer.flush()
        
        logger.info("Stopped streaming")
    
    def add_flush_callback(self, callback: Callable[[List[MarketUpdate]], None]):
        """
        Add callback for batched updates
        
        Args:
            callback: Callback function
        """
        self.buffer.add_flush_callback(callback)


# Example usage
async def main():
    """Example usage"""
    
    def process_batch(updates: List[MarketUpdate]):
        """Process batch of updates"""
        logger.info(f"Processing batch of {len(updates)} updates")
        
        # Group by symbol
        by_symbol = {}
        for update in updates:
            if update.symbol not in by_symbol:
                by_symbol[update.symbol] = []
            by_symbol[update.symbol].append(update)
        
        for symbol, symbol_updates in by_symbol.items():
            logger.info(f"{symbol}: {len(symbol_updates)} updates")
    
    # Initialize pipeline
    pipeline = RealtimeDataPipeline(
        finnhub_api_key='your_finnhub_key',
        polygon_api_key='your_polygon_key',
        buffer_size=100,
        flush_interval=5.0
    )
    
    # Add batch processor
    pipeline.add_flush_callback(process_batch)
    
    # Start streaming
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        await pipeline.start(symbols)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        await pipeline.stop()


if __name__ == '__main__':
    asyncio.run(main())
