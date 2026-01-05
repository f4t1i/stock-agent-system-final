#!/usr/bin/env python3
"""
Watchlist Monitor - Monitor Watchlist Symbols

Continuously monitors symbols in watchlists and triggers alerts
when conditions are met.

Features:
- Real-time price monitoring
- Alert evaluation for all watchlist symbols
- Configurable check intervals
- Batch processing for efficiency

Usage:
    monitor = WatchlistMonitor()
    monitor.start()  # Start monitoring in background
    monitor.stop()   # Stop monitoring
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import threading
from pathlib import Path
import sqlite3
from loguru import logger

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.alert_evaluator import AlertEvaluator
from monitoring.notification_dispatcher import NotificationDispatcher


class WatchlistMonitor:
    """Monitor watchlist symbols and trigger alerts"""
    
    def __init__(
        self,
        check_interval: int = 60,
        alerts_db_path: Optional[str] = None,
        watchlist_db_path: Optional[str] = None
    ):
        """
        Initialize watchlist monitor
        
        Args:
            check_interval: Seconds between checks (default: 60)
            alerts_db_path: Path to alerts database
            watchlist_db_path: Path to watchlist database
        """
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Database paths
        if alerts_db_path is None:
            alerts_db_path = project_root / "data" / "alerts.db"
        if watchlist_db_path is None:
            watchlist_db_path = project_root / "data" / "watchlist.db"
        
        self.alerts_db_path = Path(alerts_db_path)
        self.watchlist_db_path = Path(watchlist_db_path)
        
        # Services
        self.evaluator = AlertEvaluator()
        self.dispatcher = NotificationDispatcher()
    
    def start(self):
        """Start monitoring in background thread"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Watchlist monitor started (interval: {self.check_interval}s)")
    
    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Watchlist monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_all_alerts()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            # Sleep in small intervals to allow quick shutdown
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_all_alerts(self):
        """Check all enabled alerts"""
        try:
            # Get all enabled alerts
            with sqlite3.connect(str(self.alerts_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM alerts 
                    WHERE enabled = 1
                """)
                alerts = cursor.fetchall()
            
            if not alerts:
                return
            
            logger.debug(f"Checking {len(alerts)} alerts")
            
            # Group alerts by symbol for efficient data fetching
            symbols_to_check = set(alert["symbol"] for alert in alerts)
            
            # Fetch current prices (mock for now)
            current_prices = self._fetch_prices(list(symbols_to_check))
            
            # Check each alert
            for alert in alerts:
                try:
                    self._check_alert(alert, current_prices)
                except Exception as e:
                    logger.error(f"Error checking alert {alert['alert_id']}: {e}")
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _check_alert(self, alert: sqlite3.Row, current_prices: Dict[str, float]):
        """Check single alert"""
        symbol = alert["symbol"]
        
        if symbol not in current_prices:
            logger.warning(f"No price data for {symbol}")
            return
        
        current_value = current_prices[symbol]
        
        # Evaluate condition
        triggered = self.evaluator.evaluate_condition(
            current_value=current_value,
            condition=alert["condition"],
            threshold=alert["threshold"],
            alert_id=alert["alert_id"]
        )
        
        if triggered:
            logger.info(f"Alert {alert['alert_id']} triggered for {symbol}")
            
            # Send notification
            import json
            channels = json.loads(alert["notification_channels"])
            
            self.dispatcher.send(
                alert_id=alert["alert_id"],
                symbol=symbol,
                alert_type=alert["alert_type"],
                current_value=current_value,
                threshold=alert["threshold"],
                channels=channels
            )
            
            # Update last_triggered in database
            with sqlite3.connect(str(self.alerts_db_path)) as conn:
                conn.execute(
                    "UPDATE alerts SET last_triggered = ? WHERE alert_id = ?",
                    (datetime.now().isoformat(), alert["alert_id"])
                )
                conn.commit()
    
    def _fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to current price
        """
        # TODO: Integrate with real market data API
        # For now, return mock prices
        import random
        return {
            symbol: random.uniform(100, 200)
            for symbol in symbols
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "running": self.running,
            "check_interval": self.check_interval,
            "alerts_db": str(self.alerts_db_path),
            "watchlist_db": str(self.watchlist_db_path)
        }


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import signal
    
    # Create monitor
    monitor = WatchlistMonitor(check_interval=10)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping monitor...")
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Watchlist Monitor Test ===")
    print(f"Starting monitor (check interval: 10s)")
    print("Press Ctrl+C to stop\n")
    
    monitor.start()
    
    # Keep running
    while True:
        time.sleep(1)
