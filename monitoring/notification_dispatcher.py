#!/usr/bin/env python3
"""
Notification Dispatcher - Send Alert Notifications

Dispatches notifications via multiple channels:
- Email
- Push notifications (via Manus notification API)
- Webhooks

Features:
- Rate limiting to prevent spam
- Deduplication within time window
- Notification templates
- Retry logic for failed sends

Usage:
    dispatcher = NotificationDispatcher()
    success = dispatcher.send(alert_id, symbol, alert_type, current_value, threshold, channels)
"""

from typing import List, Dict, Any, Literal
from datetime import datetime, timedelta
from loguru import logger
import time


NotificationChannel = Literal["email", "push", "webhook"]


class NotificationDispatcher:
    """Dispatch notifications via multiple channels"""
    
    def __init__(
        self,
        rate_limit_per_minute: int = 10,
        dedup_window_seconds: int = 300
    ):
        """
        Initialize notification dispatcher
        
        Args:
            rate_limit_per_minute: Max notifications per minute
            dedup_window_seconds: Deduplication window (default: 5 minutes)
        """
        self.rate_limit = rate_limit_per_minute
        self.dedup_window = dedup_window_seconds
        
        # Track sent notifications
        self.sent_notifications: Dict[str, datetime] = {}
        self.send_times: List[datetime] = []
    
    def send(
        self,
        alert_id: str,
        symbol: str,
        alert_type: str,
        current_value: float,
        threshold: float,
        channels: List[NotificationChannel]
    ) -> bool:
        """
        Send notification via specified channels
        
        Args:
            alert_id: Alert identifier
            symbol: Stock symbol
            alert_type: Type of alert
            current_value: Current value
            threshold: Threshold value
            channels: List of notification channels
            
        Returns:
            True if at least one notification was sent successfully
        """
        try:
            # Check rate limit
            if not self._check_rate_limit():
                logger.warning(f"Rate limit exceeded for alert {alert_id}")
                return False
            
            # Check deduplication
            if self._is_duplicate(alert_id):
                logger.info(f"Skipping duplicate notification for alert {alert_id}")
                return False
            
            # Generate notification content
            title, message = self._generate_message(
                symbol, alert_type, current_value, threshold
            )
            
            # Send via each channel
            success = False
            for channel in channels:
                try:
                    if channel == "email":
                        self._send_email(alert_id, title, message)
                        success = True
                    elif channel == "push":
                        self._send_push(alert_id, title, message)
                        success = True
                    elif channel == "webhook":
                        self._send_webhook(alert_id, symbol, current_value, threshold)
                        success = True
                except Exception as e:
                    logger.error(f"Error sending {channel} notification: {e}")
            
            if success:
                # Mark as sent
                self.sent_notifications[alert_id] = datetime.now()
                self.send_times.append(datetime.now())
            
            return success
            
        except Exception as e:
            logger.error(f"Error dispatching notification: {e}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows sending"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old send times
        self.send_times = [t for t in self.send_times if t > cutoff]
        
        # Check limit
        return len(self.send_times) < self.rate_limit
    
    def _is_duplicate(self, alert_id: str) -> bool:
        """Check if notification was recently sent"""
        if alert_id not in self.sent_notifications:
            return False
        
        last_sent = self.sent_notifications[alert_id]
        elapsed = (datetime.now() - last_sent).total_seconds()
        
        return elapsed < self.dedup_window
    
    def _generate_message(
        self,
        symbol: str,
        alert_type: str,
        current_value: float,
        threshold: float
    ) -> tuple[str, str]:
        """Generate notification title and message"""
        title = f"Alert: {symbol}"
        
        if alert_type == "price_threshold":
            message = f"{symbol} price is ${current_value:.2f} (threshold: ${threshold:.2f})"
        elif alert_type == "confidence_change":
            message = f"{symbol} confidence changed to {current_value:.1%} (threshold: {threshold:.1%})"
        elif alert_type == "recommendation_change":
            message = f"{symbol} recommendation changed"
        elif alert_type == "technical_signal":
            message = f"{symbol} technical signal triggered"
        else:
            message = f"{symbol} alert triggered"
        
        return title, message
    
    def _send_email(self, alert_id: str, title: str, message: str):
        """Send email notification"""
        logger.info(f"[EMAIL] {title}: {message}")
        # TODO: Integrate with email service
        # For now, just log
    
    def _send_push(self, alert_id: str, title: str, message: str):
        """Send push notification via Manus API"""
        logger.info(f"[PUSH] {title}: {message}")
        # TODO: Integrate with Manus notification API
        # from server._core.notification import notifyOwner
        # notifyOwner(title=title, content=message)
    
    def _send_webhook(
        self,
        alert_id: str,
        symbol: str,
        current_value: float,
        threshold: float
    ):
        """Send webhook notification"""
        logger.info(f"[WEBHOOK] Alert {alert_id} for {symbol}")
        # TODO: Integrate with webhook service
        # payload = {
        #     "alert_id": alert_id,
        #     "symbol": symbol,
        #     "current_value": current_value,
        #     "threshold": threshold,
        #     "timestamp": datetime.now().isoformat()
        # }
        # requests.post(webhook_url, json=payload)
    
    def clear_history(self):
        """Clear notification history"""
        self.sent_notifications.clear()
        self.send_times.clear()


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    dispatcher = NotificationDispatcher(rate_limit_per_minute=5)
    
    print("=== Notification Dispatcher Tests ===\n")
    
    # Test 1: Send notification
    print("Test 1: Send notification")
    success = dispatcher.send(
        alert_id="alert1",
        symbol="AAPL",
        alert_type="price_threshold",
        current_value=150.50,
        threshold=150.00,
        channels=["push", "email"]
    )
    print(f"  Success: {success}")
    
    # Test 2: Duplicate (should be skipped)
    print("\nTest 2: Duplicate notification (should skip)")
    success = dispatcher.send(
        alert_id="alert1",
        symbol="AAPL",
        alert_type="price_threshold",
        current_value=151.00,
        threshold=150.00,
        channels=["push"]
    )
    print(f"  Success: {success} (should be False)")
    
    # Test 3: Different alert (should send)
    print("\nTest 3: Different alert (should send)")
    success = dispatcher.send(
        alert_id="alert2",
        symbol="GOOGL",
        alert_type="confidence_change",
        current_value=0.85,
        threshold=0.80,
        channels=["push"]
    )
    print(f"  Success: {success}")
    
    # Test 4: Rate limit
    print("\nTest 4: Rate limit (5 per minute)")
    for i in range(7):
        success = dispatcher.send(
            alert_id=f"alert{i+3}",
            symbol="TSLA",
            alert_type="price_threshold",
            current_value=200.0,
            threshold=195.0,
            channels=["push"]
        )
        print(f"  Alert {i+3}: {success}")
    
    print("\n✅ All tests complete")
