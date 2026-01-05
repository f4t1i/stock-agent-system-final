#!/usr/bin/env python3
"""
Alerts System API - Real-time Stock Alerts

Provides endpoints for creating, managing, and triggering stock price alerts
and watchlists.

Endpoints:
- POST /api/alerts/create - Create new alert
- GET /api/alerts/list - List user alerts
- PUT /api/alerts/{id}/update - Update alert
- DELETE /api/alerts/{id} - Delete alert
- POST /api/alerts/{id}/trigger - Manually trigger alert

Alert Types:
- price_threshold: Price crosses threshold
- confidence_change: Agent confidence changes significantly
- recommendation_change: Agent recommendation changes
- technical_signal: Technical indicator signal (RSI, MACD, etc.)

Usage:
    from api.alerts import router
    app.include_router(router, prefix="/api/alerts", tags=["alerts"])
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from pathlib import Path
import sys
import sqlite3
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.alert_evaluator import AlertEvaluator
from monitoring.notification_dispatcher import NotificationDispatcher

router = APIRouter()

# Initialize services
alert_evaluator = AlertEvaluator()
notification_dispatcher = NotificationDispatcher()


# ============================================================================
# Request/Response Models
# ============================================================================

AlertType = Literal["price_threshold", "confidence_change", "recommendation_change", "technical_signal"]
ConditionType = Literal["above", "below", "crosses_above", "crosses_below", "equals"]
NotificationChannel = Literal["email", "push", "webhook"]


class CreateAlertRequest(BaseModel):
    """Request to create a new alert"""
    symbol: str = Field(..., description="Stock symbol")
    alert_type: AlertType = Field(..., description="Type of alert")
    condition: ConditionType = Field(..., description="Condition to trigger alert")
    threshold: float = Field(..., description="Threshold value")
    notification_channels: List[NotificationChannel] = Field(default=["push"], description="Notification channels")
    enabled: bool = Field(default=True, description="Whether alert is active")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class UpdateAlertRequest(BaseModel):
    """Request to update an alert"""
    enabled: Optional[bool] = None
    threshold: Optional[float] = None
    notification_channels: Optional[List[NotificationChannel]] = None


class AlertResponse(BaseModel):
    """Alert response"""
    alert_id: str
    user_id: str
    symbol: str
    alert_type: AlertType
    condition: ConditionType
    threshold: float
    notification_channels: List[NotificationChannel]
    enabled: bool
    last_triggered: Optional[datetime]
    created_at: datetime
    metadata: Dict[str, Any]


class AlertTriggerResponse(BaseModel):
    """Alert trigger response"""
    alert_id: str
    triggered: bool
    current_value: float
    threshold: float
    notification_sent: bool
    timestamp: datetime


# ============================================================================
# Database Helper
# ============================================================================

class AlertDatabase:
    """SQLite database for alerts"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = project_root / "data" / "alerts.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database schema"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    notification_channels TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    last_triggered TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    notification_sent INTEGER NOT NULL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_user 
                ON alerts(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_symbol 
                ON alerts(symbol)
            """)
            
            conn.commit()


# Initialize database
alert_db = AlertDatabase()


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/create", response_model=AlertResponse)
async def create_alert(request: CreateAlertRequest, user_id: str = "demo_user"):
    """
    Create a new alert
    
    Args:
        request: Alert creation request
        user_id: User identifier (from auth)
        
    Returns:
        Created alert
    """
    try:
        alert_id = str(uuid.uuid4())
        
        with sqlite3.connect(str(alert_db.db_path)) as conn:
            import json
            conn.execute("""
                INSERT INTO alerts (
                    alert_id, user_id, symbol, alert_type, condition,
                    threshold, notification_channels, enabled, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                user_id,
                request.symbol,
                request.alert_type,
                request.condition,
                request.threshold,
                json.dumps(request.notification_channels),
                1 if request.enabled else 0,
                datetime.now().isoformat(),
                json.dumps(request.metadata or {})
            ))
            conn.commit()
        
        return AlertResponse(
            alert_id=alert_id,
            user_id=user_id,
            symbol=request.symbol,
            alert_type=request.alert_type,
            condition=request.condition,
            threshold=request.threshold,
            notification_channels=request.notification_channels,
            enabled=request.enabled,
            last_triggered=None,
            created_at=datetime.now(),
            metadata=request.metadata or {}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating alert: {str(e)}")


@router.get("/list", response_model=List[AlertResponse])
async def list_alerts(
    user_id: str = "demo_user",
    symbol: Optional[str] = None,
    enabled_only: bool = False
):
    """
    List user alerts
    
    Args:
        user_id: User identifier
        symbol: Filter by symbol (optional)
        enabled_only: Only return enabled alerts
        
    Returns:
        List of alerts
    """
    try:
        query = "SELECT * FROM alerts WHERE user_id = ?"
        params = [user_id]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if enabled_only:
            query += " AND enabled = 1"
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(str(alert_db.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            import json
            alerts = []
            for row in rows:
                alerts.append(AlertResponse(
                    alert_id=row["alert_id"],
                    user_id=row["user_id"],
                    symbol=row["symbol"],
                    alert_type=row["alert_type"],
                    condition=row["condition"],
                    threshold=row["threshold"],
                    notification_channels=json.loads(row["notification_channels"]),
                    enabled=bool(row["enabled"]),
                    last_triggered=datetime.fromisoformat(row["last_triggered"]) if row["last_triggered"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                ))
            
            return alerts
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing alerts: {str(e)}")


@router.put("/{alert_id}/update", response_model=AlertResponse)
async def update_alert(
    alert_id: str,
    request: UpdateAlertRequest,
    user_id: str = "demo_user"
):
    """
    Update an alert
    
    Args:
        alert_id: Alert identifier
        request: Update request
        user_id: User identifier
        
    Returns:
        Updated alert
    """
    try:
        # Build update query
        updates = []
        params = []
        
        if request.enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if request.enabled else 0)
        
        if request.threshold is not None:
            updates.append("threshold = ?")
            params.append(request.threshold)
        
        if request.notification_channels is not None:
            import json
            updates.append("notification_channels = ?")
            params.append(json.dumps(request.notification_channels))
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        params.extend([alert_id, user_id])
        
        with sqlite3.connect(str(alert_db.db_path)) as conn:
            conn.execute(f"""
                UPDATE alerts 
                SET {', '.join(updates)}
                WHERE alert_id = ? AND user_id = ?
            """, params)
            conn.commit()
            
            # Fetch updated alert
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            import json
            return AlertResponse(
                alert_id=row["alert_id"],
                user_id=row["user_id"],
                symbol=row["symbol"],
                alert_type=row["alert_type"],
                condition=row["condition"],
                threshold=row["threshold"],
                notification_channels=json.loads(row["notification_channels"]),
                enabled=bool(row["enabled"]),
                last_triggered=datetime.fromisoformat(row["last_triggered"]) if row["last_triggered"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating alert: {str(e)}")


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str, user_id: str = "demo_user"):
    """
    Delete an alert
    
    Args:
        alert_id: Alert identifier
        user_id: User identifier
        
    Returns:
        Success response
    """
    try:
        with sqlite3.connect(str(alert_db.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM alerts WHERE alert_id = ? AND user_id = ?",
                (alert_id, user_id)
            )
            conn.commit()
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"success": True, "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting alert: {str(e)}")


@router.post("/{alert_id}/trigger", response_model=AlertTriggerResponse)
async def trigger_alert(alert_id: str, current_value: float):
    """
    Manually trigger an alert (for testing)
    
    Args:
        alert_id: Alert identifier
        current_value: Current value to evaluate
        
    Returns:
        Trigger response
    """
    try:
        # Fetch alert
        with sqlite3.connect(str(alert_db.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            # Evaluate condition
            triggered = alert_evaluator.evaluate_condition(
                current_value=current_value,
                condition=row["condition"],
                threshold=row["threshold"]
            )
            
            notification_sent = False
            if triggered:
                # Send notification
                import json
                channels = json.loads(row["notification_channels"])
                notification_sent = notification_dispatcher.send(
                    alert_id=alert_id,
                    symbol=row["symbol"],
                    alert_type=row["alert_type"],
                    current_value=current_value,
                    threshold=row["threshold"],
                    channels=channels
                )
                
                # Update last_triggered
                conn.execute(
                    "UPDATE alerts SET last_triggered = ? WHERE alert_id = ?",
                    (datetime.now().isoformat(), alert_id)
                )
                
                # Log to history
                conn.execute("""
                    INSERT INTO alert_history (
                        alert_id, triggered_at, current_value, notification_sent
                    ) VALUES (?, ?, ?, ?)
                """, (
                    alert_id,
                    datetime.now().isoformat(),
                    current_value,
                    1 if notification_sent else 0
                ))
                
                conn.commit()
            
            return AlertTriggerResponse(
                alert_id=alert_id,
                triggered=triggered,
                current_value=current_value,
                threshold=row["threshold"],
                notification_sent=notification_sent,
                timestamp=datetime.now()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering alert: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "alerts-api",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Alerts API")
    app.include_router(router, prefix="/api/alerts", tags=["alerts"])
    
    print("Starting Alerts API on http://localhost:8001")
    print("Docs: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
