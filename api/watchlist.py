#!/usr/bin/env python3
"""
Watchlist API - Manage Stock Watchlists

Provides endpoints for creating and managing stock watchlists.

Endpoints:
- POST /api/watchlist/create - Create new watchlist
- GET /api/watchlist/list - List user watchlists
- POST /api/watchlist/{id}/add - Add symbol to watchlist
- DELETE /api/watchlist/{id}/remove/{symbol} - Remove symbol
- GET /api/watchlist/{id}/symbols - Get watchlist symbols
- DELETE /api/watchlist/{id} - Delete watchlist

Usage:
    from api.watchlist import router
    app.include_router(router, prefix="/api/watchlist", tags=["watchlist"])
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import sqlite3
import uuid

router = APIRouter()

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "watchlist.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Models
# ============================================================================

class CreateWatchlistRequest(BaseModel):
    """Request to create watchlist"""
    name: str = Field(..., description="Watchlist name")
    description: Optional[str] = Field(None, description="Watchlist description")


class AddSymbolRequest(BaseModel):
    """Request to add symbol"""
    symbol: str = Field(..., description="Stock symbol")


class WatchlistResponse(BaseModel):
    """Watchlist response"""
    watchlist_id: str
    user_id: str
    name: str
    description: Optional[str]
    symbol_count: int
    created_at: datetime


class WatchlistSymbol(BaseModel):
    """Watchlist symbol"""
    symbol: str
    added_at: datetime


# ============================================================================
# Database Init
# ============================================================================

def init_database():
    """Initialize database schema"""
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                watchlist_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_symbols (
                watchlist_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (watchlist_id, symbol),
                FOREIGN KEY (watchlist_id) REFERENCES watchlists(watchlist_id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlists_user 
            ON watchlists(user_id)
        """)
        
        conn.commit()


init_database()


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/create", response_model=WatchlistResponse)
async def create_watchlist(request: CreateWatchlistRequest, user_id: str = "demo_user"):
    """Create new watchlist"""
    try:
        watchlist_id = str(uuid.uuid4())
        
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("""
                INSERT INTO watchlists (watchlist_id, user_id, name, description, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                watchlist_id,
                user_id,
                request.name,
                request.description,
                datetime.now().isoformat()
            ))
            conn.commit()
        
        return WatchlistResponse(
            watchlist_id=watchlist_id,
            user_id=user_id,
            name=request.name,
            description=request.description,
            symbol_count=0,
            created_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating watchlist: {str(e)}")


@router.get("/list", response_model=List[WatchlistResponse])
async def list_watchlists(user_id: str = "demo_user"):
    """List user watchlists"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT w.*, COUNT(ws.symbol) as symbol_count
                FROM watchlists w
                LEFT JOIN watchlist_symbols ws ON w.watchlist_id = ws.watchlist_id
                WHERE w.user_id = ?
                GROUP BY w.watchlist_id
                ORDER BY w.created_at DESC
            """, (user_id,))
            
            rows = cursor.fetchall()
            
            return [
                WatchlistResponse(
                    watchlist_id=row["watchlist_id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    description=row["description"],
                    symbol_count=row["symbol_count"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing watchlists: {str(e)}")


@router.post("/{watchlist_id}/add_symbol")
async def add_symbol(watchlist_id: str, request: AddSymbolRequest):
    """Add symbol to watchlist"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            # Check if watchlist exists
            cursor = conn.execute(
                "SELECT watchlist_id FROM watchlists WHERE watchlist_id = ?",
                (watchlist_id,)
            )
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Watchlist not found")
            
            # Add symbol (ignore if already exists)
            try:
                conn.execute("""
                    INSERT INTO watchlist_symbols (watchlist_id, symbol, added_at)
                    VALUES (?, ?, ?)
                """, (watchlist_id, request.symbol.upper(), datetime.now().isoformat()))
                conn.commit()
            except sqlite3.IntegrityError:
                # Symbol already in watchlist
                pass
        
        return {"success": True, "watchlist_id": watchlist_id, "symbol": request.symbol}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding symbol: {str(e)}")


@router.delete("/{watchlist_id}/remove_symbol/{symbol}")
async def remove_symbol(watchlist_id: str, symbol: str):
    """Remove symbol from watchlist"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.execute("""
                DELETE FROM watchlist_symbols
                WHERE watchlist_id = ? AND symbol = ?
            """, (watchlist_id, symbol.upper()))
            conn.commit()
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Symbol not found in watchlist")
        
        return {"success": True, "watchlist_id": watchlist_id, "symbol": symbol}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing symbol: {str(e)}")


@router.get("/{watchlist_id}/symbols", response_model=List[WatchlistSymbol])
async def get_symbols(watchlist_id: str):
    """Get watchlist symbols"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT symbol, added_at
                FROM watchlist_symbols
                WHERE watchlist_id = ?
                ORDER BY added_at DESC
            """, (watchlist_id,))
            
            rows = cursor.fetchall()
            
            return [
                WatchlistSymbol(
                    symbol=row["symbol"],
                    added_at=datetime.fromisoformat(row["added_at"])
                )
                for row in rows
            ]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting symbols: {str(e)}")


@router.delete("/{watchlist_id}")
async def delete_watchlist(watchlist_id: str, user_id: str = "demo_user"):
    """Delete watchlist"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            # Delete symbols first
            conn.execute(
                "DELETE FROM watchlist_symbols WHERE watchlist_id = ?",
                (watchlist_id,)
            )
            
            # Delete watchlist
            cursor = conn.execute(
                "DELETE FROM watchlists WHERE watchlist_id = ? AND user_id = ?",
                (watchlist_id, user_id)
            )
            conn.commit()
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Watchlist not found")
        
        return {"success": True, "watchlist_id": watchlist_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting watchlist: {str(e)}")


@router.get("/{watchlist_id}/status")
async def get_watchlist_status(watchlist_id: str):
    """
    Get watchlist status with current prices and metrics
    """
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get watchlist
            cursor = conn.execute(
                "SELECT * FROM watchlists WHERE watchlist_id = ?",
                (watchlist_id,)
            )
            watchlist = cursor.fetchone()
            if not watchlist:
                raise HTTPException(status_code=404, detail="Watchlist not found")
            
            # Get symbols
            cursor = conn.execute(
                "SELECT symbol FROM watchlist_symbols WHERE watchlist_id = ?",
                (watchlist_id,)
            )
            symbols = [row["symbol"] for row in cursor.fetchall()]
        
        # Mock: Get current prices for symbols
        symbols_status = []
        for symbol in symbols:
            symbols_status.append({
                "symbol": symbol,
                "current_price": 150.0,  # Mock price
                "change_percent": 2.5,   # Mock change
                "volume": 1000000,       # Mock volume
                "last_updated": datetime.now().isoformat()
            })
        
        return {
            "watchlist_id": watchlist_id,
            "name": watchlist["name"],
            "symbol_count": len(symbols),
            "symbols": symbols_status,
            "last_updated": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting watchlist status: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "watchlist-api"}


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Watchlist API")
    app.include_router(router, prefix="/api/watchlist", tags=["watchlist"])
    
    print("Starting Watchlist API on http://localhost:8002")
    print("Docs: http://localhost:8002/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
