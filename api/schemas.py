"""
API Schemas - Pydantic models for request/response validation
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ========== Request Schemas ==========

class AnalyzeRequest(BaseModel):
    """Request schema for single symbol analysis"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    use_supervisor: bool = Field(default=False, description="Use supervisor for intelligent routing")
    lookback_days: int = Field(default=7, description="Days to look back for news", ge=1, le=90)


class BatchAnalyzeRequest(BaseModel):
    """Request schema for batch analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols")
    use_supervisor: bool = Field(default=False, description="Use supervisor for intelligent routing")
    lookback_days: int = Field(default=7, description="Days to look back for news", ge=1, le=90)


class BacktestRequest(BaseModel):
    """Request schema for backtesting"""
    symbols: List[str] = Field(..., description="List of stock symbols to backtest")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000.0, description="Initial capital", gt=0)
    use_supervisor: bool = Field(default=False, description="Use supervisor routing")


# ========== Response Schemas ==========

class AgentOutput(BaseModel):
    """Generic agent output schema"""
    recommendation: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None


class NewsAgentOutput(AgentOutput):
    """News agent specific output"""
    sentiment_score: Optional[float] = Field(None, ge=-2, le=2)
    key_events: Optional[List[str]] = None
    news_count: Optional[int] = None


class TechnicalAgentOutput(AgentOutput):
    """Technical agent specific output"""
    signal: Optional[str] = None
    signal_strength: Optional[float] = Field(None, ge=0, le=1)
    support_levels: Optional[List[float]] = None
    resistance_levels: Optional[List[float]] = None
    indicators: Optional[Dict] = None


class FundamentalAgentOutput(AgentOutput):
    """Fundamental agent specific output"""
    valuation: Optional[str] = None
    financial_health_score: Optional[float] = Field(None, ge=0, le=1)
    growth_score: Optional[float] = Field(None, ge=0, le=1)
    metrics: Optional[Dict] = None


class StrategistOutput(BaseModel):
    """Strategist output schema"""
    decision: str = Field(..., description="Final decision (buy/sell/hold)")
    confidence: float = Field(..., ge=0, le=1)
    position_size: float = Field(..., ge=0, le=1)
    entry_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str
    risk_assessment: str


class AnalysisResponse(BaseModel):
    """Response schema for single analysis"""
    symbol: str
    recommendation: str
    confidence: float
    reasoning: str
    position_size: float
    entry_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_assessment: str
    
    # Agent outputs
    agent_outputs: Dict[str, Optional[AgentOutput]]
    strategist_output: StrategistOutput
    
    # Metadata
    timestamp: str
    errors: List[str] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "recommendation": "buy",
                "confidence": 0.85,
                "reasoning": "Strong technical signals combined with positive news sentiment...",
                "position_size": 0.08,
                "entry_target": 185.50,
                "stop_loss": 178.00,
                "take_profit": 195.00,
                "risk_assessment": "Moderate risk with favorable risk/reward ratio",
                "agent_outputs": {},
                "strategist_output": {},
                "timestamp": "2024-01-04T12:00:00",
                "errors": []
            }
        }


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis"""
    results: List[AnalysisResponse]
    total_analyzed: int
    successful: int
    failed: int
    timestamp: str


class BacktestMetrics(BaseModel):
    """Backtest metrics schema"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    final_portfolio_value: float


class BacktestResponse(BaseModel):
    """Response schema for backtesting"""
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    metrics: BacktestMetrics
    trades: List[Dict]
    equity_curve: List[Dict]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information schema"""
    agent_type: str
    model_path: str
    enabled: bool
    version: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response schema for available models"""
    models: List[ModelInfo]
    supervisor_enabled: bool
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agents_loaded: Dict[str, bool]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    timestamp: str
