"""
FastAPI Server for Stock Analysis Multi-Agent System

Provides REST API endpoints for:
- Single symbol analysis
- Batch analysis
- Backtesting
- Model information
- Health checks
"""

import os
from typing import Dict, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    BatchAnalyzeRequest,
    BatchAnalysisResponse,
    BacktestRequest,
    BacktestResponse,
    ModelsResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)

from orchestration.coordinator import SystemCoordinator
from training.rl.backtester import Backtester
from utils.config_loader import load_config


# Initialize FastAPI app
app = FastAPI(
    title="Stock Analysis Multi-Agent System API",
    description="REST API for multi-agent stock analysis with LLM-based agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator instance
coordinator: SystemCoordinator = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global coordinator
    
    logger.info("Starting Stock Analysis API Server")
    
    try:
        # Load config
        config_path = os.getenv('CONFIG_PATH', 'config/system.yaml')
        
        # Initialize coordinator
        coordinator = SystemCoordinator(config_path=config_path)
        
        logger.info("System initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Stock Analysis API Server")


# ========== API Endpoints ==========

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "Stock Analysis Multi-Agent System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and agent availability.
    """
    agents_loaded = {
        'news': coordinator.agents.get('news') is not None,
        'technical': coordinator.agents.get('technical') is not None,
        'fundamental': coordinator.agents.get('fundamental') is not None,
        'supervisor': coordinator.supervisor is not None,
        'strategist': coordinator.strategist is not None
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents_loaded=agents_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def get_models():
    """
    Get information about available models.
    
    Returns configuration and status of all agents.
    """
    models = []
    
    for agent_name, agent_config in coordinator.config['agents'].items():
        models.append(ModelInfo(
            agent_type=agent_name,
            model_path=agent_config['model_path'],
            enabled=agent_config['enabled'],
            version=None  # Could be extracted from model metadata
        ))
    
    return ModelsResponse(
        models=models,
        supervisor_enabled=coordinator.config.get('supervisor', {}).get('enabled', False),
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_symbol(request: AnalyzeRequest):
    """
    Analyze a single stock symbol.
    
    Runs the complete multi-agent analysis workflow and returns
    a trading recommendation with risk management parameters.
    
    Args:
        request: Analysis request with symbol and parameters
    
    Returns:
        Complete analysis result with recommendation
    """
    try:
        logger.info(f"Analyzing symbol: {request.symbol}")
        
        # Run analysis
        result = coordinator.analyze_symbol(
            symbol=request.symbol,
            use_supervisor=request.use_supervisor,
            lookback_days=request.lookback_days
        )
        
        # Convert to response schema
        response = AnalysisResponse(
            symbol=result['symbol'],
            recommendation=result['recommendation'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            position_size=result['position_size'],
            entry_target=result.get('entry_target'),
            stop_loss=result.get('stop_loss'),
            take_profit=result.get('take_profit'),
            risk_assessment=result['risk_assessment'],
            agent_outputs=result.get('agent_outputs', {}),
            strategist_output=result.get('strategist_output', {}),
            timestamp=result['timestamp'],
            errors=result.get('errors', [])
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/batch", response_model=BatchAnalysisResponse, tags=["Analysis"])
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple stock symbols in batch.
    
    Runs analysis for each symbol sequentially.
    
    Args:
        request: Batch analysis request with symbols
    
    Returns:
        Batch analysis results
    """
    try:
        logger.info(f"Batch analyzing {len(request.symbols)} symbols")
        
        # Run batch analysis
        results = coordinator.batch_analyze(
            symbols=request.symbols,
            use_supervisor=request.use_supervisor,
            lookback_days=request.lookback_days
        )
        
        # Convert to response schema
        analysis_responses = []
        successful = 0
        failed = 0
        
        for result in results:
            if 'error' not in result or not result.get('error'):
                successful += 1
            else:
                failed += 1
            
            analysis_responses.append(AnalysisResponse(
                symbol=result['symbol'],
                recommendation=result.get('recommendation', 'hold'),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                position_size=result.get('position_size', 0.0),
                entry_target=result.get('entry_target'),
                stop_loss=result.get('stop_loss'),
                take_profit=result.get('take_profit'),
                risk_assessment=result.get('risk_assessment', ''),
                agent_outputs=result.get('agent_outputs', {}),
                strategist_output=result.get('strategist_output', {}),
                timestamp=result.get('timestamp', datetime.now().isoformat()),
                errors=result.get('errors', [])
            ))
        
        return BatchAnalysisResponse(
            results=analysis_responses,
            total_analyzed=len(results),
            successful=successful,
            failed=failed,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@app.post("/backtest", response_model=BacktestResponse, tags=["Backtesting"])
async def run_backtest(request: BacktestRequest):
    """
    Run backtesting on historical data.
    
    Simulates trading based on agent recommendations over a historical period.
    
    Args:
        request: Backtest request with symbols and date range
    
    Returns:
        Backtest results with metrics and trade history
    """
    try:
        logger.info(f"Running backtest for {len(request.symbols)} symbols")
        
        # Initialize backtester
        backtester = Backtester(
            coordinator=coordinator,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )
        
        # Run backtest
        metrics = backtester.run(
            symbols=request.symbols,
            use_supervisor=request.use_supervisor
        )
        
        # Get trades and equity curve
        trades = backtester.get_trades()
        equity_curve = backtester.get_equity_curve()
        
        return BacktestResponse(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Backtesting failed: {str(e)}"
        )


# ========== Error Handlers ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# ========== Main ==========

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
