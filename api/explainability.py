#!/usr/bin/env python3
"""
Explainability API - Agent Decision Transparency

Provides endpoints for explaining agent decisions, extracting reasoning,
and visualizing confidence factors.

Endpoints:
- GET /api/explainability/decision/{decision_id} - Get decision explanation
- POST /api/explainability/analyze - Generate explanation for new analysis
- GET /api/explainability/recent - List recent decisions

Usage:
    from api.explainability import router
    app.include_router(router, prefix="/api/explainability", tags=["explainability"])
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.reasoning_extractor import ReasoningExtractor
from agents.decision_logger import DecisionLogger

router = APIRouter()

# Initialize services
reasoning_extractor = ReasoningExtractor()
decision_logger = DecisionLogger()


# ============================================================================
# Request/Response Models
# ============================================================================

class FactorImportance(BaseModel):
    """Individual factor contributing to decision"""
    name: str = Field(..., description="Factor name")
    importance: float = Field(..., ge=0, le=1, description="Importance weight (0-1)")
    value: Any = Field(..., description="Factor value")
    description: str = Field(..., description="Human-readable description")


class AlternativeScenario(BaseModel):
    """Alternative decision scenario"""
    scenario: str = Field(..., description="Scenario description")
    recommendation: str = Field(..., description="Alternative recommendation")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in alternative")
    reasoning: str = Field(..., description="Reasoning for alternative")


class ExplainabilityResponse(BaseModel):
    """Complete explanation for an agent decision"""
    decision_id: str = Field(..., description="Unique decision ID")
    symbol: str = Field(..., description="Stock symbol")
    agent_name: str = Field(..., description="Agent name")
    recommendation: str = Field(..., description="Buy/Sell/Hold recommendation")
    confidence: float = Field(..., ge=0, le=1, description="Decision confidence")
    reasoning: str = Field(..., description="Detailed reasoning text")
    key_factors: List[FactorImportance] = Field(..., description="Key decision factors")
    alternatives: List[AlternativeScenario] = Field(default=[], description="Alternative scenarios")
    timestamp: datetime = Field(..., description="Decision timestamp")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class AnalyzeRequest(BaseModel):
    """Request to analyze and explain a decision"""
    symbol: str = Field(..., description="Stock symbol")
    agent_name: str = Field(..., description="Agent name (news/technical/fundamental/strategist)")
    agent_output: Dict[str, Any] = Field(..., description="Agent output to explain")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class DecisionSummary(BaseModel):
    """Summary of a decision for listing"""
    decision_id: str
    symbol: str
    agent_name: str
    recommendation: str
    confidence: float
    timestamp: datetime


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/decision/{decision_id}", response_model=ExplainabilityResponse)
async def get_decision_explanation(decision_id: str):
    """
    Get explanation for a specific decision by ID
    
    Args:
        decision_id: Unique decision identifier
        
    Returns:
        Complete explanation with reasoning, factors, and alternatives
        
    Raises:
        HTTPException: If decision not found
    """
    try:
        # Fetch decision from logger
        decision = decision_logger.get_decision(decision_id)
        
        if not decision:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
        
        # Extract reasoning and factors
        reasoning = reasoning_extractor.extract_reasoning(
            agent_output=decision["agent_output"],
            agent_name=decision["agent_name"]
        )
        
        factors = reasoning_extractor.extract_factors(
            agent_output=decision["agent_output"],
            agent_name=decision["agent_name"]
        )
        
        # Generate alternatives
        alternatives = reasoning_extractor.generate_alternatives(
            agent_output=decision["agent_output"],
            agent_name=decision["agent_name"]
        )
        
        return ExplainabilityResponse(
            decision_id=decision_id,
            symbol=decision["symbol"],
            agent_name=decision["agent_name"],
            recommendation=decision["recommendation"],
            confidence=decision["confidence"],
            reasoning=reasoning,
            key_factors=factors,
            alternatives=alternatives,
            timestamp=decision["timestamp"],
            metadata=decision.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching explanation: {str(e)}")


@router.post("/analyze", response_model=ExplainabilityResponse)
async def analyze_decision(request: AnalyzeRequest):
    """
    Generate explanation for a new agent decision
    
    Args:
        request: Analysis request with symbol, agent name, and output
        
    Returns:
        Complete explanation
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Log the decision
        decision_id = decision_logger.log_decision(
            symbol=request.symbol,
            agent_name=request.agent_name,
            agent_output=request.agent_output,
            context=request.context
        )
        
        # Extract reasoning
        reasoning = reasoning_extractor.extract_reasoning(
            agent_output=request.agent_output,
            agent_name=request.agent_name
        )
        
        # Extract factors
        factors = reasoning_extractor.extract_factors(
            agent_output=request.agent_output,
            agent_name=request.agent_name
        )
        
        # Generate alternatives
        alternatives = reasoning_extractor.generate_alternatives(
            agent_output=request.agent_output,
            agent_name=request.agent_name
        )
        
        # Extract recommendation and confidence
        recommendation = request.agent_output.get("recommendation", "HOLD")
        confidence = request.agent_output.get("confidence", 0.5)
        
        return ExplainabilityResponse(
            decision_id=decision_id,
            symbol=request.symbol,
            agent_name=request.agent_name,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=factors,
            alternatives=alternatives,
            timestamp=datetime.now(),
            metadata=request.context or {}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing decision: {str(e)}")


@router.get("/recent", response_model=List[DecisionSummary])
async def list_recent_decisions(
    limit: int = 10,
    agent_name: Optional[str] = None,
    symbol: Optional[str] = None
):
    """
    List recent decisions with optional filters
    
    Args:
        limit: Maximum number of decisions to return (default: 10)
        agent_name: Filter by agent name (optional)
        symbol: Filter by symbol (optional)
        
    Returns:
        List of decision summaries
    """
    try:
        decisions = decision_logger.list_recent(
            limit=limit,
            agent_name=agent_name,
            symbol=symbol
        )
        
        return [
            DecisionSummary(
                decision_id=d["decision_id"],
                symbol=d["symbol"],
                agent_name=d["agent_name"],
                recommendation=d["recommendation"],
                confidence=d["confidence"],
                timestamp=d["timestamp"]
            )
            for d in decisions
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing decisions: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "explainability-api",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Explainability API")
    app.include_router(router, prefix="/api/explainability", tags=["explainability"])
    
    print("Starting Explainability API on http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
