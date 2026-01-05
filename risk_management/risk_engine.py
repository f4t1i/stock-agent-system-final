#!/usr/bin/env python3
"""
Risk Engine - Trading Risk Management

Evaluates trading decisions against risk policies and guardrails.

Features:
- Position size limits
- Portfolio concentration limits
- Drawdown limits
- Volatility gates
- Confidence thresholds
- Custom policy rules

Usage:
    engine = RiskEngine()
    result = engine.evaluate_trade(trade, portfolio, policies)
    if result.approved:
        execute_trade(trade)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCheckStatus(Enum):
    """Risk check result status"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class RiskCheck:
    """Individual risk check result"""
    name: str
    status: RiskCheckStatus
    message: str
    value: Optional[float] = None
    limit: Optional[float] = None


@dataclass
class RiskEvaluationResult:
    """Risk evaluation result"""
    approved: bool
    risk_level: RiskLevel
    checks: List[RiskCheck]
    warnings: List[str]
    timestamp: datetime


class RiskEngine:
    """Trading risk management engine"""
    
    def __init__(self, policies: Optional[Dict[str, Any]] = None):
        """
        Initialize risk engine
        
        Args:
            policies: Risk policy configuration
        """
        self.policies = policies or self._default_policies()
    
    def _default_policies(self) -> Dict[str, Any]:
        """Get default risk policies"""
        return {
            "position_limits": {
                "max_position_size_pct": 10.0,  # % of portfolio
                "max_position_value": 100000.0,  # USD
            },
            "portfolio_limits": {
                "max_concentration_pct": 25.0,  # % in single stock
                "max_sector_concentration_pct": 40.0,  # % in single sector
                "max_drawdown_pct": 20.0,  # Max portfolio drawdown
            },
            "confidence_gates": {
                "min_confidence": 0.6,  # Minimum agent confidence
                "min_confidence_high_risk": 0.8,  # Min confidence for high-risk trades
            },
            "volatility_gates": {
                "max_volatility": 0.5,  # Max daily volatility (50%)
                "max_beta": 2.0,  # Max stock beta
            },
        }
    
    def evaluate_trade(
        self,
        trade: Dict[str, Any],
        portfolio: Dict[str, Any],
        custom_policies: Optional[Dict[str, Any]] = None
    ) -> RiskEvaluationResult:
        """
        Evaluate trade against risk policies
        
        Args:
            trade: Trade details (symbol, action, quantity, price, confidence)
            portfolio: Current portfolio state
            custom_policies: Override default policies
            
        Returns:
            RiskEvaluationResult with approval decision
        """
        policies = custom_policies or self.policies
        checks: List[RiskCheck] = []
        warnings: List[str] = []
        
        # Position size check
        checks.append(self._check_position_size(trade, portfolio, policies))
        
        # Concentration check
        checks.append(self._check_concentration(trade, portfolio, policies))
        
        # Confidence check
        checks.append(self._check_confidence(trade, policies))
        
        # Volatility check
        checks.append(self._check_volatility(trade, policies))
        
        # Drawdown check
        checks.append(self._check_drawdown(portfolio, policies))
        
        # Determine approval
        failed_checks = [c for c in checks if c.status == RiskCheckStatus.FAIL]
        warning_checks = [c for c in checks if c.status == RiskCheckStatus.WARN]
        
        approved = len(failed_checks) == 0
        
        # Determine risk level
        if len(failed_checks) > 0:
            risk_level = RiskLevel.CRITICAL
        elif len(warning_checks) >= 2:
            risk_level = RiskLevel.HIGH
        elif len(warning_checks) == 1:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Collect warnings
        for check in warning_checks:
            warnings.append(check.message)
        
        return RiskEvaluationResult(
            approved=approved,
            risk_level=risk_level,
            checks=checks,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _check_position_size(
        self,
        trade: Dict[str, Any],
        portfolio: Dict[str, Any],
        policies: Dict[str, Any]
    ) -> RiskCheck:
        """Check position size limits"""
        position_value = trade["quantity"] * trade["price"]
        portfolio_value = portfolio.get("total_value", 100000.0)
        position_pct = (position_value / portfolio_value) * 100
        
        max_pct = policies["position_limits"]["max_position_size_pct"]
        max_value = policies["position_limits"]["max_position_value"]
        
        if position_pct > max_pct:
            return RiskCheck(
                name="position_size",
                status=RiskCheckStatus.FAIL,
                message=f"Position size {position_pct:.1f}% exceeds limit {max_pct}%",
                value=position_pct,
                limit=max_pct
            )
        elif position_value > max_value:
            return RiskCheck(
                name="position_size",
                status=RiskCheckStatus.FAIL,
                message=f"Position value ${position_value:,.0f} exceeds limit ${max_value:,.0f}",
                value=position_value,
                limit=max_value
            )
        elif position_pct > max_pct * 0.8:
            return RiskCheck(
                name="position_size",
                status=RiskCheckStatus.WARN,
                message=f"Position size {position_pct:.1f}% approaching limit {max_pct}%",
                value=position_pct,
                limit=max_pct
            )
        else:
            return RiskCheck(
                name="position_size",
                status=RiskCheckStatus.PASS,
                message="Position size within limits",
                value=position_pct,
                limit=max_pct
            )
    
    def _check_concentration(
        self,
        trade: Dict[str, Any],
        portfolio: Dict[str, Any],
        policies: Dict[str, Any]
    ) -> RiskCheck:
        """Check portfolio concentration limits"""
        # Simplified: assume single stock concentration
        position_value = trade["quantity"] * trade["price"]
        portfolio_value = portfolio.get("total_value", 100000.0)
        concentration_pct = (position_value / portfolio_value) * 100
        
        max_concentration = policies["portfolio_limits"]["max_concentration_pct"]
        
        if concentration_pct > max_concentration:
            return RiskCheck(
                name="concentration",
                status=RiskCheckStatus.FAIL,
                message=f"Concentration {concentration_pct:.1f}% exceeds limit {max_concentration}%",
                value=concentration_pct,
                limit=max_concentration
            )
        else:
            return RiskCheck(
                name="concentration",
                status=RiskCheckStatus.PASS,
                message="Concentration within limits",
                value=concentration_pct,
                limit=max_concentration
            )
    
    def _check_confidence(
        self,
        trade: Dict[str, Any],
        policies: Dict[str, Any]
    ) -> RiskCheck:
        """Check agent confidence threshold"""
        confidence = trade.get("confidence", 0.0)
        min_confidence = policies["confidence_gates"]["min_confidence"]
        
        if confidence < min_confidence:
            return RiskCheck(
                name="confidence",
                status=RiskCheckStatus.FAIL,
                message=f"Confidence {confidence:.2f} below minimum {min_confidence}",
                value=confidence,
                limit=min_confidence
            )
        elif confidence < min_confidence + 0.1:
            return RiskCheck(
                name="confidence",
                status=RiskCheckStatus.WARN,
                message=f"Confidence {confidence:.2f} near minimum {min_confidence}",
                value=confidence,
                limit=min_confidence
            )
        else:
            return RiskCheck(
                name="confidence",
                status=RiskCheckStatus.PASS,
                message=f"Confidence {confidence:.2f} acceptable",
                value=confidence,
                limit=min_confidence
            )
    
    def _check_volatility(
        self,
        trade: Dict[str, Any],
        policies: Dict[str, Any]
    ) -> RiskCheck:
        """Check volatility gates"""
        volatility = trade.get("volatility", 0.2)  # Mock
        max_volatility = policies["volatility_gates"]["max_volatility"]
        
        if volatility > max_volatility:
            return RiskCheck(
                name="volatility",
                status=RiskCheckStatus.FAIL,
                message=f"Volatility {volatility:.2f} exceeds limit {max_volatility}",
                value=volatility,
                limit=max_volatility
            )
        else:
            return RiskCheck(
                name="volatility",
                status=RiskCheckStatus.PASS,
                message="Volatility within limits",
                value=volatility,
                limit=max_volatility
            )
    
    def _check_drawdown(
        self,
        portfolio: Dict[str, Any],
        policies: Dict[str, Any]
    ) -> RiskCheck:
        """Check portfolio drawdown"""
        current_value = portfolio.get("total_value", 100000.0)
        peak_value = portfolio.get("peak_value", 100000.0)
        drawdown_pct = ((peak_value - current_value) / peak_value) * 100
        
        max_drawdown = policies["portfolio_limits"]["max_drawdown_pct"]
        
        if drawdown_pct > max_drawdown:
            return RiskCheck(
                name="drawdown",
                status=RiskCheckStatus.FAIL,
                message=f"Drawdown {drawdown_pct:.1f}% exceeds limit {max_drawdown}%",
                value=drawdown_pct,
                limit=max_drawdown
            )
        elif drawdown_pct > max_drawdown * 0.8:
            return RiskCheck(
                name="drawdown",
                status=RiskCheckStatus.WARN,
                message=f"Drawdown {drawdown_pct:.1f}% approaching limit {max_drawdown}%",
                value=drawdown_pct,
                limit=max_drawdown
            )
        else:
            return RiskCheck(
                name="drawdown",
                status=RiskCheckStatus.PASS,
                message="Drawdown within limits",
                value=drawdown_pct,
                limit=max_drawdown
            )


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    engine = RiskEngine()
    
    # Test trade
    trade = {
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 100,
        "price": 150.0,
        "confidence": 0.85,
        "volatility": 0.25,
    }
    
    portfolio = {
        "total_value": 100000.0,
        "peak_value": 120000.0,
    }
    
    result = engine.evaluate_trade(trade, portfolio)
    
    print(f"Approved: {result.approved}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"\nChecks:")
    for check in result.checks:
        print(f"  {check.name}: {check.status.value} - {check.message}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
