#!/usr/bin/env python3
"""
Risk Gates System - Pre-Trade Risk Checks

Evaluates proposed trades against risk gates before execution.

Risk Gates:
- Position size limits (max % of portfolio)
- Daily loss limits (max drawdown per day)
- Concentration limits (max % in single symbol)
- Leverage limits
- Volatility gates (block trades in high volatility)

Usage:
    gates = RiskGates(config)
    result = gates.evaluate(trade, portfolio)
    if not result.passed:
        print(f"Trade blocked: {result.violations}")
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from loguru import logger


@dataclass
class TradeProposal:
    """Proposed trade"""
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    
    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class Portfolio:
    """Current portfolio state"""
    total_value: float
    positions: Dict[str, float]  # symbol -> value
    daily_pnl: float
    leverage: float


@dataclass
class GateViolation:
    """Risk gate violation"""
    gate_name: str
    severity: Literal["WARNING", "CRITICAL"]
    message: str
    current_value: float
    limit_value: float


@dataclass
class GateResult:
    """Risk gate evaluation result"""
    passed: bool
    violations: List[GateViolation]
    warnings: List[GateViolation]


class RiskGates:
    """Risk gates evaluator"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk gates
        
        Args:
            config: Risk limits configuration
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default risk limits"""
        return {
            "max_position_pct": 0.20,      # 20% max per position
            "max_daily_loss_pct": 0.05,    # 5% max daily loss
            "max_concentration_pct": 0.30,  # 30% max in single symbol
            "max_leverage": 2.0,            # 2x max leverage
            "max_volatility": 0.50          # 50% max volatility (annualized)
        }
    
    def evaluate(self, trade: TradeProposal, portfolio: Portfolio, volatility: Optional[float] = None) -> GateResult:
        """
        Evaluate trade against all risk gates
        
        Args:
            trade: Proposed trade
            portfolio: Current portfolio
            volatility: Symbol volatility (optional)
            
        Returns:
            GateResult with passed status and violations
        """
        violations = []
        warnings = []
        
        # Gate 1: Position size limit
        result = self._check_position_size(trade, portfolio)
        if result:
            if result.severity == "CRITICAL":
                violations.append(result)
            else:
                warnings.append(result)
        
        # Gate 2: Daily loss limit
        result = self._check_daily_loss(trade, portfolio)
        if result:
            if result.severity == "CRITICAL":
                violations.append(result)
            else:
                warnings.append(result)
        
        # Gate 3: Concentration limit
        result = self._check_concentration(trade, portfolio)
        if result:
            if result.severity == "CRITICAL":
                violations.append(result)
            else:
                warnings.append(result)
        
        # Gate 4: Leverage limit
        result = self._check_leverage(trade, portfolio)
        if result:
            if result.severity == "CRITICAL":
                violations.append(result)
            else:
                warnings.append(result)
        
        # Gate 5: Volatility gate
        if volatility is not None:
            result = self._check_volatility(trade, volatility)
            if result:
                if result.severity == "CRITICAL":
                    violations.append(result)
                else:
                    warnings.append(result)
        
        passed = len(violations) == 0
        
        if not passed:
            logger.warning(f"Trade blocked: {len(violations)} violations")
            for v in violations:
                logger.warning(f"  - {v.gate_name}: {v.message}")
        
        return GateResult(passed=passed, violations=violations, warnings=warnings)
    
    def _check_position_size(self, trade: TradeProposal, portfolio: Portfolio) -> Optional[GateViolation]:
        """Check position size limit"""
        max_pct = self.config["max_position_pct"]
        max_value = portfolio.total_value * max_pct
        
        if trade.value > max_value:
            return GateViolation(
                gate_name="Position Size Limit",
                severity="CRITICAL",
                message=f"Trade value ${trade.value:,.0f} exceeds {max_pct*100}% position limit (${max_value:,.0f})",
                current_value=trade.value,
                limit_value=max_value
            )
        
        # Warning at 80% of limit
        if trade.value > max_value * 0.8:
            return GateViolation(
                gate_name="Position Size Limit",
                severity="WARNING",
                message=f"Trade value ${trade.value:,.0f} is {trade.value/max_value*100:.0f}% of position limit",
                current_value=trade.value,
                limit_value=max_value
            )
        
        return None
    
    def _check_daily_loss(self, trade: TradeProposal, portfolio: Portfolio) -> Optional[GateViolation]:
        """Check daily loss limit"""
        max_loss_pct = self.config["max_daily_loss_pct"]
        max_loss = portfolio.total_value * max_loss_pct
        
        # If already at loss limit, block all trades
        if portfolio.daily_pnl < -max_loss:
            return GateViolation(
                gate_name="Daily Loss Limit",
                severity="CRITICAL",
                message=f"Daily loss ${-portfolio.daily_pnl:,.0f} exceeds {max_loss_pct*100}% limit (${max_loss:,.0f})",
                current_value=-portfolio.daily_pnl,
                limit_value=max_loss
            )
        
        # Warning at 80% of limit
        if portfolio.daily_pnl < -max_loss * 0.8:
            return GateViolation(
                gate_name="Daily Loss Limit",
                severity="WARNING",
                message=f"Daily loss ${-portfolio.daily_pnl:,.0f} is {-portfolio.daily_pnl/max_loss*100:.0f}% of limit",
                current_value=-portfolio.daily_pnl,
                limit_value=max_loss
            )
        
        return None
    
    def _check_concentration(self, trade: TradeProposal, portfolio: Portfolio) -> Optional[GateViolation]:
        """Check concentration limit"""
        max_pct = self.config["max_concentration_pct"]
        max_value = portfolio.total_value * max_pct
        
        # Calculate new position value after trade
        current_position = portfolio.positions.get(trade.symbol, 0)
        if trade.side == "BUY":
            new_position = current_position + trade.value
        else:
            new_position = max(0, current_position - trade.value)
        
        if new_position > max_value:
            return GateViolation(
                gate_name="Concentration Limit",
                severity="CRITICAL",
                message=f"Position in {trade.symbol} would be ${new_position:,.0f} ({new_position/portfolio.total_value*100:.1f}%), exceeds {max_pct*100}% limit",
                current_value=new_position,
                limit_value=max_value
            )
        
        # Warning at 80% of limit
        if new_position > max_value * 0.8:
            return GateViolation(
                gate_name="Concentration Limit",
                severity="WARNING",
                message=f"Position in {trade.symbol} would be {new_position/max_value*100:.0f}% of concentration limit",
                current_value=new_position,
                limit_value=max_value
            )
        
        return None
    
    def _check_leverage(self, trade: TradeProposal, portfolio: Portfolio) -> Optional[GateViolation]:
        """Check leverage limit"""
        max_leverage = self.config["max_leverage"]
        
        # Calculate new leverage after trade
        if trade.side == "BUY":
            new_leverage = portfolio.leverage + (trade.value / portfolio.total_value)
        else:
            new_leverage = portfolio.leverage
        
        if new_leverage > max_leverage:
            return GateViolation(
                gate_name="Leverage Limit",
                severity="CRITICAL",
                message=f"Leverage would be {new_leverage:.2f}x, exceeds {max_leverage}x limit",
                current_value=new_leverage,
                limit_value=max_leverage
            )
        
        # Warning at 80% of limit
        if new_leverage > max_leverage * 0.8:
            return GateViolation(
                gate_name="Leverage Limit",
                severity="WARNING",
                message=f"Leverage would be {new_leverage/max_leverage*100:.0f}% of limit",
                current_value=new_leverage,
                limit_value=max_leverage
            )
        
        return None
    
    def _check_volatility(self, trade: TradeProposal, volatility: float) -> Optional[GateViolation]:
        """Check volatility gate"""
        max_vol = self.config["max_volatility"]
        
        if volatility > max_vol:
            return GateViolation(
                gate_name="Volatility Gate",
                severity="CRITICAL",
                message=f"{trade.symbol} volatility {volatility*100:.1f}% exceeds {max_vol*100}% limit",
                current_value=volatility,
                limit_value=max_vol
            )
        
        # Warning at 80% of limit
        if volatility > max_vol * 0.8:
            return GateViolation(
                gate_name="Volatility Gate",
                severity="WARNING",
                message=f"{trade.symbol} volatility is {volatility/max_vol*100:.0f}% of limit",
                current_value=volatility,
                limit_value=max_vol
            )
        
        return None


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    gates = RiskGates()
    
    print("=== Risk Gates Tests ===\n")
    
    # Test portfolio
    portfolio = Portfolio(
        total_value=100000,
        positions={"AAPL": 15000, "GOOGL": 10000},
        daily_pnl=-3000,
        leverage=1.2
    )
    
    # Test 1: Normal trade (should pass)
    print("Test 1: Normal trade")
    trade = TradeProposal(symbol="MSFT", side="BUY", quantity=100, price=150)
    result = gates.evaluate(trade, portfolio, volatility=0.25)
    print(f"  Passed: {result.passed}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    # Test 2: Oversized position (should fail)
    print("\nTest 2: Oversized position")
    trade = TradeProposal(symbol="TSLA", side="BUY", quantity=500, price=250)
    result = gates.evaluate(trade, portfolio, volatility=0.30)
    print(f"  Passed: {result.passed}")
    if not result.passed:
        for v in result.violations:
            print(f"    - {v.message}")
    
    # Test 3: High volatility (should fail)
    print("\nTest 3: High volatility")
    trade = TradeProposal(symbol="GME", side="BUY", quantity=100, price=100)
    result = gates.evaluate(trade, portfolio, volatility=0.80)
    print(f"  Passed: {result.passed}")
    if not result.passed:
        for v in result.violations:
            print(f"    - {v.message}")
    
    print("\n✅ All tests complete")
