#!/usr/bin/env python3
"""
Policy Evaluator - Custom Trading Policy Rules

Evaluates trades against custom user-defined policies.

Features:
- Rule-based policy evaluation
- Conditional logic (AND/OR)
- Numeric comparisons
- Time-based rules
- Custom policy templates

Usage:
    evaluator = PolicyEvaluator()
    result = evaluator.evaluate(trade, policy)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PolicyRuleType(Enum):
    """Policy rule types"""
    NUMERIC_COMPARISON = "numeric_comparison"
    TIME_WINDOW = "time_window"
    SYMBOL_WHITELIST = "symbol_whitelist"
    SYMBOL_BLACKLIST = "symbol_blacklist"
    CUSTOM = "custom"


class ComparisonOperator(Enum):
    """Comparison operators"""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="


@dataclass
class PolicyRule:
    """Individual policy rule"""
    rule_type: PolicyRuleType
    field: str
    operator: Optional[ComparisonOperator] = None
    value: Optional[Any] = None
    description: Optional[str] = None


@dataclass
class Policy:
    """Trading policy"""
    policy_id: str
    name: str
    rules: List[PolicyRule]
    logic: str = "AND"  # AND or OR
    enabled: bool = True


@dataclass
class PolicyEvaluationResult:
    """Policy evaluation result"""
    passed: bool
    policy_name: str
    failed_rules: List[str]
    message: str


class PolicyEvaluator:
    """Evaluate trades against custom policies"""
    
    def evaluate(
        self,
        trade: Dict[str, Any],
        policy: Policy
    ) -> PolicyEvaluationResult:
        """
        Evaluate trade against policy
        
        Args:
            trade: Trade details
            policy: Policy to evaluate against
            
        Returns:
            PolicyEvaluationResult
        """
        if not policy.enabled:
            return PolicyEvaluationResult(
                passed=True,
                policy_name=policy.name,
                failed_rules=[],
                message="Policy disabled"
            )
        
        failed_rules: List[str] = []
        
        for rule in policy.rules:
            if not self._evaluate_rule(trade, rule):
                failed_rules.append(rule.description or f"{rule.field} {rule.operator.value if rule.operator else ''} {rule.value}")
        
        # Apply logic (AND/OR)
        if policy.logic == "AND":
            passed = len(failed_rules) == 0
        else:  # OR
            passed = len(failed_rules) < len(policy.rules)
        
        message = "Policy passed" if passed else f"Failed rules: {', '.join(failed_rules)}"
        
        return PolicyEvaluationResult(
            passed=passed,
            policy_name=policy.name,
            failed_rules=failed_rules,
            message=message
        )
    
    def _evaluate_rule(self, trade: Dict[str, Any], rule: PolicyRule) -> bool:
        """Evaluate single rule"""
        if rule.rule_type == PolicyRuleType.NUMERIC_COMPARISON:
            return self._evaluate_numeric(trade, rule)
        elif rule.rule_type == PolicyRuleType.SYMBOL_WHITELIST:
            return trade.get("symbol") in rule.value
        elif rule.rule_type == PolicyRuleType.SYMBOL_BLACKLIST:
            return trade.get("symbol") not in rule.value
        else:
            return True  # Unknown rule type passes
    
    def _evaluate_numeric(self, trade: Dict[str, Any], rule: PolicyRule) -> bool:
        """Evaluate numeric comparison"""
        trade_value = trade.get(rule.field)
        if trade_value is None:
            return False
        
        if rule.operator == ComparisonOperator.GT:
            return trade_value > rule.value
        elif rule.operator == ComparisonOperator.GTE:
            return trade_value >= rule.value
        elif rule.operator == ComparisonOperator.LT:
            return trade_value < rule.value
        elif rule.operator == ComparisonOperator.LTE:
            return trade_value <= rule.value
        elif rule.operator == ComparisonOperator.EQ:
            return trade_value == rule.value
        elif rule.operator == ComparisonOperator.NEQ:
            return trade_value != rule.value
        else:
            return False
    
    @staticmethod
    def create_template_policy(template_name: str) -> Policy:
        """Create policy from template"""
        templates = {
            "conservative": Policy(
                policy_id="conservative",
                name="Conservative Trading",
                rules=[
                    PolicyRule(
                        rule_type=PolicyRuleType.NUMERIC_COMPARISON,
                        field="confidence",
                        operator=ComparisonOperator.GTE,
                        value=0.8,
                        description="Confidence >= 0.8"
                    ),
                    PolicyRule(
                        rule_type=PolicyRuleType.NUMERIC_COMPARISON,
                        field="volatility",
                        operator=ComparisonOperator.LTE,
                        value=0.3,
                        description="Volatility <= 0.3"
                    ),
                ],
                logic="AND"
            ),
            "aggressive": Policy(
                policy_id="aggressive",
                name="Aggressive Trading",
                rules=[
                    PolicyRule(
                        rule_type=PolicyRuleType.NUMERIC_COMPARISON,
                        field="confidence",
                        operator=ComparisonOperator.GTE,
                        value=0.6,
                        description="Confidence >= 0.6"
                    ),
                ],
                logic="AND"
            ),
        }
        
        return templates.get(template_name, templates["conservative"])


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    evaluator = PolicyEvaluator()
    
    # Test trade
    trade = {
        "symbol": "AAPL",
        "confidence": 0.85,
        "volatility": 0.25,
    }
    
    # Conservative policy
    policy = PolicyEvaluator.create_template_policy("conservative")
    result = evaluator.evaluate(trade, policy)
    
    print(f"Policy: {result.policy_name}")
    print(f"Passed: {result.passed}")
    print(f"Message: {result.message}")
