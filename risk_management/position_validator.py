#!/usr/bin/env python3
"""Position Validator - Validate trades against policies"""
from typing import Dict

class PositionValidator:
    def validate(self, trade: Dict, portfolio: Dict) -> Dict:
        """Validate proposed trade"""
        violations = []
        
        # Check portfolio constraints
        if trade["value"] > portfolio["total_value"] * 0.2:
            violations.append("Position too large")
        
        # Calculate risk metrics
        risk_score = trade["value"] / portfolio["total_value"]
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "risk_score": risk_score,
            "reasons": violations if violations else ["Trade passes all checks"]
        }

if __name__ == "__main__":
    validator = PositionValidator()
    print("✅ Position Validator ready")
