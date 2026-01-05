#!/usr/bin/env python3
"""
Alert Evaluator - Evaluate Alert Conditions

Evaluates alert conditions against current market data and triggers
notifications when conditions are met.

Supported Conditions:
- above: Value is above threshold
- below: Value is below threshold
- crosses_above: Value crosses above threshold (was below, now above)
- crosses_below: Value crosses below threshold (was above, now below)
- equals: Value equals threshold (within tolerance)

Usage:
    evaluator = AlertEvaluator()
    triggered = evaluator.evaluate_condition(current_value, condition, threshold)
"""

from typing import Dict, Optional, Literal
from loguru import logger


ConditionType = Literal["above", "below", "crosses_above", "crosses_below", "equals"]


class AlertEvaluator:
    """Evaluate alert conditions"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize alert evaluator
        
        Args:
            tolerance: Tolerance for 'equals' condition (default: 1%)
        """
        self.tolerance = tolerance
        self.previous_values: Dict[str, float] = {}
    
    def evaluate_condition(
        self,
        current_value: float,
        condition: ConditionType,
        threshold: float,
        alert_id: Optional[str] = None
    ) -> bool:
        """
        Evaluate if condition is met
        
        Args:
            current_value: Current value to evaluate
            condition: Condition type
            threshold: Threshold value
            alert_id: Alert identifier (for tracking previous values)
            
        Returns:
            True if condition is met, False otherwise
        """
        try:
            if condition == "above":
                return current_value > threshold
            
            elif condition == "below":
                return current_value < threshold
            
            elif condition == "equals":
                # Within tolerance
                return abs(current_value - threshold) <= abs(threshold * self.tolerance)
            
            elif condition == "crosses_above":
                # Need previous value
                if alert_id and alert_id in self.previous_values:
                    prev_value = self.previous_values[alert_id]
                    result = prev_value <= threshold and current_value > threshold
                    self.previous_values[alert_id] = current_value
                    return result
                else:
                    # First evaluation - store current value
                    if alert_id:
                        self.previous_values[alert_id] = current_value
                    return False
            
            elif condition == "crosses_below":
                # Need previous value
                if alert_id and alert_id in self.previous_values:
                    prev_value = self.previous_values[alert_id]
                    result = prev_value >= threshold and current_value < threshold
                    self.previous_values[alert_id] = current_value
                    return result
                else:
                    # First evaluation - store current value
                    if alert_id:
                        self.previous_values[alert_id] = current_value
                    return False
            
            else:
                logger.warning(f"Unknown condition type: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def clear_history(self, alert_id: str):
        """Clear previous value for an alert"""
        if alert_id in self.previous_values:
            del self.previous_values[alert_id]


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    evaluator = AlertEvaluator()
    
    print("=== Alert Evaluator Tests ===\n")
    
    # Test 1: Above
    print("Test 1: Above condition")
    print(f"  100 > 90: {evaluator.evaluate_condition(100, 'above', 90)}")  # True
    print(f"  80 > 90: {evaluator.evaluate_condition(80, 'above', 90)}")   # False
    
    # Test 2: Below
    print("\nTest 2: Below condition")
    print(f"  80 < 90: {evaluator.evaluate_condition(80, 'below', 90)}")   # True
    print(f"  100 < 90: {evaluator.evaluate_condition(100, 'below', 90)}") # False
    
    # Test 3: Equals
    print("\nTest 3: Equals condition (tolerance=1%)")
    print(f"  100 ≈ 100: {evaluator.evaluate_condition(100, 'equals', 100)}")    # True
    print(f"  100.5 ≈ 100: {evaluator.evaluate_condition(100.5, 'equals', 100)}") # True
    print(f"  102 ≈ 100: {evaluator.evaluate_condition(102, 'equals', 100)}")    # False
    
    # Test 4: Crosses above
    print("\nTest 4: Crosses above condition")
    alert_id = "test_alert_1"
    print(f"  First: 80 crosses 90: {evaluator.evaluate_condition(80, 'crosses_above', 90, alert_id)}")  # False (first)
    print(f"  Second: 85 crosses 90: {evaluator.evaluate_condition(85, 'crosses_above', 90, alert_id)}") # False (still below)
    print(f"  Third: 95 crosses 90: {evaluator.evaluate_condition(95, 'crosses_above', 90, alert_id)}")  # True (crossed!)
    print(f"  Fourth: 100 crosses 90: {evaluator.evaluate_condition(100, 'crosses_above', 90, alert_id)}") # False (already above)
    
    # Test 5: Crosses below
    print("\nTest 5: Crosses below condition")
    alert_id = "test_alert_2"
    print(f"  First: 100 crosses 90: {evaluator.evaluate_condition(100, 'crosses_below', 90, alert_id)}")  # False (first)
    print(f"  Second: 95 crosses 90: {evaluator.evaluate_condition(95, 'crosses_below', 90, alert_id)}")   # False (still above)
    print(f"  Third: 85 crosses 90: {evaluator.evaluate_condition(85, 'crosses_below', 90, alert_id)}")    # True (crossed!)
    print(f"  Fourth: 80 crosses 90: {evaluator.evaluate_condition(80, 'crosses_below', 90, alert_id)}")   # False (already below)
    
    print("\n✅ All tests complete")
